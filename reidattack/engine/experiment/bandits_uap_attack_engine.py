import os
import random
from typing import List, Optional, Union

import numpy as np
from scipy import stats as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.utils import save_image
from einops import rearrange, reduce, repeat

import accelerate
from accelerate.utils import extract_model_from_parallel
import kornia as K

from engine.base_engine import BaseEngine
from . import ENGINE_REGISTRY
from utils import mkdir_if_missing


class BanditsUAPAttackEngine(BaseEngine):
    def __init__(
            self,
            train_dataloader: DataLoader,
            query_dataloader: DataLoader,
            gallery_dataloader: DataLoader,
            accelerator: accelerate.Accelerator,
            agent_models: List[nn.Module],
            target_model: nn.Module,
            segment_model: nn.Module,
            image_size: List[float],
            algorithm: str = "bandits_uap") -> None:
        super().__init__(
            train_dataloader, query_dataloader, gallery_dataloader,
            accelerator, agent_models, target_model, segment_model, algorithm)
        self.image_size = image_size
        self.uap = torch.rand(
            (1, 3, 256, 128),
            device=self.accelerator.device) * 2 * (4 / 255) - (4 / 255)

        self.prior_size = (64, 32)
        self.prior = torch.zeros(
            1, 3, *self.prior_size, device=self.uap.device)

    def bandits_uap_attack(self, imgs, camids):
        max_queries = 6146

        exploration = 1.
        fd_eta = 0.1
        step_size = 0.01

        dim = self.prior.nelement()
        adv_imgs = torch.clamp(imgs + self.uap, 0, 1)

        feats = self._reid_model_forward(self.target_model, imgs, camids)
        for _ in range(max_queries // 2):
            exp_noise = exploration * torch.randn_like(self.prior) / (dim**0.5)
            q1 = F.interpolate(
                self.prior + exp_noise, size=adv_imgs.shape[-2:])
            q1_norm = torch.linalg.vector_norm(q1, dim=(1, 2, 3), keepdim=True)
            input1 = adv_imgs + fd_eta * q1 / q1_norm
            adv_feats1 = self._reid_model_forward(
                self.target_model, input1, camids)
            l1 = (F.normalize(adv_feats1) * F.normalize(feats)).sum(dim=1).mean()

            q2 = F.interpolate(
                self.prior - exp_noise, size=adv_imgs.shape[-2:])
            q2_norm = torch.linalg.vector_norm(q2, dim=(1, 2, 3), keepdim=True)
            input2 = adv_imgs + fd_eta * q2 / q2_norm
            adv_feats2 = self._reid_model_forward(
                self.target_model, input2, camids)
            l2 = (F.normalize(adv_feats2) * F.normalize(feats)).sum(dim=1).mean()
            est_deriv = (l1 - l2) / (fd_eta * exploration)
            est_grad = est_deriv.view(-1, 1, 1, 1) * exp_noise

            self.prior = self.prior_step(self.prior, est_grad)

            grad = F.interpolate(
                self.prior, size=adv_imgs.shape[-2:])

            self.uap -= step_size * grad.sign()
            self.uap = torch.clamp(
                self.uap, min=-self.epsilon, max=self.epsilon)
            adv_imgs = torch.clamp(imgs + self.uap, 0, 1)
        return adv_imgs

    def prior_step(self, prior, est_grad, lr=100):
        real_prior = (prior + 1) / 2  # from 0 center to 0.5 center
        pos = real_prior * torch.exp(lr * est_grad)
        neg = (1 - real_prior) * torch.exp(-lr * est_grad)
        new_prior = pos / (pos + neg)
        return new_prior * 2 - 1

    def _bandits_uap_training_step(self, imgs, pids, camids):
        adv_imgs = self.bandits_uap_attack(imgs, camids)

        # for display loss only
        adv_feats = self._reid_model_forward(
            self.target_model, adv_imgs, camids)
        feats = self._reid_model_forward(self.target_model, imgs, camids)
        loss = (F.normalize(adv_feats) * F.normalize(feats)).sum(-1).mean()
        return loss

    def training_step(self, batch, batch_idx):
        imgs, pids, camids, imgs_path, _ = batch.values()

        loss = self._bandits_uap_training_step(imgs, pids, camids)

        return {'loss': loss.detach()}

    def val_step(self, batch, batch_idx, is_query=True):
        imgs, pids, camids, imgs_path, _ = batch.values()

        if is_query:
            uap = torch.clamp(self.uap, -self.epsilon, self.epsilon)
            adv_imgs = torch.clamp(imgs + uap, 0, 1)


            if batch_idx == 1 and self.accelerator.is_main_process:
                self._make_log_dir_if_missing(imgs_path[0].split(os.sep)[-3])
                save_image(
                    adv_imgs[: 16],
                    f'{self.log_dir}/{self.target_model.name}_adv_imgs.png',
                    pad_value=1)
                save_image(
                    adv_imgs[: 16] - imgs[: 16],
                    f'{self.log_dir}/{self.target_model.name}_delta.png',
                    normalize=True, pad_value=1)
            imgs = adv_imgs

        feats = self._reid_model_forward(self.target_model, imgs, camids)

        return feats, pids, camids

    def save_state(self, epoch, map, is_best=False):
        torch.save(
            self.uap, f'{self.log_dir}/{self.target_model.name}_uap.pth')

    def _reid_model_forward(self, model, imgs, camids):
        if 'transreid' in model.name:
            feats = model(imgs, cam_label=camids)
        else:
            feats = model(imgs)
        return feats


@ENGINE_REGISTRY.register()
def bandits_uap(**trainer_params):
    return BanditsUAPAttackEngine(**trainer_params)
