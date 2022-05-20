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

from engine.base_trainer import BaseTrainer
from . import TRAINER_REGISTRY
from utils import mkdir_if_missing


class RGFUAPAttackTrainer(BaseTrainer):
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
            algorithm: str = "rgf_uap",
            use_normalized: bool = True,
            normalize_mean: Optional[List[float]] = None,
            normalize_std: Optional[List[float]] = None) -> None:
        super().__init__(
            train_dataloader, query_dataloader, gallery_dataloader,
            accelerator, agent_models, target_model, segment_model, algorithm,
            use_normalized, normalize_mean, normalize_std)
        self.image_size = image_size
        self.uap = torch.rand(
            (1, 3, 256, 128),
            device=self.accelerator.device) * 2 * (4 / 255) - (4 / 255)

        self.momentum = 0.

    def rgf_uap_attack(self, imgs, camids):
        max_queries = 6146

        fd_eta = 0.1
        step_size = 0.01
        decay = 0.9

        adv_imgs = torch.clamp(imgs + self.uap, 0, 1)
        feats = self._reid_model_forward(self.target_model, imgs, camids)
        for _ in range(max_queries // 2):
            exp_noise = torch.randn_like(self.uap)
            exp_noise = exp_noise / torch.norm(exp_noise)

            input1 = adv_imgs + fd_eta * exp_noise
            adv_feats1 = self._reid_model_forward(
                self.target_model, input1, camids)
            l1 = (F.normalize(adv_feats1) * F.normalize(feats)).sum(dim=1).mean()

            input2 = adv_imgs
            adv_feats2 = self._reid_model_forward(
                self.target_model, input2, camids)
            l2 = (F.normalize(adv_feats2) * F.normalize(feats)).sum(dim=1).mean()
            est_deriv = (l1 - l2) / fd_eta
            grad = est_deriv.view(-1, 1, 1, 1) * exp_noise

            grad = self.momentum * decay + grad / torch.linalg.vector_norm(
                grad, ord=1, dim=(1, 2, 3), keepdim=True)
            self.momentum = grad
            self.uap -= step_size * grad.sign()

            self.uap = torch.clamp(
                self.uap, min=-self.epsilon, max=self.epsilon)
            adv_imgs = torch.clamp(imgs + self.uap, 0, 1)
        return adv_imgs

    def _rgf_uap_training_step(self, imgs, pids, camids):
        adv_imgs = self.rgf_uap_attack(imgs, camids)

        # for display loss only
        adv_feats = self._reid_model_forward(
            self.target_model, adv_imgs, camids)
        feats = self._reid_model_forward(self.target_model, imgs, camids)
        loss = (F.normalize(adv_feats) * F.normalize(feats)).sum(-1).mean()
        return loss

    def training_step(self, batch, batch_idx):
        imgs, pids, camids, imgs_path, _ = batch.values()

        loss = self._rgf_uap_training_step(imgs, pids, camids)

        return {'loss': loss.detach()}

    def val_step(self, batch, batch_idx, is_query=True):
        imgs, pids, camids, imgs_path, _ = batch.values()

        if is_query:
            uap = torch.clamp(self.uap, -self.epsilon, self.epsilon)
            adv_imgs = torch.clamp(imgs + uap, 0, 1)

            self._make_log_dir_if_missing(imgs_path[0].split(os.sep)[-3])

            if batch_idx == 1 and self.accelerator.is_main_process:
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
        if self._use_fliplr:
            imgs_fliplr = T.functional.hflip(imgs)
            feats_fliplr = self.target_model(imgs_fliplr)
            feats = (feats + feats_fliplr) / 2.

        return feats, pids, camids

    def save_model(self, epoch, map, is_best=False):
        torch.save(
            self.uap, f'{self.log_dir}/{self.target_model.name}_uap.pth')

    def _reid_model_forward(self, model, imgs, camids):
        if 'transreid' in model.name:
            feats = model(imgs, cam_label=camids)
        else:
            feats = model(imgs)
        return feats


@TRAINER_REGISTRY.register()
def rgf_uap(**trainer_params):
    return RGFUAPAttackTrainer(**trainer_params)
