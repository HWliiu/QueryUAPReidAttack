import os
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import accelerate
import kornia as K
from einops import rearrange, reduce, repeat

from engine.base_trainer import BaseTrainer
from .no_attack_evaluator import NoAttackEvaluator
from . import TRAINER_REGISTRY
from utils import mkdir_if_missing


class BanditsAttackTrainer(NoAttackEvaluator):
    def __init__(
            self,
            train_dataloader: DataLoader,
            query_dataloader: DataLoader,
            gallery_dataloader: DataLoader,
            accelerator: accelerate.Accelerator,
            agent_models: nn.Module,
            target_model: nn.Module,
            segment_model: nn.Module,
            algorithm: str,
            use_normalized: bool = True,
            normalize_mean: Optional[List[float]] = None,
            normalize_std: Optional[List[float]] = None) -> None:
        super().__init__(
            train_dataloader, query_dataloader, gallery_dataloader,
            accelerator, agent_models, target_model, segment_model, algorithm,
            use_normalized, normalize_mean, normalize_std)

    def bandits_attack(self, imgs, pids, camids):
        max_queries = 500
        prior_size = (64, 32)
        exploration = 1.
        fd_eta = 0.1
        step_size = 0.01
        prior = torch.zeros(
            imgs.shape[0], 3, *prior_size, device=imgs.device)
        dim = prior.nelement() / imgs.shape[0]
        adv_imgs = imgs.clone()

        feats = self._reid_model_forward(self.target_model, imgs, pids, camids)
        for _ in range(max_queries // 2):
            exp_noise = exploration * torch.randn_like(prior) / (dim**0.5)
            q1 = F.interpolate(
                prior + exp_noise, size=adv_imgs.shape[-2:])
            q1_norm = torch.linalg.vector_norm(q1, dim=(1, 2, 3), keepdim=True)
            input1 = adv_imgs + fd_eta * q1 / q1_norm
            adv_feats1 = self._reid_model_forward(
                self.target_model, input1, pids, camids)
            l1 = (F.normalize(adv_feats1) * F.normalize(feats)).sum(dim=1)

            q2 = F.interpolate(
                prior - exp_noise, size=adv_imgs.shape[-2:])
            q2_norm = torch.linalg.vector_norm(q2, dim=(1, 2, 3), keepdim=True)
            input2 = adv_imgs + fd_eta * q2 / q2_norm
            adv_feats2 = self._reid_model_forward(
                self.target_model, input2, pids, camids)
            l2 = (F.normalize(adv_feats2) * F.normalize(feats)).sum(dim=1)
            est_deriv = (l1 - l2) / (fd_eta * exploration)
            est_grad = est_deriv.view(-1, 1, 1, 1) * exp_noise

            prior = self.prior_step(prior, est_grad)

            grad = F.interpolate(
                prior, size=adv_imgs.shape[-2:])

            adv_imgs -= step_size * grad.sign()
            delta = torch.clamp(adv_imgs - imgs, min=-10 / 255, max=10 / 255)
            adv_imgs = torch.clamp(imgs + delta, min=0, max=1)
        return adv_imgs

    def prior_step(self, prior, est_grad, lr=100):
        real_prior = (prior + 1) / 2  # from 0 center to 0.5 center
        pos = real_prior * torch.exp(lr * est_grad)
        neg = (1 - real_prior) * torch.exp(-lr * est_grad)
        new_prior = pos / (pos + neg)
        return new_prior * 2 - 1

    def val_step(self, batch, batch_idx, is_query=True):
        imgs, pids, camids, imgs_path, _ = batch.values()

        if is_query:
            adv_imgs = self.bandits_attack(imgs, pids, camids)

            self._make_log_dir_if_missing(imgs_path[0].split(os.sep)[-3])
            cache_path = os.path.join(
                self.log_dir, f'{self.target_model.name}')
            mkdir_if_missing(cache_path)

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
            # cache adv_imgs
            torch.save(adv_imgs, f'{cache_path}/batch_{batch_idx}.pth')

        # extract_features
        feats = self._reid_model_forward(self.target_model, imgs, pids, camids)

        if self._use_fliplr:
            imgs_fliplr = T.functional.hflip(imgs)
            feats_fliplr = self.target_model(imgs_fliplr)
            feats = (feats + feats_fliplr) / 2.

        return feats, pids, camids

    def _reid_model_forward(self, model, imgs, pids, camids):
        if 'transreid' in model.name:
            feats = model(imgs, cam_label=camids)
        else:
            feats = model(imgs)
        return feats


@TRAINER_REGISTRY.register()
def bandits(**trainer_params):
    return BanditsAttackTrainer(**trainer_params)
