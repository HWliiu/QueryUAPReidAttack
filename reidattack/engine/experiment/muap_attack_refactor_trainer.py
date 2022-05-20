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
from layers.losses import APLoss, TVLoss
from . import TRAINER_REGISTRY


class MUAPAttackRefactorTrainer(BaseTrainer):
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
            algorithm: str = "muap_refactor",
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

        self.momentum = torch.zeros_like(self.uap)
        self.step_size = 2 / 255
        self.decay = 0.5

    def configure_criterions(self):
        self.ap_loss = APLoss()
        self.tv_loss = TVLoss()

    def _muap_training_step(self, imgs, pids, camids):
        agent_model = self.agent_models[0]

        feats = self._reid_model_forward(agent_model, imgs, pids, camids)

        self.uap.requires_grad_(True)
        adv_imgs = torch.clamp(imgs + self.uap, 0, 1)
        adv_feats = self._reid_model_forward(
            agent_model, adv_imgs, pids, camids)

        # loss = self.ap_loss(adv_feats, feats,
        #                     pids) + 0.5 * self.tv_loss(self.uap / self.epsilon)
        # same like original paper code (original paper code adv_imgs is after normalized)zs, convergence very slow
        loss = self.ap_loss(adv_feats, feats,
                            pids) + 200 * self.tv_loss(adv_imgs)

        grad = torch.autograd.grad(loss, self.uap)[0]

        grad = self.momentum * self.decay + grad / torch.linalg.vector_norm(
            grad, ord=1, dim=(1, 2, 3), keepdim=True)
        self.momentum = grad

        self.uap.detach_()
        self.uap -= self.step_size * grad.sign()
        # project to ball
        self.uap = torch.clamp(self.uap, -self.epsilon, self.epsilon)

        return loss

    def training_step(self, batch, batch_idx):
        imgs, pids, camids, imgs_path, _ = batch.values()

        loss = self._muap_training_step(imgs, pids, camids)

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
                    f'{self.log_dir}/{self.agent_models[0].name}_adv_imgs.png',
                    pad_value=1)
                save_image(
                    adv_imgs[: 16] - imgs[: 16],
                    f'{self.log_dir}/{self.agent_models[0].name}_delta.png',
                    normalize=True, pad_value=1)
            imgs = adv_imgs

        feats = self._reid_model_forward(self.target_model, imgs, pids, camids)
        if self._use_fliplr:
            imgs_fliplr = T.functional.hflip(imgs)
            feats_fliplr = self.target_model(imgs_fliplr)
            feats = (feats + feats_fliplr) / 2.

        return feats, pids, camids

    def save_model(self, epoch, map, is_best=False):
        torch.save(
            self.uap, f'{self.log_dir}/{self.agent_models[0].name}_uap.pth')

    def _reid_model_forward(self, model, imgs, pids, camids):
        if 'transreid' in model.name:
            feats = model(imgs, cam_label=camids)
        else:
            feats = model(imgs)
        return feats


@TRAINER_REGISTRY.register()
def muap_refactor(**trainer_params):
    return MUAPAttackRefactorTrainer(**trainer_params)
