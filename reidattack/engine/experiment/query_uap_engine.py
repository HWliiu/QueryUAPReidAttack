import os
import random
from typing import List, Optional, Union

import numpy as np
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


class QueryUAPAttackEngine(BaseEngine):
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
            algorithm: str = "query_uap") -> None:
        super().__init__(
            train_dataloader, query_dataloader, gallery_dataloader,
            accelerator, agent_models, target_model, segment_model, algorithm)
        torch.backends.cudnn.benchmark = False
        self.image_size = image_size
        self.uap = torch.rand(
            (1, 3, *image_size),
            device=self.accelerator.device) * 2 * (4 / 255) - (4 / 255)

        self.momentum = torch.zeros_like(self.uap)
        self.step_size = 10 / 255
        self.decay = 1

    def _get_mask(self, imgs, background_weight=0.5):
        mask = self.segment_model(imgs)
        tau = .1
        mask = torch.softmax(self.segment_model(imgs) / tau, dim=1)[:, 1:]
        # dilate mask
        mask = K.morphology.dilation(mask, kernel=mask.new_ones(15, 15))
        # blur mask
        mask = K.filters.box_blur(mask, kernel_size=(7, 7))
        # add inv mask
        mask_inv = 1 - mask
        mask = mask + mask_inv * background_weight
        return mask

    def _draw_coords(self, imgs, coords_num, background_weight=0.05):
        masks = self._get_mask(imgs, background_weight=background_weight)
        masks_flat = repeat(masks, 'b c h w->b (3 c h w)')
        coords = torch.multinomial(masks_flat, coords_num)
        draw_masks = torch.zeros_like(masks_flat, dtype=torch.int64)
        for i in range(len(draw_masks)):
            draw_masks[i, coords[i]] = 1
        draw_masks = draw_masks.view_as(imgs)
        return draw_masks

    def _query_uap_training_step(self, imgs, pids, camids):
        coords_num = 64 * 32 * 3
        fd = 0.01
        coords_batch = 128

        feats = self._reid_model_forward(self.target_model, imgs, camids)
        masks = self._draw_coords(imgs, coords_num)

        for i in range(len(imgs)):
            img, feat = imgs[i:i + 1], feats[i:i + 1]
            mask, camid = masks[i:i + 1], camids[i:i + 1]

            adv_img = torch.clamp(img + self.uap, 0, 1)
            adv_feat = self._reid_model_forward(
                self.target_model, adv_img, camid)
            loss1 = (F.normalize(adv_feat) * F.normalize(feat)).sum(-1)
            coords = mask.nonzero(as_tuple=True)

            grads = torch.zeros(coords_num, device=self.uap.device)
            for j in range(coords_num // coords_batch):
                adv_imgs_fd = repeat(adv_img, 'b c h w->(n b) c h w',
                                     n=coords_batch).clone()
                coords_fd_batch = (
                    torch.arange(coords_batch, device=coords[0].device),
                    coords[1][j * coords_batch: (j + 1) * coords_batch],
                    coords[2][j * coords_batch: (j + 1) * coords_batch],
                    coords[3][j * coords_batch: (j + 1) * coords_batch])
                adv_imgs_fd[coords_fd_batch] += torch.ones(
                    coords_batch, device=self.uap.device) * fd

                camids_fd = repeat(camid, 'b->(n b)', n=coords_batch).clone()
                adv_feats_fd = self._reid_model_forward(
                    self.target_model, adv_imgs_fd, camids_fd)

                loss2 = (F.normalize(adv_feats_fd) * F.normalize(repeat(feat, 'b c->(n b) c',
                                                                        n=coords_batch))).sum(-1)
                grad = (loss2 - loss1) / fd
                grads[j * coords_batch: (j + 1) * coords_batch] += grad

            blur_momentum = K.filters.gaussian_blur2d(
                self.momentum, kernel_size=(3, 3), sigma=(0.4, 0.4))
            # blur_momentum = self.momentum
            grads = blur_momentum[coords] * self.decay + \
                grads / torch.linalg.vector_norm(grads, ord=1).clamp(min=1e-8)
            self.momentum[coords] = grads

            self.uap[coords] -= self.step_size * grads.sign()
            self.uap = torch.clamp(self.uap, -self.epsilon, self.epsilon)

        # for display loss only
        adv_imgs = torch.clamp(imgs + self.uap, 0, 1)
        adv_feats = self._reid_model_forward(
            self.target_model, adv_imgs, camids)
        loss = (F.normalize(adv_feats) * F.normalize(feats)).sum(-1).mean()
        return loss

    def training_step(self, batch, batch_idx):
        imgs, pids, camids, imgs_path, _ = batch.values()

        loss = self._query_uap_training_step(imgs, pids, camids)
        # results = self.test()
        # self.logger.info(f"batch:{batch_idx} map:{results[0]}")

        return {'loss': loss.detach()}

    def val_step(self, batch, batch_idx, is_query=True):
        imgs, pids, camids, imgs_path, _ = batch.values()

        if is_query:
            uap = torch.clamp(self.uap, -self.epsilon, self.epsilon)
            # # project uap
            # uap = F.hardtanh(2 * uap, -self.epsilon, self.epsilon)
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
            self.uap,
            f'{self.log_dir}/{self.target_model.name}_uap_map={map}.pth')

    def _reid_model_forward(self, model, imgs, camids):
        if 'transreid' in model.name:
            feats = model(imgs, cam_label=camids)
        else:
            feats = model(imgs)
        return feats


@ENGINE_REGISTRY.register()
def query_uap(**trainer_params):
    return QueryUAPAttackEngine(**trainer_params)
