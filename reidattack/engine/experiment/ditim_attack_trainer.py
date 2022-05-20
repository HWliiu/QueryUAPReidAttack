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


class DITIMAttackTrainer(NoAttackEvaluator):
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

    def _input_diversity(self, input1, input2):
        img_size = input1.shape[-1]
        resize_rate = 0.9
        img_resize = int(img_size * resize_rate)

        if resize_rate < 1:
            img_size = img_resize
            img_resize = input1.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize,
                            size=(1,), dtype=torch.int32)
        ratio = input1.shape[-2] // input1.shape[-1]
        rescaled1 = F.interpolate(input1, size=[rnd * ratio, rnd],
                                  mode='bilinear', align_corners=False)
        rescaled2 = F.interpolate(input2, size=[rnd * ratio, rnd],
                                  mode='bilinear', align_corners=False)
        h_rem = (img_resize - rnd) * ratio
        w_rem = img_resize - rnd
        pad_top = torch.randint(
            low=0, high=h_rem.item(),
            size=(1,),
            dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(
            low=0, high=w_rem.item(),
            size=(1,),
            dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded1 = F.pad(
            rescaled1,
            [pad_left.item(),
             pad_right.item(),
             pad_top.item(),
             pad_bottom.item()],
            value=0)
        padded2 = F.pad(
            rescaled2,
            [pad_left.item(),
             pad_right.item(),
             pad_top.item(),
             pad_bottom.item()],
            value=0)
        diversity_prob = 0.7
        return (padded1, padded2) if torch.rand(1) < diversity_prob else (input1, input2)

    @torch.enable_grad()
    def di_tim(self, imgs, pids, camids):
        agent_model = self.agent_models[0]

        adv_imgs = imgs.clone()
        momentum = 0.
        decay = 1.
        step_size = 1.6 / 255
        for _ in range(10):
            adv_imgs.requires_grad_(True)
            imgs_pad, adv_imgs_pad = self._input_diversity(
                imgs, adv_imgs)
            feats = self._reid_model_forward(
                agent_model, imgs_pad, pids, camids)

            adv_feats = self._reid_model_forward(
                agent_model, adv_imgs_pad, pids, camids)
            loss = (F.normalize(adv_feats)
                    * F.normalize(feats)).sum(dim=1).mean()
            grad = torch.autograd.grad(loss, adv_imgs)[0]

            grad = K.filters.gaussian_blur2d(grad, (7, 7), (3, 3))

            grad = momentum * decay + grad / torch.mean(
                torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            momentum = grad

            adv_imgs.detach_()
            adv_imgs -= step_size * grad.sign()
            delta = torch.clamp(
                adv_imgs - imgs, min=-self.epsilon, max=self.epsilon)
            adv_imgs = torch.clamp(imgs + delta, min=0, max=1)
        return adv_imgs

    def val_step(self, batch, batch_idx, is_query=True):
        imgs, pids, camids, imgs_path, _ = batch.values()

        if is_query:
            adv_imgs = self.di_tim(imgs, pids, camids)

            self._make_log_dir_if_missing(imgs_path[0].split(os.sep)[-3])
            cache_path = os.path.join(
                self.log_dir, f'{self.agent_models[0].name}')
            mkdir_if_missing(cache_path)

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
def ditim(**trainer_params):
    return DITIMAttackTrainer(**trainer_params)
