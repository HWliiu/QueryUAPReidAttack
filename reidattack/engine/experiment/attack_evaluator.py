import os
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import accelerate
import kornia as K
from einops import repeat

from engine.base_trainer import BaseTrainer
from .no_attack_evaluator import NoAttackEvaluator
from . import TRAINER_REGISTRY
from utils import mkdir_if_missing


class AttackEvaluator(NoAttackEvaluator):
    # For UAP
    # def val_step(self, batch, batch_idx, is_query=True):
    #     imgs, pids, camids, imgs_path, _ = batch.values()

    #     uap_path = 'logs/muap/dukemtmc-reid/inceptionv3_bot_uap.pth'
    #     if is_query:
    #         uap = torch.load(
    #             uap_path, map_location=self.accelerator.device)
    #         uap = torch.clamp(uap, -self.epsilon, self.epsilon)
    #         # uap = uap / self.epsilon
    #         # epsilon = 10 / 255
    #         # uap = uap * epsilon
    #         # uap = torch.clamp(uap, -epsilon, epsilon)
    #         adv_imgs = torch.clamp(imgs + uap, 0, 1)

    #         self._make_log_dir_if_missing(imgs_path[0].split(os.sep)[-3])

    #         if batch_idx == 1 and self.accelerator.is_main_process:
    #             save_image(
    #                 adv_imgs[: 16],
    #                 f'{self.log_dir}/{self.target_model.name}_adv_imgs.png',
    #                 pad_value=1)
    #             save_image(
    #                 adv_imgs[: 16] - imgs[: 16],
    #                 f'{self.log_dir}/{self.target_model.name}_delta.png',
    #                 normalize=True, pad_value=1)
    #         imgs = adv_imgs

    #     # extract_features
    #     if 'transreid' in self.target_model.name:
    #         feats = self.target_model(imgs, cam_label=camids)
    #     else:
    #         feats = self.target_model(imgs)

    #     if self._use_fliplr:
    #         imgs_fliplr = T.functional.hflip(imgs)
    #         feats_fliplr = self.target_model(imgs_fliplr)
    #         feats = (feats + feats_fliplr) / 2.

    #     return feats, pids, camids

    # For ditim
    def val_step(self, batch, batch_idx, is_query=True):
        imgs, pids, camids, imgs_path, _ = batch.values()

        adv_imgs_path = 'logs/ditim/market1501/inceptionv3_bot'
        if is_query:
            adv_imgs = torch.load(
                f'{adv_imgs_path}/batch_{batch_idx}.pth',
                map_location=self.accelerator.device)

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

        # extract_features
        if 'transreid' in self.target_model.name:
            feats = self.target_model(imgs, cam_label=camids)
        else:
            feats = self.target_model(imgs)

        if self._use_fliplr:
            imgs_fliplr = T.functional.hflip(imgs)
            feats_fliplr = self.target_model(imgs_fliplr)
            feats = (feats + feats_fliplr) / 2.

        return feats, pids, camids


@TRAINER_REGISTRY.register()
def attack_eval(**trainer_params):
    return AttackEvaluator(**trainer_params)
