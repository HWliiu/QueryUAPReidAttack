import random
from typing import List, Optional, Union

import accelerate
import kornia as K
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from accelerate.utils import extract_model_from_parallel
from einops import rearrange, reduce, repeat
from engine.base_engine import BaseEngine
from scipy import stats as st
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from . import ENGINE_REGISTRY


class VanillaUAPAttackEngine(BaseEngine):
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
        algorithm: str = "vanilla_uap",
    ) -> None:
        super().__init__(
            train_dataloader,
            query_dataloader,
            gallery_dataloader,
            accelerator,
            agent_models,
            target_model,
            segment_model,
            algorithm,
        )
        self.image_size = image_size
        self.uap = torch.zeros((1, 3, *image_size), device=self.accelerator.device)

        self.momentum = torch.zeros_like(self.uap)
        self.step_size = 2 / 255
        self.decay = 0.5

    def _vanilla_uap_training_step(self, imgs, pids, camids):
        agent_model = self.agent_models[0]

        feats = self._reid_model_forward(agent_model, imgs, pids, camids)

        for i in range(len(imgs)):
            img, feat = imgs[i : i + 1], feats[i : i + 1]
            pid, camid = pids[i : i + 1], camids[i : i + 1]

            self.uap.requires_grad_(True)
            adv_img = torch.clamp(img + self.uap, 0, 1)
            adv_feat = self._reid_model_forward(agent_model, adv_img, pid, camid)

            loss = (F.normalize(adv_feat) * F.normalize(feat)).sum(dim=1).mean()

            grad = torch.autograd.grad(loss, self.uap)[0]

            grad = self.momentum * self.decay + grad / torch.linalg.vector_norm(
                grad, ord=1, dim=(1, 2, 3), keepdim=True
            )
            self.momentum = grad

            self.uap.detach_()
            self.uap -= (2 / 255) * grad.sign()
            # project to ball
            self.uap.clamp_(-self.epsilon, self.epsilon)

        return loss

    def training_step(self, batch, batch_idx):
        imgs, pids, camids, imgs_path, _ = batch.values()

        loss = self._vanilla_uap_training_step(imgs, pids, camids)

        return {"loss": loss.detach()}

    def val_step(self, batch, batch_idx, is_query=True):
        imgs, pids, camids, imgs_path, _ = batch.values()

        if is_query:
            uap = torch.clamp(self.uap, -self.epsilon, self.epsilon)
            adv_imgs = torch.clamp(imgs + uap, 0, 1)

            if batch_idx == 1 and self.accelerator.is_main_process:
                save_image(
                    adv_imgs[:16],
                    f"{self.log_dir}/{self.target_model.name}_adv_imgs.png",
                    pad_value=1,
                )
                save_image(
                    adv_imgs[:16] - imgs[:16],
                    f"{self.log_dir}/{self.target_model.name}_delta.png",
                    normalize=True,
                    pad_value=1,
                )
            imgs = adv_imgs

        feats = self._reid_model_forward(self.target_model, imgs, pids, camids)

        return feats, pids, camids

    def save_state(self, epoch, map, is_best=False):
        torch.save(self.uap, f"{self.log_dir}/{self.target_model.name}_uap.pth")

    def _reid_model_forward(self, model, imgs, pids, camids):
        if "transreid" in model.name:
            feats = model(imgs, cam_label=camids)
        else:
            feats = model(imgs)
        return feats


@ENGINE_REGISTRY.register()
def vanilla_uap(**trainer_params):
    return VanillaUAPAttackEngine(**trainer_params)
