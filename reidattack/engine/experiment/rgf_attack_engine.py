import os
from typing import List, Optional, Union

import accelerate
import kornia as K
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from einops import rearrange, reduce, repeat
from engine.base_engine import BaseEngine
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from utils import mkdir_if_missing

from . import ENGINE_REGISTRY
from .evaluate_engine import EvaluateEngine


class RGFAttackEngine(EvaluateEngine):
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

    def rgf_attack(self, imgs, camids):
        max_queries = 5000

        fd_eta = 0.1
        step_size = 0.01
        momentum = 0.0
        decay = 1.0

        # adv_imgs = imgs.clone()
        adv_imgs = torch.clamp(
            imgs + torch.rand_like(imgs) * 2 * (2 / 255) - (2 / 255), 0, 1
        )
        feats = self._reid_model_forward(self.target_model, imgs, camids)
        for _ in range(max_queries // 2):
            exp_noise = torch.randn_like(adv_imgs)
            exp_noise = exp_noise / torch.norm(exp_noise)

            input1 = adv_imgs + fd_eta * exp_noise
            adv_feats1 = self._reid_model_forward(self.target_model, input1, camids)
            l1 = (F.normalize(adv_feats1) * F.normalize(feats)).sum(dim=1).mean()

            input2 = adv_imgs
            adv_feats2 = self._reid_model_forward(self.target_model, input2, camids)
            l2 = (F.normalize(adv_feats2) * F.normalize(feats)).sum(dim=1).mean()
            est_deriv = (l1 - l2) / fd_eta
            grad = est_deriv.view(-1, 1, 1, 1) * exp_noise

            grad = momentum * decay + grad / torch.linalg.vector_norm(
                grad, ord=1, dim=(1, 2, 3), keepdim=True
            ).clamp(min=1e-8)
            momentum = grad
            adv_imgs -= step_size * grad.sign()

            delta = torch.clamp(adv_imgs - imgs, min=-self.epsilon, max=self.epsilon)
            adv_imgs = torch.clamp(imgs + delta, 0, 1)
        return adv_imgs

    def val_step(self, batch, batch_idx, is_query=True):
        imgs, pids, camids, imgs_path, _ = batch.values()

        if is_query:
            adv_imgs = self.rgf_attack(imgs, camids)

            self._make_log_dir_if_missing(imgs_path[0].split(os.sep)[-3])
            cache_path = os.path.join(self.log_dir, f"{self.target_model.name}")
            mkdir_if_missing(cache_path)

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
            # cache adv_imgs
            torch.save(adv_imgs, f"{cache_path}/batch_{batch_idx}.pth")

        # extract_features
        feats = self._reid_model_forward(self.target_model, imgs, camids)

        return feats, pids, camids

    def _reid_model_forward(self, model, imgs, camids):
        if "transreid" in model.name:
            feats = model(imgs, cam_label=camids)
        else:
            feats = model(imgs)
        return feats


@ENGINE_REGISTRY.register()
def rgf(**trainer_params):
    return RGFAttackEngine(**trainer_params)
