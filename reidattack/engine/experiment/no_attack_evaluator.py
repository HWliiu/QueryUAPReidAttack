from typing import List, Optional, Union

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import accelerate
import kornia as K

from engine.base_trainer import BaseTrainer
from . import TRAINER_REGISTRY


class NoAttackEvaluator(BaseTrainer):
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

    def val_step(self, batch, batch_idx, is_query=True):
        imgs, pids, camids, imgs_path, _ = batch.values()

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
def no_attack(**trainer_params):
    return NoAttackEvaluator(**trainer_params)
