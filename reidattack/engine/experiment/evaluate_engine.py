from typing import List, Optional, Union

import accelerate
import kornia as K
import torch
import torch.nn as nn
import torchvision.transforms as T
from engine.base_engine import BaseEngine
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from . import ENGINE_REGISTRY


class EvaluateEngine(BaseEngine):
    def val_step(self, batch, batch_idx, is_query=True):
        imgs, pids, camids, _, _ = batch.values()

        # extract_features
        if "transreid" in self.target_model.name:
            feats = self.target_model(imgs, cam_label=camids)
        else:
            feats = self.target_model(imgs)

        return feats, pids, camids


@ENGINE_REGISTRY.register()
def no_attack(**engine_params):
    return EvaluateEngine(**engine_params)
