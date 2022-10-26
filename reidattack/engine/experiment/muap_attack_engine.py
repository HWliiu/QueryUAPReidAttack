import os
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
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from utils import mkdir_if_missing

from . import ENGINE_REGISTRY

# Code same as https://github.com/wenjie710/MUAP


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


def normalize(x, axis=1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1.0 * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y, square=False):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    if square:
        dist = dist.clamp(min=1e-12)
    else:
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


# https://arxiv.org/pdf/1906.07589.pdf
def map_loss_v3(atta_feat, feat, target, bin_num, margin=0):
    # import numpy as np
    # np.save('atta_feat.npy', atta_feat.data.cpu().numpy())
    # np.save('feat.npy', feat.data.cpu().numpy())
    # np.save('target.npy', target.data.cpu().numpy())
    N = atta_feat.size(0)
    atta_feat = normalize(atta_feat)
    feat = normalize(feat)
    dist_raw = euclidean_dist(atta_feat, feat)
    dist = dist_raw.clone()
    # bin_num = 20
    # bin_len = 2./(bin_num-1)
    bin_len = (2.0 + margin) / (bin_num - 1)
    is_pos = target.expand(N, N).eq(target.expand(N, N).t()).float()
    is_neg = target.expand(N, N).ne(target.expand(N, N).t()).float()
    total_true_indicator = torch.zeros(N).to("cuda")
    total_all_indicator = torch.zeros(N).to("cuda")
    AP = torch.zeros(N).to("cuda")

    # import pdb
    # pdb.set_trace()
    if margin is None:
        pass
    else:
        is_pos_index = target.expand(N, N).eq(target.expand(N, N).t())
        is_neg_index = target.expand(N, N).ne(target.expand(N, N).t())
        dist[is_pos_index] = dist_raw[is_pos_index] - margin / 2
        dist[is_neg_index] = dist_raw[is_neg_index] + margin / 2

    for i in range(1, bin_num + 1):
        # bm = 1 - (i-1) * bin_len
        bm = (i - 1) * bin_len - margin / 2.0
        indicator = (1 - torch.abs(dist - bm) / bin_len).clamp(min=0)
        true_indicator = is_pos * indicator
        all_indicator = indicator
        sum_true_indicator = torch.sum(true_indicator, 1)
        sum_all_indicator = torch.sum(all_indicator, 1)
        total_true_indicator = total_true_indicator + sum_true_indicator
        total_all_indicator = total_all_indicator + sum_all_indicator
        Pm = total_true_indicator / total_all_indicator.clamp(min=1e-12)
        rm = sum_true_indicator / 4
        ap_bin = Pm * rm
        AP = AP + ap_bin
        # import pdb
        # pdb.set_trace()
    final_AP = torch.sum(AP) / N
    return final_AP


class MapLoss(nn.Module):
    def __init__(self):
        super(MapLoss, self).__init__()
        # self.name = 'map'

    def forward(self, atta_feat, feat, target, bin_num, margin=0):
        loss = map_loss_v3(atta_feat, feat, target, bin_num, margin=margin)
        return loss


def attack_update(att_img, grad, pre_sat, g, rate=0.8, base=False, i=10, radiu=10):

    norm = torch.sum(torch.abs(grad).view((grad.shape[0], -1)), dim=1).view(
        -1, 1, 1
    ) + torch.tensor([[[1e-12]], [[1e-12]], [[1e-12]]], device=grad.device)
    # norm = torch.max(torch.abs(grad).flatten())
    x_grad = grad / norm
    if torch.isnan(x_grad).any() or torch.isnan(g).any():
        import pdb

        pdb.set_trace()
    g = 0.4 * g + x_grad
    att_img = att_img - 0.004 * g.sign()
    # att_img = att_img - 0.008 * g.sign()
    radiu = radiu / 255.0
    att_img = torch.clamp(att_img, -radiu, radiu)

    pre_sat = torch.div(
        torch.sum(torch.eq(torch.abs(att_img), radiu), dtype=torch.float32),
        torch.tensor(
            att_img.flatten().size(), dtype=torch.float32, device=att_img.device
        ),
    )

    if not base:
        img_abs = torch.abs(att_img)
        img_sort = torch.sort(img_abs.flatten(), descending=True)[0]
        new_rate = max(pre_sat, rate)
        if pre_sat < rate and i > 0:
            img_median = img_sort[int((len(img_sort) * new_rate))]
            att_img = att_img * (radiu / (img_median + 1e-6))
            # print('median', img_median)
            att_img = torch.clamp(att_img, -radiu, radiu)

    sat = torch.div(
        torch.sum(torch.eq(torch.abs(att_img), radiu), dtype=torch.float32),
        torch.tensor(
            att_img.flatten().size(), dtype=torch.float32, device=att_img.device
        ),
    )

    # print('presat:', pre_sat, 'sat: ', sat)

    return att_img, sat, g


class MUAPAttackEngine(BaseEngine):
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
        algorithm: str = "muap",
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

        torch.manual_seed(1)
        self.attack_img = (
            torch.rand(3, 256, 128, requires_grad=True, device=self.accelerator.device)
            * 1e-6
        )
        self.normalize_transform = T.Normalize(
            mean=[0, 0, 0], std=[0.229, 0.224, 0.225]
        )
        self.g = torch.tensor([0.0], device=self.attack_img.device)
        self.pre_sat = 1.0
        self.loss_fn1 = MapLoss()
        self.loss_fn2 = TVLoss(TVLoss_weight=10)
        self.pre_loss = np.inf

    def _muap_training_step(self, imgs, pids, camids):
        imgs = T.functional.normalize(
            imgs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        model = self.agent_models[0][1]
        loss_fn1 = self.loss_fn1
        loss_fn2 = self.loss_fn2
        base = False
        scale_rate = 0.8
        radiu = 10
        i = self.current_epoch - 1

        attack_img = Variable(self.attack_img, requires_grad=True)
        images = imgs
        target = pids
        normed_atta_img = self.normalize_transform(attack_img)
        # ! images unclip
        pertubated_images = torch.add(images, normed_atta_img)

        median_img = pertubated_images
        feat = model(images)

        attack_feat = model(median_img)

        map_loss = loss_fn1(attack_feat, feat, target, 5)

        if base:
            total_loss = map_loss
        else:
            tvl_loss = loss_fn2(median_img)
            total_loss = tvl_loss + map_loss
        total_loss.backward()
        attack_grad = attack_img.grad
        model.zero_grad()
        self.attack_img.detach_()
        self.attack_img, sat, g = attack_update(
            attack_img, attack_grad, self.pre_sat, self.g, scale_rate, base, i, radiu
        )
        self.pre_sat = sat

        return total_loss

    def training_step(self, batch, batch_idx):
        imgs, pids, camids, imgs_path, _ = batch.values()
        loss = self._muap_training_step(imgs, pids, camids)

        return {"loss": loss.detach()}

    def val_step(self, batch, batch_idx, is_query=True):
        imgs, pids, camids, imgs_path, _ = batch.values()

        if is_query:
            uap = self.attack_img
            uap = torch.clamp(uap, -self.epsilon, self.epsilon)
            adv_imgs = torch.clamp(imgs + uap, 0, 1)

            if batch_idx == 1 and self.accelerator.is_main_process:
                self._make_log_dir_if_missing(imgs_path[0].split(os.sep)[-3])
                save_image(
                    adv_imgs[:16],
                    f"{self.log_dir}/{self.agent_models[0].name}_adv_imgs.png",
                    pad_value=1,
                )
                save_image(
                    adv_imgs[:16] - imgs[:16],
                    f"{self.log_dir}/{self.agent_models[0].name}_delta.png",
                    normalize=True,
                    pad_value=1,
                )
            imgs = adv_imgs

        feats = self._reid_model_forward(self.target_model, imgs, pids, camids)

        return feats, pids, camids

    def save_state(self, epoch, map, is_best=False):
        torch.save(
            self.attack_img, f"{self.log_dir}/{self.agent_models[0].name}_uap.pth"
        )

    def _reid_model_forward(self, model, imgs, pids, camids):
        if "transreid" in model.name:
            feats = model(imgs, cam_label=camids)
        else:
            feats = model(imgs)
        return feats


@ENGINE_REGISTRY.register()
def muap(**trainer_params):
    return MUAPAttackEngine(**trainer_params)
