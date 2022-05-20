import os
import time
import logging
from collections import OrderedDict
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import accelerate
import kornia as K
from tqdm.auto import tqdm

from evaluation import ReidMetric
from utils import mkdir_if_missing
from .utils import MetricMeter, MutiInputSequential


class BaseTrainer:
    def __init__(
            self,
            train_dataloader: DataLoader,
            query_dataloader: DataLoader,
            gallery_dataloader: DataLoader,
            accelerator: accelerate.Accelerator,
            agent_models: List[nn.Module],
            target_model: nn.Module,
            segment_model: nn.Module,
            algorithm: str,
            use_normalized: bool = True,
            normalize_mean: Optional[List[float]] = None,
            normalize_std: Optional[List[float]] = None) -> None:

        self.train_dataloader = train_dataloader
        self.query_dataloader = query_dataloader
        self.gallery_dataloader = gallery_dataloader

        self.accelerator = accelerator
        self.algorithm = algorithm

        self.use_normalized = use_normalized
        self.normalize_mean = torch.tensor(
            normalize_mean, device=self.accelerator.device) if normalize_mean is not None else None
        self.normalize_std = torch.tensor(
            normalize_std, device=self.accelerator.device) if normalize_std is not None else None

        # warp every model with normalization
        self.agent_models = [
            m if not use_normalized else MutiInputSequential(
                K.enhance.Normalize(
                    mean=self.normalize_mean, std=self.normalize_std), m)
            for m in agent_models]
        for m in self.agent_models:
            if isinstance(m, nn.Sequential):
                # add name property for distinguish
                m.name = self.accelerator.unwrap_model(m[1]).name

        self.target_model = target_model if not use_normalized else MutiInputSequential(
            K.enhance.Normalize(mean=self.normalize_mean, std=self.normalize_std), target_model)
        if isinstance(self.target_model, nn.Sequential):
            self.target_model.name = self.accelerator.unwrap_model(
                self.target_model[1]).name

        self.segment_model = segment_model if not use_normalized else nn.Sequential(
            K.enhance.Normalize(mean=self.normalize_mean, std=self.normalize_std), segment_model)

        self.log_subdirectory = algorithm
        self.writer = None
        self.logger = logging.getLogger(
            'reidattack.' + self.__class__.__qualname__)

        self.reid_metric = ReidMetric(
            distributed_rank=self.accelerator.process_index)

        self.configure_criterions()

        self._bar_format = "{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]"
        self._saved_history = []

    def train(self):
        train_bar = tqdm(
            total=len(self.train_dataloader),
            leave=False,
            unit='batch',
            bar_format=self._bar_format,
            disable=not self.accelerator.is_local_main_process)
        train_bar.set_description(
            f'Train [{self.current_epoch}/{self.max_epoch}]')

        average_meters = MetricMeter()

        self.on_train_start()
        for step, batch in enumerate(self.train_dataloader, start=1):
            train_bar.update()
            losses = self.training_step(batch, step)
            average_meters.update(losses)

            avg_losses = {}
            for name, avg_meter in average_meters.meters.items():
                avg_losses[name] = round(avg_meter.avg, 2)
            train_bar.set_postfix(avg_losses)
        self.on_train_end()
        train_bar.close()

        return avg_losses

    def test(self, rerank=False):
        self.on_val_start()
        test_bar = tqdm(
            total=len(self.query_dataloader) + len(self.gallery_dataloader) + 1
            if not self.reid_metric.gallery_cached else
            len(self.query_dataloader) + 1,  # +1 for compute metrics
            leave=False,
            unit='batch',
            bar_format=self._bar_format,
            disable=not self.accelerator.is_local_main_process)
        with torch.no_grad():
            self._extract_feats(test_bar, is_query=True)
            if not self.reid_metric.gallery_cached:
                self._extract_feats(test_bar, is_query=False)

        test_bar.set_description(f'Compute metrics')
        results = self.reid_metric.compute(
            rerank=rerank, reset_all=self.current_epoch >= self.max_epoch)
        self.accelerator.wait_for_everyone()

        test_bar.update()
        self.on_val_end()
        test_bar.close()
        return results

    def _extract_feats(self, test_bar, is_query=True):
        test_bar.set_description(
            f'Extract query' if is_query else f'Extract gallery')
        for step, batch in enumerate(
                self.query_dataloader if is_query else self.gallery_dataloader, 1):
            test_bar.update()
            feats, pids, camids = self.val_step(batch, step, is_query)
            if self.accelerator.num_processes > 1:
                self.accelerator.wait_for_everyone()
                feats, pids, camids = self.accelerator.gather(
                    (feats, pids, camids))
            self.reid_metric.update(feats, pids, camids, is_query)

    def run(
            self,
            max_epoch: int = 1,
            epsilon: float = 10 / 255,
            *,
            eval_only: bool = False,
            use_fliplr: bool = False,
            eval_period: int = 1,
            rerank: bool = False) -> None:

        self.max_epoch = max_epoch
        self.epsilon = epsilon
        self.eval_only = eval_only
        self._use_fliplr = use_fliplr

        self.current_epoch = 1

        if eval_only:
            self.logger.info('=> Evaluating only')
            results = self.test(rerank=rerank)
            return self._log_metrics(results, rerank)

        # if self.writer is None and self.accelerator.is_main_process:
        #     log_dir = os.path.join(
        #         'logs', self.log_subdirectory, 'runs')
        #     mkdir_if_missing(log_dir)

        #     self.writer = SummaryWriter(
        #         os.path.join(
        #             log_dir, f'version_{self._get_next_version(log_dir)}'))

        self.logger.info('=> Start training')

        best_map = 1.
        for self.current_epoch in range(1, self.max_epoch + 1):
            start = time.time()
            avg_losses = self.train()
            train_time = time.time() - start
            results = None
            if self.current_epoch % eval_period == 0:
                results = self.test(rerank=rerank)
            total_time = time.time() - start
            self.logger.info(
                f'=> Epoch end: {self.current_epoch}/{self.max_epoch}'
                f'  Average_loss: {avg_losses}'
                f'  Elapse: train {train_time:.2f}s val: {total_time-train_time:.2f}s total {total_time:.2f}s')
            if results:
                self._log_metrics(results, rerank)
                # self._log_tensorboard(rerank, results, avg_losses)

                map = results[0]['map']
                self.save_model(self.current_epoch, map,
                                is_best=map < best_map)
                if map < best_map:
                    best_map = map

        if self.writer is not None:
            self.writer.close()

    def _log_tensorboard(self, rerank, results, avg_losses):
        if self.writer is not None:
            self.writer.add_scalars(
                'losses', avg_losses, self.current_epoch)
            if rerank:
                self.writer.add_scalars(
                    'metrics/before_rerank', results[0],
                    self.current_epoch)
                self.writer.add_scalars(
                    'metrics/after_rerank', results[1],
                    self.current_epoch)
            else:
                self.writer.add_scalars(
                    'metrics/before_rerank', results[0], self.current_epoch)

    def _log_metrics(self, results, rerank):
        if results is None:
            return

        if rerank:
            self.logger.info(
                f"=> Compute metrics:\n"
                f"  before re-ranking: {results[0]}\n"
                f"  after re-ranking: {results[1]}")
        else:
            self.logger.info(f"=> Compute metrics: {results[0]}")

    def _make_log_dir_if_missing(self, dataset_name):
        if self.accelerator.is_main_process:
            self.log_dir = os.path.join(
                'logs', self.log_subdirectory, dataset_name)
            mkdir_if_missing(self.log_dir)
            return self.log_dir

    def save_model(self, epoch, map, is_best=False):
        ...

    def configure_criterions(self):
        """Configures the set of criters"""
        pass

    def training_step(self, batch, batch_idx):
        return NotImplemented

    def val_step(self, batch, batch_idx, is_query=True):
        return NotImplemented

    def on_train_start(self):
        ...

    def on_train_end(self):
        ...

    def on_val_start(self):
        ...

    def on_val_end(self):
        ...

    def _get_next_version(self, log_dir):
        try:
            listdir_info = [os.path.join(log_dir, file)
                            for file in os.listdir(log_dir)]
        except OSError:
            return 0

        existing_versions = []
        for d in listdir_info:
            bn = os.path.basename(d)
            if os.path.isdir(d) and bn.startswith("version_"):
                dir_ver = bn.split("_")[1].replace("/", "")
                existing_versions.append(int(dir_ver))
        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1
