import logging
import os
import time
from typing import List

import accelerate
import torch
import torch.nn as nn
from evaluation import ReidMetric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils import mkdir_if_missing

from .utils import MetricMeter


class BaseEngine:
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
    ) -> None:

        self.train_dataloader = train_dataloader
        self.query_dataloader = query_dataloader
        self.gallery_dataloader = gallery_dataloader

        self.accelerator = accelerator
        self.algorithm = algorithm

        self.agent_models = agent_models
        self.target_model = target_model
        self.segment_model = segment_model

        self.logger = logging.getLogger("reidattack." + self.__class__.__qualname__)

        self.reid_metric = ReidMetric(distributed_rank=self.accelerator.process_index)

        self._bar_format = "{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]"

    def train(self):
        train_bar = tqdm(
            total=len(self.train_dataloader),
            leave=False,
            unit="batch",
            bar_format=self._bar_format,
            disable=not self.accelerator.is_local_main_process,
        )
        train_bar.set_description(f"Train [{self.current_epoch}/{self.max_epoch}]")

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
            if not self.reid_metric.gallery_cached
            else len(self.query_dataloader) + 1,  # +1 for compute metrics
            leave=False,
            unit="batch",
            bar_format=self._bar_format,
            disable=not self.accelerator.is_local_main_process,
        )
        with torch.no_grad():
            self._extract_feats(test_bar, is_query=True)
            if not self.reid_metric.gallery_cached:
                self._extract_feats(test_bar, is_query=False)

        test_bar.set_description(f"Compute metrics")
        results = self.reid_metric.compute(
            rerank, reset_all=self.current_epoch >= self.max_epoch
        )
        self.accelerator.wait_for_everyone()

        test_bar.update()
        self.on_val_end()
        test_bar.close()
        return results

    def _extract_feats(self, test_bar, is_query=True):
        test_bar.set_description(f"Extract query" if is_query else f"Extract gallery")
        for step, batch in enumerate(
            self.query_dataloader if is_query else self.gallery_dataloader, 1
        ):
            test_bar.update()
            feats, pids, camids = self.val_step(batch, step, is_query)
            if self.accelerator.num_processes > 1:
                self.accelerator.wait_for_everyone()
                feats, pids, camids = self.accelerator.gather((feats, pids, camids))
            self.reid_metric.update(feats, pids, camids, is_query)

    def run(
        self,
        max_epoch: int = 1,
        epsilon: float = 10 / 255,
        *,
        eval_only: bool = False,
        eval_period: int = 1,
        rerank: bool = False,
    ) -> None:

        self.max_epoch = max_epoch
        self.epsilon = epsilon
        self.eval_only = eval_only

        self.current_epoch = 1

        if eval_only:
            self.logger.info("=> Evaluating only")
            results = self.test()
            return self._log_metrics(results)

        self.logger.info("=> Start training")

        best_map = 1.0
        for self.current_epoch in range(1, self.max_epoch + 1):
            start = time.perf_counter()
            avg_losses = self.train()
            train_time = time.perf_counter() - start
            results = None
            if self.current_epoch % eval_period == 0 or self.current_epoch >= max_epoch:
                results = self.test()
            total_time = time.perf_counter() - start
            self.logger.info(
                f"=> Epoch end: {self.current_epoch}/{self.max_epoch}"
                f"  Average_loss: {avg_losses}"
                f"  Elapse: train {train_time:.2f}s val: {total_time-train_time:.2f}s total {total_time:.2f}s"
            )
            if results:
                self._log_metrics(results)

                map = results["map"]
                self.save_state(self.current_epoch, map, is_best=map < best_map)
                if map < best_map:
                    best_map = map

    def _log_metrics(self, results):
        if self.accelerator.is_local_main_process:
            self.logger.info(f"=> Compute metrics: {results}")

    def _make_log_dir_if_missing(self, dataset_name):
        if self.accelerator.is_main_process:
            self.log_dir = os.path.join("logs", self.algorithm, dataset_name)
            mkdir_if_missing(self.log_dir)
            return self.log_dir

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

    def save_state(self, epoch, map, is_best=False):
        ...
