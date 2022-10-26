from contextlib import ExitStack

import accelerate
from data import build_test_dataloader, build_train_dataloader
from engine import *
from modeling import (build_agent_models, build_segment_model,
                      build_target_model)
from utils.context import ctx_noparamgrad_and_eval
from yacs.config import CfgNode


def launch(cfg: CfgNode, accelerator: accelerate.Accelerator):
    # build train dataloader
    train_dataloader = build_train_dataloader(cfg.DATA)
    train_dataloader = accelerator.prepare(train_dataloader)

    # build test dataloaders
    query_dataloader, gallery_dataloader = build_test_dataloader(cfg.DATA)
    query_dataloader, gallery_dataloader = accelerator.prepare(
        query_dataloader, gallery_dataloader
    )

    mean = cfg.DATA.TRANSFORM.NORM.PIXEL_MEAN
    std = cfg.DATA.TRANSFORM.NORM.PIXEL_STD

    agt_cfg = cfg.MODULE.AGENT_MODELS
    agent_models = build_agent_models(agt_cfg.NAMES, agt_cfg.WEIGHTS, mean, std)
    agent_models = accelerator.prepare(*agent_models)
    if not isinstance(agent_models, tuple):
        agent_models = (agent_models,)

    # build target model
    tgt_cfg = cfg.MODULE.TARGET_MODEL
    target_model = build_target_model(tgt_cfg.NAME, tgt_cfg.WEIGHT, mean, std)
    target_model = accelerator.prepare(target_model)

    # build segment model
    seg_cfg = cfg.MODULE.SEGMENT_MODEL
    segment_model = build_segment_model(seg_cfg.NAME, seg_cfg.WEIGHT, mean, std)
    segment_model = accelerator.prepare(segment_model)

    egn_cfg = cfg.ENGINE

    engine_params = {
        "train_dataloader": train_dataloader,
        "query_dataloader": query_dataloader,
        "gallery_dataloader": gallery_dataloader,
        "accelerator": accelerator,
        "agent_models": agent_models,
        "target_model": target_model,
        "segment_model": segment_model,
        "algorithm": egn_cfg.ATTACK_ALGORITHM,
    }

    if "uap" in egn_cfg.ATTACK_ALGORITHM:
        engine_params["image_size"] = cfg.DATA.TRANSFORM.SIZE_TRAIN
    engine = ENGINE_REGISTRY.get(egn_cfg.ATTACK_ALGORITHM)(**engine_params)

    engine.run(
        max_epoch=egn_cfg.MAX_EPOCH,
        epsilon=egn_cfg.EPSILON,
        eval_only=egn_cfg.EVAL_ONLY,
        eval_period=egn_cfg.EVAL_PERIOD,
        rerank=egn_cfg.RERANKING,
    )
