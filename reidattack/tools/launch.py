from contextlib import ExitStack

from yacs.config import CfgNode
import accelerate

from data import build_train_dataloader, build_test_dataloader
from modeling import build_target_model, build_agent_models, build_segment_model
from utils.context import ctx_noparamgrad_and_eval
from engine import *


def launch(
        cfg: CfgNode,
        accelerator: accelerate.Accelerator):
    # build train dataloader
    train_dataloader = build_train_dataloader(cfg.DATA)
    train_dataloader = accelerator.prepare(train_dataloader)

    # build test dataloaders
    query_dataloader, gallery_dataloader = build_test_dataloader(cfg.DATA)
    query_dataloader, gallery_dataloader = accelerator.prepare(
        query_dataloader, gallery_dataloader)

    agt_cfg = cfg.MODULE.AGENT_MODELS
    agent_models = build_agent_models(agt_cfg)
    agent_models = accelerator.prepare(*agent_models)
    if not isinstance(agent_models, tuple):
        agent_models = (agent_models,)

    # build target model
    tgt_cfg = cfg.MODULE.TARGET_MODEL
    target_model = build_target_model(tgt_cfg)
    target_model = accelerator.prepare(target_model)

    # build segment model
    seg_cfg = cfg.MODULE.SEGMENT_MODEL
    segment_model = build_segment_model(seg_cfg)
    segment_model = accelerator.prepare(segment_model)

    with ExitStack() as stack:
        # fixed all models
        [stack.enter_context(ctx_noparamgrad_and_eval(m))
         for m in agent_models]
        stack.enter_context(ctx_noparamgrad_and_eval(target_model))
        stack.enter_context(ctx_noparamgrad_and_eval(segment_model))

        tfm_norm_cfg = cfg.DATA.TRANSFORM.NORM
        trn_cfg = cfg.TRAINER

        trainer_params = {
            "train_dataloader": train_dataloader,
            "query_dataloader": query_dataloader,
            "gallery_dataloader": gallery_dataloader,
            "accelerator": accelerator,
            "agent_models": agent_models,
            "target_model": target_model,
            "segment_model": segment_model,
            "algorithm": trn_cfg.ATTACK_ALGORITHM,
            "use_normalized": tfm_norm_cfg.ENABLED,
            "normalize_mean": tfm_norm_cfg.PIXEL_MEAN,
            "normalize_std": tfm_norm_cfg.PIXEL_STD}

        if 'uap' in trn_cfg.ATTACK_ALGORITHM:
            trainer_params["image_size"] = cfg.DATA.TRANSFORM.SIZE_TRAIN
        trainer = TRAINER_REGISTRY.get(
            trn_cfg.ATTACK_ALGORITHM)(
            **trainer_params)

        trainer.run(
            max_epoch=trn_cfg.MAX_EPOCH,
            epsilon=trn_cfg.EPSILON,
            eval_only=trn_cfg.EVAL_ONLY,
            eval_period=trn_cfg.EVAL_PERIOD,
            rerank=trn_cfg.RERANKING)
