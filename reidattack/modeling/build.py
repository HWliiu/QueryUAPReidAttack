import logging
from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
from yacs.config import CfgNode

from .reid_models import REID_MODEL_BUILDER_REGISTRY
from .segment_models import BiSeNetV2
from .utils import (get_missing_parameters_message,
                    get_unexpected_parameters_message,
                    get_deleted_parameters_message)
from utils import check_isfile, HiddenPrints


def build_segment_model(seg_cfg: CfgNode) -> nn.Module:
    logger = logging.getLogger(
        'reidattack.' + build_segment_model.__qualname__)

    model_name = seg_cfg.NAME
    weights_path = seg_cfg.WEIGHT

    assert isinstance(weights_path, str)
    assert len(
        weights_path) > 0, 'Segmentation model weights_path must be specified'
    assert check_isfile(weights_path)

    logger.info(f'Building segmentation model `{model_name}`')
    if model_name == 'bisenetv2':
        with HiddenPrints():
            segment_model = BiSeNetV2(n_classes=2, aux_mode='eval')
    else:
        raise ValueError(f"Unknown segment model: {model_name}")

    # load model weights
    loaded_dict = torch.load(
        weights_path, map_location=torch.device('cpu'))
    logger.info(f"Loading pretrained model from {weights_path}")
    incompatible = segment_model.load_state_dict(
        loaded_dict, strict=False)
    if incompatible.missing_keys:
        logger.warn(
            get_missing_parameters_message(incompatible.missing_keys)
        )
    # if incompatible.unexpected_keys:
    #     logger.warn(
    #         get_unexpected_parameters_message(
    #             incompatible.unexpected_keys))

    return segment_model


def build_agent_models(agt_cfg: CfgNode) -> nn.Module:
    models = list()
    model_names = agt_cfg.NAMES
    model_weights = agt_cfg.WEIGHTS
    for name, path in zip(model_names, model_weights):
        models.append(_build_reid_model(name, path, is_agent=True))
    return models


def build_target_model(tgt_cfg: CfgNode, ) -> nn.Module:
    model_name = tgt_cfg.NAME
    weight_path = tgt_cfg.WEIGHT
    return _build_reid_model(model_name, weight_path, is_agent=False)


def _build_reid_model(
        model_name: str = None, weights_path: str = None, num_classes: int = 1,
        is_agent=False) -> nn.Module:
    logger = logging.getLogger(
        'reidattack.' + _build_reid_model.__qualname__)

    assert isinstance(weights_path, str)
    assert len(weights_path) > 0, 'Reid model weights_path must be specified'
    assert check_isfile(weights_path)

    # build reid model
    model_builder = REID_MODEL_BUILDER_REGISTRY.get(model_name)

    logger.info(
        f"Building {'agent' if is_agent else 'target'} model `{model_name}`")
    with HiddenPrints():
        if 'transreid' in model_name:
            # transreid used extra camera id infomation
            # TODO: remove dependency of weight path name
            if 'market1501' in weights_path:
                camera_num = 6
            elif 'dukemtmc' in weights_path:
                camera_num = 8
            elif 'msmt17' in weights_path:
                camera_num = 15
            reid_model = model_builder(num_classes, camera_num=camera_num)
        else:
            reid_model = model_builder(num_classes)

    # load model weights
    loaded_dict = torch.load(
        weights_path, map_location=torch.device('cpu'))
    logger.info(f"Loading pretrained model from {weights_path}")

    new_dict = OrderedDict()
    origin_dict = reid_model.state_dict()
    deleted_keys = list()
    for k, v in loaded_dict.items():
        # fixed key name of state dict
        if k.startswith('module.'):
            k = k[7:]
        # classify layer is unused
        if k in origin_dict and v.shape != origin_dict[k].shape:
            deleted_keys.append(k)
        else:
            new_dict[k] = v

    if deleted_keys:
        logger.info(get_deleted_parameters_message(deleted_keys))

    incompatible = reid_model.load_state_dict(
        new_dict, strict=False)
    del loaded_dict, origin_dict, new_dict

    # if incompatible.missing_keys:
    #     logger.warn(
    #         get_missing_parameters_message(incompatible.missing_keys)
    #     )
    if incompatible.unexpected_keys:
        logger.warn(
            get_unexpected_parameters_message(incompatible.unexpected_keys)
        )

    return reid_model
