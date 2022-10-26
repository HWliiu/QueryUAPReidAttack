import logging
from collections import OrderedDict

import kornia as K
import torch
import torch.nn as nn
from utils import HiddenPrints, check_isfile

from .reid_models import REID_MODEL_BUILDER_REGISTRY
from .segment_models import BiSeNetV2
from .utils import (MutiInputSequential, get_deleted_parameters_message,
                    get_missing_parameters_message,
                    get_unexpected_parameters_message)


def build_segment_model(
    model_name: str, weights_path: str, mean: tuple[float], std: tuple[float]
) -> nn.Module:
    logger = logging.getLogger("reidattack." + build_segment_model.__qualname__)

    assert isinstance(weights_path, str)
    assert len(weights_path) > 0, "Segmentation model weights_path must be specified"
    assert check_isfile(weights_path)

    logger.info(f"Building segmentation model `{model_name}`")
    if model_name == "bisenetv2":
        with HiddenPrints():
            segment_model = BiSeNetV2(n_classes=2, aux_mode="eval")
    else:
        raise ValueError(f"Unknown segment model: {model_name}")

    # load model weights
    loaded_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    logger.info(f"Loading pretrained model from {weights_path}")
    incompatible = segment_model.load_state_dict(loaded_dict, strict=False)
    if incompatible.missing_keys:
        logger.warn(get_missing_parameters_message(incompatible.missing_keys))
    # if incompatible.unexpected_keys:
    #     logger.warn(
    #         get_unexpected_parameters_message(
    #             incompatible.unexpected_keys))

    segment_model = nn.Sequential(K.enhance.Normalize(mean, std), segment_model).eval()
    segment_model.requires_grad_(False)

    return segment_model


def build_agent_models(
    model_names: list[str],
    model_weights: list[str],
    mean: tuple[float],
    std: tuple[float],
) -> nn.Module:
    models = list()

    for name, path in zip(model_names, model_weights):
        models.append(_build_reid_model(name, path, mean, std, is_agent=True))
    return models


def build_target_model(
    model_name: str, weight_path: str, mean: tuple[float], std: tuple[float]
) -> nn.Module:
    return _build_reid_model(model_name, weight_path, mean, std, is_agent=False)


def _build_reid_model(
    model_name: str,
    weights_path: str,
    mean: tuple[float],
    std: tuple[float],
    num_classes: int = 1,
    is_agent=False,
) -> nn.Module:
    logger = logging.getLogger("reidattack." + _build_reid_model.__qualname__)

    assert isinstance(weights_path, str)
    assert len(weights_path) > 0, "Reid model weights_path must be specified"
    assert check_isfile(weights_path)

    # build reid model
    model_builder = REID_MODEL_BUILDER_REGISTRY.get(model_name)

    logger.info(f"Building {'agent' if is_agent else 'target'} model `{model_name}`")
    with HiddenPrints():
        if "transreid" in model_name:
            # transreid used extra camera id infomation
            # TODO: remove dependency of weight path name
            if "market1501" in weights_path:
                camera_num = 6
            elif "dukemtmc" in weights_path:
                camera_num = 8
            elif "msmt17" in weights_path:
                camera_num = 15
            reid_model = model_builder(num_classes, camera_num=camera_num)
        else:
            reid_model = model_builder(num_classes)

    # load model weights
    loaded_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    logger.info(f"Loading pretrained model from {weights_path}")

    new_dict = OrderedDict()
    origin_dict = reid_model.state_dict()
    deleted_keys = list()
    for k, v in loaded_dict.items():
        # fixed key name of state dict
        if k.startswith("module."):
            k = k[7:]
        # classify layer is unused
        if k in origin_dict and v.shape != origin_dict[k].shape:
            deleted_keys.append(k)
        else:
            new_dict[k] = v

    if deleted_keys:
        logger.info(get_deleted_parameters_message(deleted_keys))

    incompatible = reid_model.load_state_dict(new_dict, strict=False)
    del loaded_dict, origin_dict, new_dict

    # if incompatible.missing_keys:
    #     logger.warn(
    #         get_missing_parameters_message(incompatible.missing_keys)
    #     )
    if incompatible.unexpected_keys:
        logger.warn(get_unexpected_parameters_message(incompatible.unexpected_keys))

    # warp model
    reid_model = MutiInputSequential(K.enhance.Normalize(mean, std), reid_model).eval()
    reid_model.requires_grad_(False)
    setattr(reid_model, "name", model_name)

    return reid_model
