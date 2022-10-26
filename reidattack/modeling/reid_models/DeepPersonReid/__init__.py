# Project imported from https://github.com/KaiyangZhou/deep-person-reid
import sys

import torch

from .. import REID_MODEL_BUILDER_REGISTRY, _set_function_name
from .models import __model_factory, build_model

model_names = list(name + "_dpr" for name in __model_factory.keys())

__all__ = model_names


_module = sys.modules[__name__]

for model_name in model_names:

    @REID_MODEL_BUILDER_REGISTRY.register()
    @_set_function_name(model_name)
    def _(num_classes, *, name=model_name.replace("_dpr", ""), **kwargs):
        # default `name` parameter for avoid delay binding
        model = build_model(
            name=name,
            num_classes=num_classes,
            pretrained=False,
            use_gpu=torch.cuda.is_available(),
        )
        return model

    setattr(_module, model_name, _)
