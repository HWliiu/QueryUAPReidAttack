# Project imported from https://github.com/michuanhaohao/reid-strong-baseline
import sys

from .. import REID_MODEL_BUILDER_REGISTRY, _set_function_name
from .modeling import build_model

# Bag of tricks version
bag_of_tricks_models = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "se_resnet50",
    "se_resnet101",
    "se_resnet152",
    "se_resnext50",
    "se_resnext101",
    "senet154",
    "resnet50_ibn_a",
    "vgg19",
    "densnet121",
    "shufflenetv2",
    "mobilenetv2",
    "inceptionv3",
    "efficientnet_b0",
    "regnet_x_1_6gf",
    "convnext_tiny",
]

__all__ = [name + "_bot" for name in bag_of_tricks_models]

_module = sys.modules[__name__]

for model_name in __all__:

    @REID_MODEL_BUILDER_REGISTRY.register()
    @_set_function_name(model_name)
    def _(num_classes, *, name=model_name.replace("_bot", ""), **kwargs):
        # default `name` parameter for avoid delay binding
        model = build_model(num_classes=num_classes, model_name=name)
        return model

    setattr(_module, model_name, _)
