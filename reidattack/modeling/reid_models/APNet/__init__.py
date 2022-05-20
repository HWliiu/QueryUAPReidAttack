# Project imported from https://github.com/CHENGY12/APNet
from .modeling import Baseline
from .. import REID_MODEL_BUILDER_REGISTRY, set_model_name


@REID_MODEL_BUILDER_REGISTRY.register()
@set_model_name()
def resnet50_ap(num_classes, **kwargs):
    model = Baseline(num_classes=num_classes, last_stride=1,
                     level=2, model_path=None, msmt=False, **kwargs)
    return model
