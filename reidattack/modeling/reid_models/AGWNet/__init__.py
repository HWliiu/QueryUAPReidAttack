# Project imported from https://github.com/mangye16/ReID-Survey/
from .modeling import Baseline

from .. import REID_MODEL_BUILDER_REGISTRY


@REID_MODEL_BUILDER_REGISTRY.register()
def resnet50_agw(num_classes, **kwargs):
    model = Baseline(
        num_classes, last_stride=1, model_path=None, model_name='resnet50_nl',
        gem_pool='on', pretrain_choice=None)
    return model
