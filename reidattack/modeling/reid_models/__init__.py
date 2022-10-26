from utils.registry import Registry

REID_MODEL_BUILDER_REGISTRY = Registry("REID_MODEL")
REID_MODEL_BUILDER_REGISTRY.__doc__ = """Registry for reid model"""


def _set_function_name(function_name):
    """For dynamically regist model builder."""

    def decorator(func):
        func.__name__ = function_name
        func.__qualname__ = function_name
        return func

    return decorator


from .ABDNet import resnet50_abd
from .AGWNet import resnet50_agw
from .APNet import resnet50_ap
from .DeepPersonReid import *
from .ReidStrongBaseline import *
from .TransReID import *
