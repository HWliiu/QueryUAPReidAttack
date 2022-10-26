from utils.registry import Registry

SAMPLER_REGISTRY = Registry("SAMPLER")
SAMPLER_REGISTRY.__doc__ = """Registry for Samplers"""

from .reid_sampler import (RandomDatasetSampler, RandomDomainSampler,
                           RandomIdentitySampler)
