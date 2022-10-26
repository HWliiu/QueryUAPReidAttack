from utils.registry import Registry

ENGINE_REGISTRY = Registry("TRAINERL")
ENGINE_REGISTRY.__doc__ = """Registry for trainer"""

from .bandits_attack_engine import BanditsAttackEngine
from .bandits_uap_attack_engine import BanditsUAPAttackEngine
from .ditim_attack_engine import DITIMAttackEngine
from .evaluate_attack_engine import EvaluateAttackEngine
from .evaluate_engine import EvaluateEngine
from .muap_attack_engine import MUAPAttackEngine
from .muap_attack_refactor_engine import MUAPAttackRefactorEngine
from .query_uap_engine import QueryUAPAttackEngine
from .rgf_attack_engine import RGFAttackEngine
from .rgf_uap_attack_engine import RGFUAPAttackEngine
from .vanilla_uap_attack_engine import VanillaUAPAttackEngine
