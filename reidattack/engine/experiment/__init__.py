from utils.registry import Registry
TRAINER_REGISTRY = Registry('TRAINERL')
TRAINER_REGISTRY.__doc__ = """Registry for trainer"""

from .query_uap_trainer import QueryUAPAttackTrainer
from .no_attack_evaluator import NoAttackEvaluator
from .attack_evaluator import AttackEvaluator
from .muap_attack_trainer import MUAPAttackTrainer
from .muap_attack_refactor_trainer import MUAPAttackRefactorTrainer
from .vanilla_uap_attack_trainer import VanillaUAPAttackTrainer
from .ditim_attack_trainer import DITIMAttackTrainer
from .bandits_attack_trainer import BanditsAttackTrainer
from .bandits_uap_attack_trainer import BanditsUAPAttackTrainer
from .rgf_uap_attack_trainer import RGFUAPAttackTrainer
from .rgf_attack_trainer import RGFAttackTrainer