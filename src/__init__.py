"""
DRL Assignment 3: Meta and Transfer Learning
"""

from .utils import TrainingConfig, TrainingStats, DEVICE
from .environments import create_env, list_environments, get_env_info
from .actor_critic import ActorCritic, ActorCriticTrainer, train_individual_network, load_trained_model
from .actor_critic_fast import ActorCriticFast, FastTrainer, train_fast
from .fine_tuning import fine_tune, run_section2_experiments
from .progressive_networks import train_progressive_network, run_section3_experiments

# Optional: sb3_baseline requires stable_baselines3 which may have version conflicts
try:
    from .sb3_baseline import train_sb3_baseline, run_all_baselines, compare_with_custom, evaluate_sb3_model
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False
    train_sb3_baseline = None
    run_all_baselines = None
    compare_with_custom = None
    evaluate_sb3_model = None

__all__ = [
    'TrainingConfig',
    'TrainingStats',
    'DEVICE',
    'create_env',
    'list_environments',
    'get_env_info',
    'ActorCritic',
    'ActorCriticTrainer',
    'train_individual_network',
    'load_trained_model',
    'ActorCriticFast',
    'FastTrainer',
    'train_fast',
    'fine_tune',
    'run_section2_experiments',
    'train_progressive_network',
    'run_section3_experiments',
]

# Add sb3 exports only if available
if _SB3_AVAILABLE:
    __all__.extend([
        'train_sb3_baseline',
        'run_all_baselines',
        'compare_with_custom',
        'evaluate_sb3_model',
    ])
