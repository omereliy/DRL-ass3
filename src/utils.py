"""
Utility functions and configuration for DRL Assignment 3.
Meta and Transfer Learning with Actor-Critic.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standardized dimensions for transfer learning
# CartPole: obs=4, act=2 (discrete)
# Acrobot: obs=6, act=3 (discrete)
# MountainCarContinuous: obs=2, act=1 (continuous, but we'll discretize)
STANDARDIZED_OBS_DIM = 6  # Max observation dimension (Acrobot has 6)
STANDARDIZED_ACT_DIM = 3  # Max action dimension (Acrobot has 3)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    gamma: float = 0.99
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    hidden_dim: int = 128
    max_episodes: int = 2000
    max_steps: int = 1000
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    log_interval: int = 10
    save_interval: int = 100
    early_stop_reward: Optional[float] = None
    seed: int = 42


@dataclass
class TrainingStats:
    """Statistics from training."""
    total_episodes: int = 0
    total_steps: int = 0
    training_time: float = 0.0
    final_avg_reward: float = 0.0
    convergence_episode: Optional[int] = None
    rewards_history: list = None

    def __post_init__(self):
        if self.rewards_history is None:
            self.rewards_history = []


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def init_weights(module: nn.Module):
    """Initialize network weights using Xavier initialization."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def get_model_path(env_name: str, model_type: str = "actor_critic") -> str:
    """Get the path for saving/loading models."""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    return os.path.join(models_dir, f"{model_type}_{env_name}.pt")


def get_log_dir(experiment_name: str) -> str:
    """Get the directory for TensorBoard logs."""
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(logs_dir, f"{experiment_name}_{timestamp}")


def create_tensorboard_writer(experiment_name: str) -> SummaryWriter:
    """Create a TensorBoard writer for logging."""
    log_dir = get_log_dir(experiment_name)
    return SummaryWriter(log_dir=log_dir)


def compute_returns(rewards: list, gamma: float, normalize: bool = False) -> torch.Tensor:
    """Compute discounted returns."""
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
    if normalize and len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def compute_gae(rewards: list, values: list, next_value: float,
                gamma: float = 0.99, lam: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation."""
    values = values + [next_value]
    gae = 0
    advantages = []
    returns = []

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] - values[step]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[step])

    advantages = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
    returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

    # Normalize advantages
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


def moving_average(data: list, window: int = 100) -> list:
    """Compute moving average of data."""
    if len(data) < window:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return list((cumsum[window:] - cumsum[:-window]) / window)


def check_convergence(rewards_history: list, threshold: float,
                      window: int = 100, min_episodes: int = 100) -> bool:
    """Check if training has converged based on average reward."""
    if len(rewards_history) < min_episodes:
        return False
    avg_reward = np.mean(rewards_history[-window:])
    return avg_reward >= threshold


def print_training_stats(stats: TrainingStats, env_name: str):
    """Print training statistics."""
    print(f"\n{'='*50}")
    print(f"Training Statistics for {env_name}")
    print(f"{'='*50}")
    print(f"Total Episodes: {stats.total_episodes}")
    print(f"Total Steps: {stats.total_steps}")
    print(f"Training Time: {stats.training_time:.2f} seconds")
    print(f"Final Avg Reward (last 100): {stats.final_avg_reward:.2f}")
    if stats.convergence_episode:
        print(f"Converged at Episode: {stats.convergence_episode}")
    print(f"{'='*50}\n")


def log_to_tensorboard(writer: SummaryWriter, episode: int,
                       reward: float, loss: float = None,
                       actor_loss: float = None, critic_loss: float = None,
                       entropy: float = None, avg_reward: float = None):
    """Log training metrics to TensorBoard."""
    writer.add_scalar("Reward/Episode", reward, episode)
    if avg_reward is not None:
        writer.add_scalar("Reward/Average", avg_reward, episode)
    if loss is not None:
        writer.add_scalar("Loss/Total", loss, episode)
    if actor_loss is not None:
        writer.add_scalar("Loss/Actor", actor_loss, episode)
    if critic_loss is not None:
        writer.add_scalar("Loss/Critic", critic_loss, episode)
    if entropy is not None:
        writer.add_scalar("Entropy", entropy, episode)
