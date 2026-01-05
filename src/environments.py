"""
Standardized environment wrappers for transfer learning.
All environments will have the same observation and action dimensions.
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Any, Optional, Dict
from dataclasses import dataclass

from .utils import STANDARDIZED_OBS_DIM, STANDARDIZED_ACT_DIM


@dataclass
class EnvConfig:
    """Configuration for each environment."""
    name: str
    original_obs_dim: int
    original_act_dim: int
    is_continuous: bool
    convergence_threshold: float
    max_steps: int
    description: str


# Environment configurations
ENV_CONFIGS = {
    "CartPole-v1": EnvConfig(
        name="CartPole-v1",
        original_obs_dim=4,
        original_act_dim=2,
        is_continuous=False,
        convergence_threshold=475.0,  # Solved when avg >= 475 over 100 episodes
        max_steps=500,
        description="Balance a pole on a cart"
    ),
    "Acrobot-v1": EnvConfig(
        name="Acrobot-v1",
        original_obs_dim=6,
        original_act_dim=3,
        is_continuous=False,
        convergence_threshold=-100.0,  # Solved when avg >= -100 over 100 episodes
        max_steps=500,
        description="Swing the lower link to reach a height"
    ),
    "MountainCarContinuous-v0": EnvConfig(
        name="MountainCarContinuous-v0",
        original_obs_dim=2,
        original_act_dim=3,  # Discretized: left, nothing, right
        is_continuous=True,  # Original is continuous
        convergence_threshold=90.0,  # Solved when avg >= 90 over 100 episodes
        max_steps=999,
        description="Drive up a steep mountain"
    ),
}


class StandardizedEnv:
    """
    Wrapper that standardizes environment observations and actions.
    All environments will have the same input/output dimensions for transfer learning.
    """

    def __init__(self, env_name: str, seed: Optional[int] = None):
        """
        Initialize the standardized environment.

        Args:
            env_name: Name of the gymnasium environment
            seed: Random seed for reproducibility
        """
        self.env_name = env_name
        self.config = ENV_CONFIGS[env_name]
        self.env = gym.make(env_name)

        if seed is not None:
            self.env.reset(seed=seed)

        # Store original dimensions
        self.original_obs_dim = self.config.original_obs_dim
        self.original_act_dim = self.config.original_act_dim
        self.is_continuous = self.config.is_continuous

        # Standardized dimensions
        self.obs_dim = STANDARDIZED_OBS_DIM
        self.act_dim = STANDARDIZED_ACT_DIM

        # For continuous action spaces, we discretize
        if self.is_continuous:
            # MountainCarContinuous: action in [-1, 1]
            # Discretize to: left (-1), nothing (0), right (1)
            self.discrete_actions = [-1.0, 0.0, 1.0]

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment and return padded observation."""
        if seed is not None:
            obs, _ = self.env.reset(seed=seed)
        else:
            obs, _ = self.env.reset()
        return self._pad_observation(obs)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment.

        Args:
            action: Standardized action index (0 to STANDARDIZED_ACT_DIM-1)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Convert standardized action to environment-specific action
        env_action = self._convert_action(action)

        obs, reward, terminated, truncated, info = self.env.step(env_action)

        return self._pad_observation(obs), reward, terminated, truncated, info

    def _pad_observation(self, obs: np.ndarray) -> np.ndarray:
        """Pad observation to standardized dimension."""
        if len(obs) == self.obs_dim:
            return obs.astype(np.float32)

        padded = np.zeros(self.obs_dim, dtype=np.float32)
        padded[:len(obs)] = obs
        return padded

    def _convert_action(self, action: int) -> Any:
        """Convert standardized action to environment-specific action."""
        if self.is_continuous:
            # For MountainCarContinuous, convert discrete to continuous
            if action < len(self.discrete_actions):
                return np.array([self.discrete_actions[action]], dtype=np.float32)
            else:
                # Empty action - default to no action
                return np.array([0.0], dtype=np.float32)
        else:
            # For discrete action spaces
            if action < self.original_act_dim:
                return action
            else:
                # Empty action - return action 0 (should never happen in practice)
                return 0

    def get_valid_actions(self) -> list:
        """Get list of valid action indices for this environment."""
        return list(range(self.original_act_dim))

    def close(self):
        """Close the environment."""
        self.env.close()

    @property
    def convergence_threshold(self) -> float:
        """Get the convergence threshold for this environment."""
        return self.config.convergence_threshold


def create_env(env_name: str, seed: Optional[int] = None) -> StandardizedEnv:
    """Create a standardized environment."""
    return StandardizedEnv(env_name, seed)


def get_env_info(env_name: str) -> Dict[str, Any]:
    """Get information about an environment."""
    config = ENV_CONFIGS[env_name]
    return {
        "name": config.name,
        "original_obs_dim": config.original_obs_dim,
        "original_act_dim": config.original_act_dim,
        "standardized_obs_dim": STANDARDIZED_OBS_DIM,
        "standardized_act_dim": STANDARDIZED_ACT_DIM,
        "is_continuous": config.is_continuous,
        "convergence_threshold": config.convergence_threshold,
        "description": config.description,
    }


def list_environments() -> list:
    """List all available environment names."""
    return list(ENV_CONFIGS.keys())
