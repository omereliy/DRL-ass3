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
        original_act_dim=2,  # Discretized: left, right (no no-op to prevent reward hacking)
        is_continuous=True,  # Original is continuous
        convergence_threshold=0.0,  # Achievable with 2 discrete actions
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
            # Discretize to: left (-1), right (1) - no no-op to prevent reward hacking
            self.discrete_actions = [-1.0, 1.0]

        # Reward shaping state for MountainCarContinuous
        self._max_position = -float('inf')
        self._prev_position = None
        self._prev_velocity = None

        # Reward shaping state for Acrobot
        self._max_tip_height = -float('inf')

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment and return padded observation."""
        if seed is not None:
            obs, _ = self.env.reset(seed=seed)
        else:
            obs, _ = self.env.reset()

        # Reset reward shaping state for MountainCarContinuous
        if self.env_name == "MountainCarContinuous-v0":
            self._max_position = obs[0]  # Initial position
            self._prev_position = obs[0]
            self._prev_velocity = obs[1]  # Initial velocity

        # Reset reward shaping state for Acrobot
        if self.env_name == "Acrobot-v1":
            self._max_tip_height = self._compute_acrobot_tip_height(obs)

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

        # Apply reward shaping for MountainCarContinuous
        if self.env_name == "MountainCarContinuous-v0":
            reward = self._shape_mountaincar_reward(obs, reward, terminated)

        # Apply reward shaping for Acrobot
        if self.env_name == "Acrobot-v1":
            reward = self._shape_acrobot_reward(obs, reward, terminated)

        return self._pad_observation(obs), reward, terminated, truncated, info

    def _compute_acrobot_tip_height(self, obs: np.ndarray) -> float:
        """
        Compute the height of the Acrobot tip from the observation.

        Acrobot observation:
        - obs[0]: cos(theta1)
        - obs[1]: sin(theta1)
        - obs[2]: cos(theta2)
        - obs[3]: sin(theta2)
        - obs[4]: angular velocity of link 1
        - obs[5]: angular velocity of link 2

        The tip height is computed as: -cos(theta1) - cos(theta1 + theta2)
        Link lengths are both 1.0 in the environment.
        """
        cos_theta1 = obs[0]
        sin_theta1 = obs[1]
        cos_theta2 = obs[2]
        sin_theta2 = obs[3]

        # cos(theta1 + theta2) = cos(theta1)*cos(theta2) - sin(theta1)*sin(theta2)
        cos_theta1_plus_theta2 = cos_theta1 * cos_theta2 - sin_theta1 * sin_theta2

        # Height of tip (goal is height > 1.0, which means tip is above the base)
        # Height = -cos(theta1) - cos(theta1 + theta2)
        tip_height = -cos_theta1 - cos_theta1_plus_theta2

        return tip_height

    def _shape_acrobot_reward(self, obs: np.ndarray, original_reward: float,
                               terminated: bool) -> float:
        """
        Apply reward shaping for Acrobot using Potential-Based Reward Shaping.

        PBRS: R'(s,a,s') = R(s,a,s') + gamma * Phi(s') - Phi(s)

        This provides denser reward signal while preserving optimal policy.
        The potential is based on tip height (goal is height > 1.0).
        """
        tip_height = self._compute_acrobot_tip_height(obs)
        gamma = 0.99

        # Potential function: normalized tip height
        # Tip height ranges from -2.0 to +2.0, normalize to [0, 1]
        current_potential = (tip_height + 2.0) / 4.0
        prev_potential = (self._max_tip_height + 2.0) / 4.0

        # PBRS shaping
        shaping_reward = gamma * current_potential - prev_potential

        # Update max tip height for next step
        self._max_tip_height = tip_height

        # Apply shaping with moderate scale
        shaped_reward = original_reward + shaping_reward * 10.0

        return shaped_reward

    def _shape_mountaincar_reward(self, obs: np.ndarray, original_reward: float,
                                   terminated: bool) -> float:
        """
        Apply enhanced reward shaping for MountainCarContinuous.

        Uses Potential-Based Reward Shaping (PBRS) with:
        - Position potential: rewards progress toward goal
        - Velocity potential: rewards momentum building
        - Milestone bonuses: extra reward for reaching new maximum heights

        MountainCarContinuous state:
        - obs[0]: position (range: -1.2 to 0.6, goal: >= 0.45)
        - obs[1]: velocity (range: -0.07 to 0.07)
        """
        position = obs[0]
        velocity = obs[1]
        gamma = 0.99

        # Position potential: normalized position (higher = closer to goal)
        # Phi_pos(s) = (position + 1.2) / 1.8, maps [-1.2, 0.6] to [0, 1]
        current_pos_potential = (position + 1.2) / 1.8
        prev_pos_potential = (self._prev_position + 1.2) / 1.8 if self._prev_position is not None else current_pos_potential

        # Velocity potential: reward for momentum in the right direction
        # When on the left side of valley (position < -0.5), rightward velocity is good
        # When on the right side, leftward velocity helps build momentum for next swing
        # Simplified: reward absolute velocity (momentum building)
        velocity_scale = 0.3 / 0.07  # Normalize velocity to ~[0, 0.3] range
        current_vel_potential = abs(velocity) * velocity_scale
        prev_vel_potential = abs(self._prev_velocity) * velocity_scale if self._prev_velocity is not None else current_vel_potential

        # Combined potential
        current_potential = current_pos_potential + current_vel_potential
        prev_potential = prev_pos_potential + prev_vel_potential

        # PBRS: gamma * Phi(s') - Phi(s)
        shaping_reward = gamma * current_potential - prev_potential

        # Milestone bonus: reward for reaching new maximum heights
        milestone_bonus = 0.0
        if position > self._max_position:
            milestone_bonus = (position - self._max_position) * 10.0
            self._max_position = position

        # Apply shaping with moderate scale factor
        shaped_reward = original_reward + shaping_reward * 5.0 + milestone_bonus

        # Update previous state for next step
        self._prev_position = position
        self._prev_velocity = velocity

        return shaped_reward

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
