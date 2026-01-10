"""
Stable Baselines3 baseline for comparing with custom implementation.
Uses PPO (similar to Actor-Critic) for fair comparison.
"""

import os
import time
import json
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from .utils import TrainingConfig, get_log_dir
from .environments import ENV_CONFIGS


@dataclass
class SB3TrainingStats:
    """Statistics from SB3 training."""
    env_name: str
    algorithm: str
    total_timesteps: int
    training_time: float
    final_avg_reward: float
    convergence_timestep: Optional[int] = None
    rewards_history: List[float] = None

    def __post_init__(self):
        if self.rewards_history is None:
            self.rewards_history = []


class ConvergenceCallback(BaseCallback):
    """
    Callback to track convergence and episode rewards.
    Similar to the custom implementation's convergence checking.
    """

    def __init__(self, convergence_threshold: float, check_freq: int = 1000,
                 window: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.convergence_threshold = convergence_threshold
        self.check_freq = check_freq
        self.window = window
        self.episode_rewards = []
        self.convergence_timestep = None
        self.converged = False

    def _on_step(self) -> bool:
        # Get episode rewards from the monitor wrapper
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'r' in info:
                    self.episode_rewards.append(info['r'])

        # Check convergence
        if not self.converged and len(self.episode_rewards) >= self.window:
            avg_reward = np.mean(self.episode_rewards[-self.window:])
            if avg_reward >= self.convergence_threshold:
                self.converged = True
                self.convergence_timestep = self.num_timesteps
                if self.verbose > 0:
                    print(f"Converged at timestep {self.num_timesteps} with avg reward {avg_reward:.2f}")

        return True


def train_sb3_baseline(
    env_name: str,
    algorithm: str = "PPO",
    total_timesteps: int = 100000,
    config: Optional[TrainingConfig] = None,
    log_tensorboard: bool = True,
    verbose: int = 1
) -> SB3TrainingStats:
    """
    Train an SB3 agent as baseline.

    Args:
        env_name: Gymnasium environment name
        algorithm: "PPO" or "A2C"
        total_timesteps: Total training timesteps
        config: Training configuration (for hyperparameters)
        log_tensorboard: Whether to log to TensorBoard
        verbose: Verbosity level

    Returns:
        Training statistics
    """
    if config is None:
        config = TrainingConfig()

    env_config = ENV_CONFIGS[env_name]

    # Create environment with Monitor wrapper for episode tracking
    def make_env():
        env = gym.make(env_name)
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])

    # Setup TensorBoard logging
    tensorboard_log = None
    if log_tensorboard:
        tensorboard_log = get_log_dir(f"sb3_{algorithm}_{env_name}")

    # Create algorithm
    policy_kwargs = dict(
        net_arch=[config.hidden_dim, config.hidden_dim]  # Match custom implementation
    )

    if algorithm == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config.lr_actor,
            gamma=config.gamma,
            ent_coef=config.entropy_coef,
            vf_coef=config.value_loss_coef,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=config.seed
        )
    elif algorithm == "A2C":
        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=config.lr_actor,
            gamma=config.gamma,
            ent_coef=config.entropy_coef,
            vf_coef=config.value_loss_coef,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=config.seed
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Setup convergence callback
    convergence_callback = ConvergenceCallback(
        convergence_threshold=env_config.convergence_threshold,
        verbose=verbose
    )

    # Train
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=convergence_callback)
    training_time = time.time() - start_time

    # Compute final average reward
    final_avg_reward = 0.0
    if len(convergence_callback.episode_rewards) >= 100:
        final_avg_reward = np.mean(convergence_callback.episode_rewards[-100:])
    elif len(convergence_callback.episode_rewards) > 0:
        final_avg_reward = np.mean(convergence_callback.episode_rewards)

    # Save model
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"sb3_{algorithm}_{env_name}")
    model.save(model_path)

    env.close()

    return SB3TrainingStats(
        env_name=env_name,
        algorithm=algorithm,
        total_timesteps=total_timesteps,
        training_time=training_time,
        final_avg_reward=final_avg_reward,
        convergence_timestep=convergence_callback.convergence_timestep,
        rewards_history=convergence_callback.episode_rewards
    )


def evaluate_sb3_model(
    model_path: str,
    env_name: str,
    n_episodes: int = 100,
    verbose: int = 1
) -> Dict[str, float]:
    """
    Evaluate a trained SB3 model.

    Args:
        model_path: Path to saved model
        env_name: Environment name
        n_episodes: Number of evaluation episodes
        verbose: Verbosity level

    Returns:
        Evaluation metrics
    """
    model = PPO.load(model_path)
    env = gym.make(env_name)

    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)

        if verbose > 0 and (ep + 1) % 20 == 0:
            print(f"Episode {ep + 1}/{n_episodes}: Reward = {total_reward:.2f}")

    env.close()

    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "min_reward": np.min(rewards),
        "max_reward": np.max(rewards)
    }


def run_all_baselines(
    total_timesteps: int = 100000,
    algorithm: str = "PPO",
    config: Optional[TrainingConfig] = None,
    verbose: int = 1
) -> Dict[str, SB3TrainingStats]:
    """
    Train SB3 baselines on all environments.

    Args:
        total_timesteps: Training timesteps per environment
        algorithm: "PPO" or "A2C"
        config: Training configuration
        verbose: Verbosity level

    Returns:
        Dictionary of environment name to training stats
    """
    results = {}

    for env_name in ENV_CONFIGS.keys():
        print(f"\n{'='*60}")
        print(f"Training SB3 {algorithm} on {env_name}")
        print(f"{'='*60}")

        stats = train_sb3_baseline(
            env_name=env_name,
            algorithm=algorithm,
            total_timesteps=total_timesteps,
            config=config,
            verbose=verbose
        )

        results[env_name] = stats

        print(f"\nResults for {env_name}:")
        print(f"  Training Time: {stats.training_time:.2f}s")
        print(f"  Final Avg Reward: {stats.final_avg_reward:.2f}")
        print(f"  Convergence Timestep: {stats.convergence_timestep}")
        print(f"  Total Episodes: {len(stats.rewards_history)}")

    return results


def compare_with_custom(
    custom_results_path: str,
    sb3_results: Dict[str, SB3TrainingStats],
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare SB3 results with custom implementation results.

    Args:
        custom_results_path: Path to custom implementation results JSON
        sb3_results: SB3 training results
        output_path: Optional path to save comparison

    Returns:
        Comparison dictionary
    """
    # Load custom results
    with open(custom_results_path, 'r') as f:
        custom_results = json.load(f)

    comparison = {}

    for env_name in ENV_CONFIGS.keys():
        if env_name in custom_results and env_name in sb3_results:
            custom = custom_results[env_name]
            sb3 = sb3_results[env_name]

            comparison[env_name] = {
                "custom": {
                    "training_time": custom.get("training_time", 0),
                    "final_avg_reward": custom.get("final_avg_reward", 0),
                    "convergence_episode": custom.get("convergence_episode"),
                    "total_episodes": custom.get("total_episodes", 0)
                },
                "sb3": {
                    "training_time": sb3.training_time,
                    "final_avg_reward": sb3.final_avg_reward,
                    "convergence_timestep": sb3.convergence_timestep,
                    "total_episodes": len(sb3.rewards_history)
                },
                "comparison": {
                    "time_ratio": sb3.training_time / max(custom.get("training_time", 1), 0.001),
                    "reward_diff": sb3.final_avg_reward - custom.get("final_avg_reward", 0)
                }
            }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)

    return comparison


def print_comparison(comparison: Dict[str, Any]):
    """Print formatted comparison between custom and SB3 implementations."""
    print("\n" + "="*80)
    print("COMPARISON: Custom Implementation vs Stable Baselines3")
    print("="*80)

    for env_name, data in comparison.items():
        print(f"\n{env_name}")
        print("-" * 40)
        print(f"{'Metric':<25} {'Custom':<15} {'SB3':<15}")
        print("-" * 40)

        custom = data["custom"]
        sb3 = data["sb3"]

        print(f"{'Training Time (s)':<25} {custom['training_time']:<15.2f} {sb3['training_time']:<15.2f}")
        print(f"{'Final Avg Reward':<25} {custom['final_avg_reward']:<15.2f} {sb3['final_avg_reward']:<15.2f}")
        print(f"{'Total Episodes':<25} {custom['total_episodes']:<15} {sb3['total_episodes']:<15}")

        if custom.get('convergence_episode'):
            print(f"{'Convergence Episode':<25} {custom['convergence_episode']:<15}")
        if sb3.get('convergence_timestep'):
            print(f"{'Convergence Timestep':<25} {'':<15} {sb3['convergence_timestep']:<15}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SB3 baselines for comparison")
    parser.add_argument("--env", type=str, default=None, help="Specific environment to train")
    parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "A2C"])
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--compare", type=str, default=None, help="Path to custom results JSON for comparison")
    args = parser.parse_args()

    config = TrainingConfig()

    if args.env:
        # Train single environment
        stats = train_sb3_baseline(
            env_name=args.env,
            algorithm=args.algorithm,
            total_timesteps=args.timesteps,
            config=config
        )
        print(f"\nFinal Results:")
        print(f"  Training Time: {stats.training_time:.2f}s")
        print(f"  Final Avg Reward: {stats.final_avg_reward:.2f}")
        print(f"  Convergence Timestep: {stats.convergence_timestep}")
    else:
        # Train all environments
        results = run_all_baselines(
            total_timesteps=args.timesteps,
            algorithm=args.algorithm,
            config=config
        )

        if args.compare:
            comparison = compare_with_custom(args.compare, results)
            print_comparison(comparison)
