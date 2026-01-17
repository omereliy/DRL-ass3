"""
Fine-tuning implementation for Section 2.
Fine-tune a pre-trained model on a new target environment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import time
from typing import Optional, Tuple

from .utils import (
    DEVICE, STANDARDIZED_OBS_DIM, STANDARDIZED_ACT_DIM,
    TrainingConfig, TrainingStats, get_model_path,
    create_tensorboard_writer, compute_returns, log_to_tensorboard,
    check_convergence, print_training_stats, set_seed, init_weights
)
from .environments import create_env, StandardizedEnv
from .actor_critic import ActorCritic, load_trained_model


class FineTuningTrainer:
    """
    Trainer for fine-tuning a pre-trained model on a new target environment.

    Procedure:
    1. Load the model fully trained on the source environment
    2. Re-initialize the weights of the output layer
    3. Train the new network on the target environment
    """

    def __init__(self, source_env_name: str, target_env_name: str,
                 config: TrainingConfig = None):
        """
        Initialize fine-tuning trainer.

        Args:
            source_env_name: Name of source environment (for loading pre-trained model)
            target_env_name: Name of target environment (for training)
            config: Training configuration
        """
        self.source_env_name = source_env_name
        self.target_env_name = target_env_name
        self.config = config or TrainingConfig()

        # Create target environment
        self.env = create_env(target_env_name)
        self.valid_actions = self.env.get_valid_actions()

        # Load pre-trained model from source
        self.model = self._load_and_prepare_model()

        # Optimizer (only train the model, not frozen parts)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr_actor)

        # Statistics
        self.stats = TrainingStats()

    def _load_and_prepare_model(self) -> ActorCritic:
        """Load pre-trained model and re-initialize output layers."""
        # Load the pre-trained model
        model = ActorCritic(
            obs_dim=STANDARDIZED_OBS_DIM,
            act_dim=STANDARDIZED_ACT_DIM,
            hidden_dim=self.config.hidden_dim
        ).to(DEVICE)

        source_path = get_model_path(self.source_env_name)
        checkpoint = torch.load(source_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Loaded pre-trained model from {self.source_env_name}")

        # Re-initialize output layers (fc3 in actor and critic)
        print("Re-initializing output layer weights...")

        # Re-initialize actor output layer
        init_weights(model.actor.fc3)

        # Re-initialize critic output layer
        init_weights(model.critic.fc3)

        return model

    def train_episode(self) -> Tuple[float, dict]:
        """Train for one episode."""
        state = self.env.reset()
        log_probs = []
        values = []
        rewards = []
        entropies = []
        done = False
        steps = 0

        while not done and steps < self.config.max_steps:
            action, log_prob, value = self.model.get_action(state, self.valid_actions)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            # Compute entropy for this state
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            entropy = self.model.actor.get_entropy(state_tensor, self.valid_actions)
            entropies.append(entropy)

            state = next_state
            steps += 1

        # Compute returns
        returns = compute_returns(rewards, self.config.gamma)

        # Convert to tensors
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)

        # Compute advantages and normalize
        advantages = returns - values.detach()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute losses
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, returns)

        # Entropy bonus (average over trajectory)
        entropy = torch.stack(entropies).mean()

        # Total loss
        loss = actor_loss + self.config.value_loss_coef * critic_loss - self.config.entropy_coef * entropy

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        episode_reward = sum(rewards)
        return episode_reward, {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
            'total_loss': loss.item(),
            'steps': steps
        }

    def train(self, experiment_name: str = None) -> TrainingStats:
        """Train the agent until convergence or max episodes."""
        set_seed(self.config.seed)

        if experiment_name is None:
            experiment_name = f"section2_finetune_{self.source_env_name}_to_{self.target_env_name}"

        writer = create_tensorboard_writer(experiment_name)
        start_time = time.time()

        rewards_history = []
        total_steps = 0

        print(f"\nFine-tuning {self.source_env_name} -> {self.target_env_name}")
        print("=" * 50)

        for episode in range(self.config.max_episodes):
            episode_reward, losses = self.train_episode()
            rewards_history.append(episode_reward)
            total_steps += losses['steps']

            # Compute moving average
            avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)

            # Log to TensorBoard
            log_to_tensorboard(
                writer, episode,
                reward=episode_reward,
                loss=losses['total_loss'],
                actor_loss=losses['actor_loss'],
                critic_loss=losses['critic_loss'],
                entropy=losses['entropy'],
                avg_reward=avg_reward
            )

            # Print progress
            if episode % self.config.log_interval == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}, "
                      f"Avg Reward: {avg_reward:.2f}, Loss: {losses['total_loss']:.4f}")

            # Check convergence
            if check_convergence(rewards_history, self.env.convergence_threshold):
                if self.stats.convergence_episode is None:
                    self.stats.convergence_episode = episode
                    print(f"\nConverged at episode {episode}!")

                # Continue training a bit after convergence
                if episode - self.stats.convergence_episode >= 100:
                    break

            # Save checkpoint
            if episode % self.config.save_interval == 0:
                self.save_model()

        # Final save
        self.save_model()

        # Update statistics
        self.stats.total_episodes = len(rewards_history)
        self.stats.total_steps = total_steps
        self.stats.training_time = time.time() - start_time
        self.stats.final_avg_reward = np.mean(rewards_history[-100:])
        self.stats.rewards_history = rewards_history

        writer.close()
        print_training_stats(self.stats, f"{self.source_env_name} -> {self.target_env_name}")

        return self.stats

    def save_model(self, path: str = None):
        """Save model to disk."""
        from dataclasses import asdict
        if path is None:
            path = get_model_path(f"finetuned_{self.source_env_name}_to_{self.target_env_name}")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': asdict(self.config),  # Convert to dict to avoid pickling issues
            'source_env': self.source_env_name,
            'target_env': self.target_env_name,
        }, path)
        print(f"Model saved to {path}")


def fine_tune(source_env_name: str, target_env_name: str,
              config: TrainingConfig = None) -> TrainingStats:
    """
    Fine-tune a pre-trained model from source environment to target environment.

    Args:
        source_env_name: Name of source environment
        target_env_name: Name of target environment
        config: Training configuration

    Returns:
        Training statistics
    """
    trainer = FineTuningTrainer(source_env_name, target_env_name, config)
    stats = trainer.train()
    trainer.env.close()
    return stats


def run_section2_experiments(config: TrainingConfig = None) -> dict:
    """
    Run all Section 2 fine-tuning experiments.

    Experiments:
    1. Acrobot -> CartPole
    2. CartPole -> MountainCar

    Returns:
        Dictionary with training statistics for each experiment
    """
    results = {}

    # Experiment 1: Acrobot -> CartPole
    print("\n" + "=" * 60)
    print("Section 2 Experiment 1: Acrobot-v1 -> CartPole-v1")
    print("=" * 60)
    results['acrobot_to_cartpole'] = fine_tune(
        "Acrobot-v1", "CartPole-v1", config
    )

    # Experiment 2: CartPole -> MountainCar
    print("\n" + "=" * 60)
    print("Section 2 Experiment 2: CartPole-v1 -> MountainCarContinuous-v0")
    print("=" * 60)
    results['cartpole_to_mountaincar'] = fine_tune(
        "CartPole-v1", "MountainCarContinuous-v0", config
    )

    return results
