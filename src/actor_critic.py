"""
Actor-Critic implementation for DRL Assignment 3.
Supports standardized input/output dimensions for transfer learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import time
from typing import Optional, Tuple, List

from .utils import (
    DEVICE, STANDARDIZED_OBS_DIM, STANDARDIZED_ACT_DIM,
    TrainingConfig, TrainingStats, init_weights, get_model_path,
    create_tensorboard_writer, compute_returns, log_to_tensorboard,
    check_convergence, print_training_stats, set_seed, moving_average
)
from .environments import create_env, StandardizedEnv


class Actor(nn.Module):
    """Actor network for policy."""

    def __init__(self, obs_dim: int = STANDARDIZED_OBS_DIM,
                 act_dim: int = STANDARDIZED_ACT_DIM,
                 hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, act_dim)

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returns action logits."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def get_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Get hidden layer activations (for progressive networks)."""
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))

    def get_action(self, state: np.ndarray, valid_actions: List[int] = None) -> Tuple[int, torch.Tensor]:
        """
        Select action using the policy.

        Args:
            state: Current state
            valid_actions: List of valid action indices (for masking)

        Returns:
            Tuple of (action, log_probability)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        logits = self.forward(state_tensor)

        # Mask invalid actions if specified
        if valid_actions is not None:
            mask = torch.ones(self.act_dim, device=DEVICE) * float('-inf')
            mask[valid_actions] = 0
            logits = logits + mask

        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.squeeze()

    def get_entropy(self, state: torch.Tensor, valid_actions: List[int] = None) -> torch.Tensor:
        """Compute entropy of the policy distribution."""
        logits = self.forward(state)

        if valid_actions is not None:
            mask = torch.ones(logits.shape[-1], device=DEVICE) * float('-inf')
            mask[valid_actions] = 0
            logits = logits + mask

        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        return dist.entropy().mean()


class Critic(nn.Module):
    """Critic network for value function."""

    def __init__(self, obs_dim: int = STANDARDIZED_OBS_DIM,
                 hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returns state value."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def get_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Get hidden layer activations (for progressive networks)."""
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))


class ActorCritic(nn.Module):
    """Combined Actor-Critic architecture."""

    def __init__(self, obs_dim: int = STANDARDIZED_OBS_DIM,
                 act_dim: int = STANDARDIZED_ACT_DIM,
                 hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim

        self.actor = Actor(obs_dim, act_dim, hidden_dim)
        self.critic = Critic(obs_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returns (action_logits, value)."""
        return self.actor(x), self.critic(x)

    def get_action(self, state: np.ndarray, valid_actions: List[int] = None) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select action and compute value.

        Returns:
            Tuple of (action, log_probability, state_value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        action_logits, value = self.forward(state_tensor)

        # Mask invalid actions
        if valid_actions is not None:
            mask = torch.ones(self.act_dim, device=DEVICE) * float('-inf')
            mask[valid_actions] = 0
            action_logits = action_logits + mask

        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.squeeze(), value.squeeze()

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor,
                 valid_actions: List[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate states and actions for training.

        Returns:
            Tuple of (log_probs, values, entropy)
        """
        action_logits, values = self.forward(states)

        if valid_actions is not None:
            mask = torch.ones(action_logits.shape[-1], device=DEVICE) * float('-inf')
            mask[valid_actions] = 0
            action_logits = action_logits + mask

        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        return log_probs, values.squeeze(), entropy

    def get_actor_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Get actor hidden layer activations."""
        return self.actor.get_hidden(x)

    def get_critic_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Get critic hidden layer activations."""
        return self.critic.get_hidden(x)


class ActorCriticTrainer:
    """Trainer for Actor-Critic agent."""

    def __init__(self, env: StandardizedEnv, config: TrainingConfig = None):
        self.env = env
        self.config = config or TrainingConfig()
        self.valid_actions = env.get_valid_actions()

        # Initialize networks
        self.model = ActorCritic(
            obs_dim=STANDARDIZED_OBS_DIM,
            act_dim=STANDARDIZED_ACT_DIM,
            hidden_dim=self.config.hidden_dim
        ).to(DEVICE)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr_actor)

        # Statistics
        self.stats = TrainingStats()

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
            experiment_name = f"section1_{self.env.env_name}"

        writer = create_tensorboard_writer(experiment_name)
        start_time = time.time()

        rewards_history = []
        total_steps = 0

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

                # Continue training a bit after convergence to ensure stability
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
        print_training_stats(self.stats, self.env.env_name)

        return self.stats

    def save_model(self, path: str = None):
        """Save model to disk."""
        if path is None:
            path = get_model_path(self.env.env_name)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'env_name': self.env.env_name,
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str = None):
        """Load model from disk."""
        if path is None:
            path = get_model_path(self.env.env_name)

        checkpoint = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")


def train_individual_network(env_name: str, config: TrainingConfig = None) -> TrainingStats:
    """Train an individual network for a specific environment."""
    env = create_env(env_name)
    trainer = ActorCriticTrainer(env, config)
    stats = trainer.train()
    env.close()
    return stats


def load_trained_model(env_name: str) -> ActorCritic:
    """Load a trained model for a specific environment."""
    model = ActorCritic().to(DEVICE)
    path = get_model_path(env_name)
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model
