"""
Optimized Actor-Critic for faster training on GPU.
Uses vectorized environments and batch updates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import time
from typing import Optional, Tuple, List
import multiprocessing as mp

from .utils import (
    DEVICE, STANDARDIZED_OBS_DIM, STANDARDIZED_ACT_DIM,
    TrainingConfig, TrainingStats, init_weights, get_model_path,
    create_tensorboard_writer, log_to_tensorboard,
    check_convergence, print_training_stats, set_seed
)
from .environments import create_env, StandardizedEnv, ENV_CONFIGS


class ActorCriticFast(nn.Module):
    """Optimized Actor-Critic with shared backbone."""

    def __init__(self, obs_dim: int = STANDARDIZED_OBS_DIM,
                 act_dim: int = STANDARDIZED_ACT_DIM,
                 hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head
        self.actor = nn.Linear(hidden_dim, act_dim)

        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(x)
        return self.actor(features), self.critic(features)

    def get_action(self, state: np.ndarray, valid_actions: List[int] = None) -> Tuple[int, torch.Tensor, torch.Tensor]:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        logits, value = self.forward(state_tensor)

        if valid_actions is not None:
            mask = torch.ones(self.act_dim, device=DEVICE) * float('-inf')
            mask[valid_actions] = 0
            logits = logits + mask

        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.squeeze(), value.squeeze()

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor,
                         valid_actions: List[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(states)

        if valid_actions is not None:
            mask = torch.ones(logits.shape[-1], device=DEVICE) * float('-inf')
            mask[valid_actions] = 0
            logits = logits + mask

        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        return log_probs, values.squeeze(-1), entropy

    def get_hidden(self, x: torch.Tensor) -> torch.Tensor:
        return self.shared(x)


class ParallelEnvs:
    """Run multiple environment instances in parallel."""

    def __init__(self, env_name: str, num_envs: int = 4, seed: int = 42):
        self.env_name = env_name
        self.num_envs = num_envs
        self.envs = [create_env(env_name, seed + i) for i in range(num_envs)]
        self.valid_actions = self.envs[0].get_valid_actions()
        self.obs_dim = STANDARDIZED_OBS_DIM
        self.act_dim = STANDARDIZED_ACT_DIM

    def reset(self) -> np.ndarray:
        states = np.array([env.reset() for env in self.envs])
        return states

    def step(self, actions: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        states = np.array([r[0] for r in results])
        rewards = np.array([r[1] for r in results])
        terminated = np.array([r[2] for r in results])
        truncated = np.array([r[3] for r in results])
        return states, rewards, terminated, truncated

    def reset_done(self, dones: np.ndarray, states: np.ndarray) -> np.ndarray:
        for i, done in enumerate(dones):
            if done:
                states[i] = self.envs[i].reset()
        return states

    @property
    def convergence_threshold(self) -> float:
        return ENV_CONFIGS[self.env_name].convergence_threshold

    def close(self):
        for env in self.envs:
            env.close()


class FastTrainer:
    """Fast trainer using parallel environments and batch updates."""

    def __init__(self, env_name: str, config: TrainingConfig = None,
                 num_envs: int = 8, rollout_steps: int = 128):
        self.env_name = env_name
        self.config = config or TrainingConfig()
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps

        # Parallel environments
        self.envs = ParallelEnvs(env_name, num_envs, self.config.seed)
        self.valid_actions = self.envs.valid_actions

        # Model
        self.model = ActorCriticFast(
            hidden_dim=self.config.hidden_dim
        ).to(DEVICE)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr_actor)

        # Statistics
        self.stats = TrainingStats()

    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                    dones: torch.Tensor, next_value: torch.Tensor,
                    gamma: float = 0.99, lam: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns

    def collect_rollout(self, states: np.ndarray) -> Tuple[dict, np.ndarray, list]:
        """Collect rollout from parallel environments."""
        rollout = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'dones': []
        }
        episode_rewards = []
        current_episode_rewards = [0.0] * self.num_envs

        for _ in range(self.rollout_steps):
            states_tensor = torch.FloatTensor(states).to(DEVICE)

            with torch.no_grad():
                logits, values = self.model(states_tensor)

                # Mask invalid actions
                mask = torch.ones(self.model.act_dim, device=DEVICE) * float('-inf')
                mask[self.valid_actions] = 0
                logits = logits + mask

                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            # Step environments
            actions_np = actions.cpu().numpy()
            next_states, rewards, terminated, truncated = self.envs.step(actions_np)
            dones = terminated | truncated

            # Store rollout data
            rollout['states'].append(states_tensor)
            rollout['actions'].append(actions)
            rollout['log_probs'].append(log_probs)
            rollout['rewards'].append(torch.FloatTensor(rewards).to(DEVICE))
            rollout['values'].append(values.squeeze(-1))
            rollout['dones'].append(torch.FloatTensor(dones).to(DEVICE))

            # Track episode rewards
            for i in range(self.num_envs):
                current_episode_rewards[i] += rewards[i]
                if dones[i]:
                    episode_rewards.append(current_episode_rewards[i])
                    current_episode_rewards[i] = 0.0

            # Reset done environments
            states = self.envs.reset_done(dones, next_states)

        # Stack rollout tensors
        for key in rollout:
            rollout[key] = torch.stack(rollout[key])

        return rollout, states, episode_rewards

    def update(self, rollout: dict) -> dict:
        """Perform PPO-style update on collected rollout."""
        states = rollout['states'].view(-1, self.model.obs_dim)
        actions = rollout['actions'].view(-1)
        old_log_probs = rollout['log_probs'].view(-1)
        rewards = rollout['rewards']
        values = rollout['values']
        dones = rollout['dones']

        # Compute next value for GAE
        with torch.no_grad():
            _, next_value = self.model(states[-self.num_envs:])
            next_value = next_value.squeeze(-1)

        # Compute advantages
        advantages, returns = self.compute_gae(
            rewards.view(-1), values.view(-1),
            dones.view(-1), next_value.mean(),
            self.config.gamma
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update
        log_probs, new_values, entropy = self.model.evaluate_actions(
            states, actions, self.valid_actions
        )

        # Actor loss
        ratio = torch.exp(log_probs - old_log_probs.view(-1))
        clip_range = 0.2
        actor_loss1 = -advantages * ratio
        actor_loss2 = -advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        actor_loss = torch.max(actor_loss1, actor_loss2).mean()

        # Critic loss
        critic_loss = F.mse_loss(new_values, returns)

        # Total loss
        loss = actor_loss + self.config.value_loss_coef * critic_loss - self.config.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
            'total_loss': loss.item()
        }

    def train(self, experiment_name: str = None) -> TrainingStats:
        """Train the agent."""
        set_seed(self.config.seed)

        if experiment_name is None:
            experiment_name = f"fast_{self.env_name}"

        writer = create_tensorboard_writer(experiment_name)
        start_time = time.time()

        states = self.envs.reset()
        all_episode_rewards = []
        total_updates = 0
        max_updates = self.config.max_episodes * 5 // self.num_envs  # Approximate

        while total_updates < max_updates:
            rollout, states, episode_rewards = self.collect_rollout(states)
            losses = self.update(rollout)

            all_episode_rewards.extend(episode_rewards)
            total_updates += 1

            if episode_rewards:
                avg_reward = np.mean(all_episode_rewards[-100:]) if len(all_episode_rewards) >= 100 else np.mean(all_episode_rewards)

                log_to_tensorboard(
                    writer, len(all_episode_rewards),
                    reward=np.mean(episode_rewards),
                    loss=losses['total_loss'],
                    actor_loss=losses['actor_loss'],
                    critic_loss=losses['critic_loss'],
                    entropy=losses['entropy'],
                    avg_reward=avg_reward
                )

                if total_updates % 10 == 0:
                    print(f"Update {total_updates}, Episodes: {len(all_episode_rewards)}, "
                          f"Avg Reward: {avg_reward:.2f}, Loss: {losses['total_loss']:.4f}")

                # Check convergence
                if check_convergence(all_episode_rewards, self.envs.convergence_threshold):
                    if self.stats.convergence_episode is None:
                        self.stats.convergence_episode = len(all_episode_rewards)
                        print(f"\nConverged at episode {len(all_episode_rewards)}!")

                    if len(all_episode_rewards) - self.stats.convergence_episode >= 100:
                        break

        # Save model
        self.save_model()

        # Update statistics
        self.stats.total_episodes = len(all_episode_rewards)
        self.stats.total_steps = total_updates * self.rollout_steps * self.num_envs
        self.stats.training_time = time.time() - start_time
        self.stats.final_avg_reward = np.mean(all_episode_rewards[-100:]) if all_episode_rewards else 0
        self.stats.rewards_history = all_episode_rewards

        writer.close()
        print_training_stats(self.stats, self.env_name)

        return self.stats

    def save_model(self, path: str = None):
        if path is None:
            path = get_model_path(self.env_name)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'env_name': self.env_name,
        }, path)
        print(f"Model saved to {path}")


def train_fast(env_name: str, config: TrainingConfig = None,
               num_envs: int = 8) -> TrainingStats:
    """Train using fast parallel trainer."""
    trainer = FastTrainer(env_name, config, num_envs=num_envs)
    stats = trainer.train()
    trainer.envs.close()
    return stats
