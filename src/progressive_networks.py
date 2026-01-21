"""
Progressive Networks implementation for Section 3.
Transfer learning from multiple source networks to a target network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import time
from typing import List, Tuple, Optional

from .utils import (
    DEVICE, STANDARDIZED_OBS_DIM, STANDARDIZED_ACT_DIM,
    TrainingConfig, TrainingStats, get_model_path,
    create_tensorboard_writer, compute_returns, log_to_tensorboard,
    check_convergence, print_training_stats, set_seed, init_weights
)
from .environments import create_env, StandardizedEnv
from .actor_critic import ActorCritic, Actor, Critic


class ProgressiveActor(nn.Module):
    """
    Progressive Actor network that combines frozen source networks
    with a trainable target network.

    Architecture per Rusu et al. 2016 (Progressive Neural Networks):
    - Source networks are frozen and provide hidden layer features
    - Target network has TWO hidden layers (matching source architecture)
    - Lateral connections: source layer 1 outputs feed into target layer 2
    - Formula: h2_target = f(W2 * h1_target + sum(U_j * h1_source_j))
    """

    def __init__(self, source_actors: List[nn.Module],
                 obs_dim: int = STANDARDIZED_OBS_DIM,
                 act_dim: int = STANDARDIZED_ACT_DIM,
                 hidden_dim: int = 128,
                 use_laterals: bool = True):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.num_sources = len(source_actors)
        self.use_laterals = use_laterals

        # Store as regular Python list to avoid PyTorch training interference
        # BUG FIX: Using nn.ModuleList for frozen modules breaks training!
        self._source_actors = source_actors
        for actor in self._source_actors:
            actor.eval()
            for param in actor.parameters():
                param.requires_grad = False

        # Target network - TWO hidden layers (matching source architecture)
        self.target_fc1 = nn.Linear(obs_dim, hidden_dim)
        self.target_fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Output layer
        self.fc3 = nn.Linear(hidden_dim, act_dim)

        # Lateral connections from each source's layer 1 to target's layer 2
        # U_j: hidden_dim -> hidden_dim for each source j
        if self.num_sources > 0 and use_laterals:
            self.lateral_fc2 = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_sources)
            ])
            # Initialize lateral connections with small weights
            for lateral in self.lateral_fc2:
                nn.init.xavier_uniform_(lateral.weight, gain=0.1)
                nn.init.zeros_(lateral.bias)

        # Initialize trainable weights
        init_weights(self.target_fc1)
        init_weights(self.target_fc2)
        init_weights(self.fc3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with progressive network architecture."""
        # Layer 1: target hidden
        h1_target = F.relu(self.target_fc1(x))

        if self.use_laterals and self.num_sources > 0:
            # Get source hidden activations (layer 1) - frozen, no gradients
            lateral_sum = torch.zeros_like(h1_target)
            with torch.no_grad():
                for i, actor in enumerate(self._source_actors):
                    h1_source = F.relu(actor.fc1(x))
                    lateral_sum = lateral_sum + self.lateral_fc2[i](h1_source)

            # Layer 2 with lateral connections:
            # h2 = f(W2 * h1_target + sum(U_j * h1_source_j))
            h2_target = F.relu(self.target_fc2(h1_target) + lateral_sum)
        else:
            # No lateral connections - standard forward pass
            h2_target = F.relu(self.target_fc2(h1_target))

        # Output layer
        return self.fc3(h2_target)

    def get_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Get target hidden layer activations (after both hidden layers)."""
        h1 = F.relu(self.target_fc1(x))
        return F.relu(self.target_fc2(h1))

    def get_entropy(self, state: torch.Tensor, valid_actions: list = None) -> torch.Tensor:
        """Compute entropy of the policy distribution."""
        logits = self.forward(state)

        if valid_actions is not None:
            mask = torch.ones(logits.shape[-1], device=logits.device) * float('-inf')
            mask[valid_actions] = 0
            logits = logits + mask

        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        return dist.entropy().mean()


class ProgressiveCritic(nn.Module):
    """
    Progressive Critic network that combines frozen source networks
    with a trainable target network.

    Architecture per Rusu et al. 2016 (Progressive Neural Networks):
    - Source networks are frozen and provide hidden layer features
    - Target network has TWO hidden layers (matching source architecture)
    - Lateral connections: source layer 1 outputs feed into target layer 2
    - Formula: h2_target = f(W2 * h1_target + sum(U_j * h1_source_j))
    """

    def __init__(self, source_critics: List[nn.Module],
                 obs_dim: int = STANDARDIZED_OBS_DIM,
                 hidden_dim: int = 128,
                 use_laterals: bool = True):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.num_sources = len(source_critics)
        self.use_laterals = use_laterals

        # Store as regular Python list to avoid PyTorch training interference
        # BUG FIX: Using nn.ModuleList for frozen modules breaks training!
        self._source_critics = source_critics
        for critic in self._source_critics:
            critic.eval()
            for param in critic.parameters():
                param.requires_grad = False

        # Target network - TWO hidden layers (matching source architecture)
        self.target_fc1 = nn.Linear(obs_dim, hidden_dim)
        self.target_fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Output layer
        self.fc3 = nn.Linear(hidden_dim, 1)

        # Lateral connections from each source's layer 1 to target's layer 2
        if self.num_sources > 0 and use_laterals:
            self.lateral_fc2 = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_sources)
            ])
            # Initialize lateral connections with small weights
            for lateral in self.lateral_fc2:
                nn.init.xavier_uniform_(lateral.weight, gain=0.1)
                nn.init.zeros_(lateral.bias)

        # Initialize trainable weights
        init_weights(self.target_fc1)
        init_weights(self.target_fc2)
        init_weights(self.fc3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with progressive network architecture."""
        # Layer 1: target hidden
        h1_target = F.relu(self.target_fc1(x))

        if self.use_laterals and self.num_sources > 0:
            # Get source hidden activations (layer 1) - frozen, no gradients
            lateral_sum = torch.zeros_like(h1_target)
            with torch.no_grad():
                for i, critic in enumerate(self._source_critics):
                    h1_source = F.relu(critic.fc1(x))
                    lateral_sum = lateral_sum + self.lateral_fc2[i](h1_source)

            # Layer 2 with lateral connections
            h2_target = F.relu(self.target_fc2(h1_target) + lateral_sum)
        else:
            # No lateral connections - standard forward pass
            h2_target = F.relu(self.target_fc2(h1_target))

        # Output layer
        return self.fc3(h2_target)


class ProgressiveActorCritic(nn.Module):
    """
    Progressive Actor-Critic that uses frozen source networks
    to assist training on a new target task.
    """

    def __init__(self, source_models: List[ActorCritic],
                 obs_dim: int = STANDARDIZED_OBS_DIM,
                 act_dim: int = STANDARDIZED_ACT_DIM,
                 hidden_dim: int = 128,
                 use_laterals: bool = True):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.use_laterals = use_laterals

        if use_laterals and len(source_models) > 0:
            # Use Progressive networks with lateral connections
            source_actors = [model.actor for model in source_models]
            source_critics = [model.critic for model in source_models]
            self.actor = ProgressiveActor(source_actors, obs_dim, act_dim, hidden_dim, use_laterals)
            self.critic = ProgressiveCritic(source_critics, obs_dim, hidden_dim, use_laterals)
        else:
            # Use standard Actor/Critic (known to work from Section 1)
            # This ensures baseline functionality when laterals are disabled
            self.actor = Actor(obs_dim, act_dim, hidden_dim)
            self.critic = Critic(obs_dim, hidden_dim)
            print("Using standard Actor/Critic (laterals disabled or no source models)")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returns (action_logits, value)."""
        return self.actor(x), self.critic(x)

    def get_action(self, state: np.ndarray, valid_actions: List[int] = None) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Select action and compute value."""
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

        return action.item(), log_prob, value.squeeze()

    def get_entropy(self, state: torch.Tensor, valid_actions: List[int] = None) -> torch.Tensor:
        """Compute entropy of the policy distribution."""
        action_logits = self.actor(state)

        if valid_actions is not None:
            mask = torch.ones(action_logits.shape[-1], device=DEVICE) * float('-inf')
            mask[valid_actions] = 0
            action_logits = action_logits + mask

        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)
        return dist.entropy().mean()


class ProgressiveNetworkTrainer:
    """Trainer for Progressive Networks."""

    def __init__(self, source_env_names: List[str], target_env_name: str,
                 config: TrainingConfig = None, use_laterals: bool = True):
        """
        Initialize progressive network trainer.

        Args:
            source_env_names: List of source environment names
            target_env_name: Target environment name
            config: Training configuration
            use_laterals: Whether to use lateral connections from source networks
        """
        self.source_env_names = source_env_names
        self.target_env_name = target_env_name
        self.config = config or TrainingConfig()
        self.use_laterals = use_laterals

        # Create target environment
        self.env = create_env(target_env_name)
        self.valid_actions = self.env.get_valid_actions()

        # Load source models
        source_models = self._load_source_models()

        # Create progressive network
        self.model = ProgressiveActorCritic(
            source_models,
            obs_dim=STANDARDIZED_OBS_DIM,
            act_dim=STANDARDIZED_ACT_DIM,
            hidden_dim=self.config.hidden_dim,
            use_laterals=use_laterals
        ).to(DEVICE)

        print(f"Progressive network created with use_laterals={use_laterals}")

        # Optimizer (only trains non-frozen parameters)
        # Use same learning rate as Section 1 - zero-initialized laterals won't interfere
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(trainable_params, lr=self.config.lr_actor)

        # Statistics
        self.stats = TrainingStats()

    def _load_source_models(self) -> List[ActorCritic]:
        """Load pre-trained source models."""
        source_models = []
        for env_name in self.source_env_names:
            model = ActorCritic(
                obs_dim=STANDARDIZED_OBS_DIM,
                act_dim=STANDARDIZED_ACT_DIM,
                hidden_dim=self.config.hidden_dim
            ).to(DEVICE)

            path = get_model_path(env_name)
            checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            # Freeze all parameters
            for param in model.parameters():
                param.requires_grad = False

            source_models.append(model)
            print(f"Loaded source model: {env_name}")

        return source_models

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
            entropy = self.model.get_entropy(state_tensor, self.valid_actions)
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
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 0.5
        )
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
        """Train the progressive network until convergence or max episodes."""
        set_seed(self.config.seed)

        if experiment_name is None:
            sources = "_".join([s.split("-")[0] for s in self.source_env_names])
            target = self.target_env_name.split("-")[0]
            experiment_name = f"section3_progressive_{sources}_to_{target}"

        writer = create_tensorboard_writer(experiment_name)
        start_time = time.time()

        rewards_history = []
        total_steps = 0

        print(f"\nProgressive Network: {self.source_env_names} -> {self.target_env_name}")
        print("=" * 60)

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

        sources_str = ", ".join(self.source_env_names)
        print_training_stats(self.stats, f"Progressive: {{{sources_str}}} -> {self.target_env_name}")

        return self.stats

    def save_model(self, path: str = None):
        """Save model to disk."""
        if path is None:
            sources = "_".join([s.split("-")[0] for s in self.source_env_names])
            target = self.target_env_name.split("-")[0]
            path = get_model_path(f"progressive_{sources}_to_{target}")

        from dataclasses import asdict
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': asdict(self.config),  # Convert to dict to avoid pickling issues
            'source_envs': self.source_env_names,
            'target_env': self.target_env_name,
        }, path)
        print(f"Model saved to {path}")


def train_progressive_network(source_env_names: List[str], target_env_name: str,
                               config: TrainingConfig = None,
                               use_laterals: bool = True) -> TrainingStats:
    """
    Train a progressive network from source environments to target environment.

    Args:
        source_env_names: List of source environment names
        target_env_name: Target environment name
        config: Training configuration
        use_laterals: Whether to use lateral connections from source networks

    Returns:
        Training statistics
    """
    trainer = ProgressiveNetworkTrainer(source_env_names, target_env_name, config, use_laterals)
    stats = trainer.train()
    trainer.env.close()
    return stats


def run_section3_experiments(config: TrainingConfig = None, use_laterals: bool = True) -> dict:
    """
    Run all Section 3 progressive network experiments.

    Experiments:
    1. {Acrobot, MountainCar} -> CartPole
    2. {CartPole, Acrobot} -> MountainCar

    Args:
        config: Training configuration
        use_laterals: Whether to use lateral connections (default True)

    Returns:
        Dictionary with training statistics for each experiment
    """
    # Print git commit for version verification
    import subprocess
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                         stderr=subprocess.DEVNULL).decode().strip()
        commit_msg = subprocess.check_output(['git', 'log', '-1', '--pretty=%s'],
                                             stderr=subprocess.DEVNULL).decode().strip()
        print(f"\n[Git] Running commit: {commit} - {commit_msg}")
    except:
        print("\n[Git] Could not get commit info")

    print(f"\n[Config] use_laterals={use_laterals}")

    results = {}

    # Experiment 1: {Acrobot, MountainCar} -> CartPole
    print("\n" + "=" * 70)
    print("Section 3 Experiment 1: {Acrobot-v1, MountainCarContinuous-v0} -> CartPole-v1")
    print("=" * 70)
    results['acrobot_mountaincar_to_cartpole'] = train_progressive_network(
        ["Acrobot-v1", "MountainCarContinuous-v0"],
        "CartPole-v1",
        config,
        use_laterals
    )

    # Experiment 2: {CartPole, Acrobot} -> MountainCar
    print("\n" + "=" * 70)
    print("Section 3 Experiment 2: {CartPole-v1, Acrobot-v1} -> MountainCarContinuous-v0")
    print("=" * 70)
    results['cartpole_acrobot_to_mountaincar'] = train_progressive_network(
        ["CartPole-v1", "Acrobot-v1"],
        "MountainCarContinuous-v0",
        config,
        use_laterals
    )

    return results
