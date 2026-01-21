#!/usr/bin/env python3
"""
Main training script for DRL Assignment 3: Meta and Transfer Learning

This script runs all three sections of the assignment:
- Section 1: Train individual networks for CartPole, Acrobot, and MountainCar
- Section 2: Fine-tune pre-trained models on new tasks
- Section 3: Progressive Networks transfer learning

Usage:
    python train.py --section all    # Run all sections
    python train.py --section 1      # Run Section 1 only
    python train.py --section 2      # Run Section 2 only
    python train.py --section 3      # Run Section 3 only

TensorBoard:
    tensorboard --logdir logs/
"""

import argparse
import json
import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import TrainingConfig, TrainingStats, DEVICE
from src.environments import list_environments, get_env_info
from src.actor_critic import train_individual_network
from src.fine_tuning import run_section2_experiments
from src.progressive_networks import run_section3_experiments


def run_section1(config: TrainingConfig) -> dict:
    """
    Section 1: Train individual networks for each environment.

    Requirements:
    - Train actor-critic for CartPole-v1, Acrobot-v1, and MountainCarContinuous-v0
    - Standardized input/output dimensions for all environments
    - At least one hidden layer in each network
    """
    print("\n" + "=" * 70)
    print("SECTION 1: Training Individual Networks")
    print("=" * 70)

    results = {}
    environments = ["CartPole-v1", "Acrobot-v1", "MountainCarContinuous-v0"]

    for env_name in environments:
        print(f"\n{'='*60}")
        print(f"Training on {env_name}")
        print(f"{'='*60}")

        env_info = get_env_info(env_name)
        print(f"Environment Info:")
        print(f"  Original obs dim: {env_info['original_obs_dim']}")
        print(f"  Original act dim: {env_info['original_act_dim']}")
        print(f"  Standardized obs dim: {env_info['standardized_obs_dim']}")
        print(f"  Standardized act dim: {env_info['standardized_act_dim']}")
        print(f"  Convergence threshold: {env_info['convergence_threshold']}")
        print()

        stats = train_individual_network(env_name, config)
        results[env_name] = {
            'total_episodes': stats.total_episodes,
            'total_steps': stats.total_steps,
            'training_time': stats.training_time,
            'final_avg_reward': stats.final_avg_reward,
            'convergence_episode': stats.convergence_episode,
        }

    return results


def run_section2(config: TrainingConfig) -> dict:
    """
    Section 2: Fine-tune pre-trained models on new tasks.

    Tasks:
    1. Acrobot -> CartPole
    2. CartPole -> MountainCar

    Procedure:
    - Take model trained on source
    - Re-initialize output layer weights
    - Train on target
    """
    print("\n" + "=" * 70)
    print("SECTION 2: Fine-tuning Pre-trained Models")
    print("=" * 70)

    results = run_section2_experiments(config)

    # Convert to serializable format
    serializable_results = {}
    for key, stats in results.items():
        serializable_results[key] = {
            'total_episodes': stats.total_episodes,
            'total_steps': stats.total_steps,
            'training_time': stats.training_time,
            'final_avg_reward': stats.final_avg_reward,
            'convergence_episode': stats.convergence_episode,
        }

    return serializable_results


def run_section3(config: TrainingConfig) -> dict:
    """
    Section 3: Progressive Networks transfer learning.

    Tasks:
    1. {Acrobot, MountainCar} -> CartPole
    2. {CartPole, Acrobot} -> MountainCar

    Procedure:
    - Use frozen source networks
    - Connect hidden layers of sources to target output via lateral connections
    """
    print("\n" + "=" * 70)
    print("SECTION 3: Progressive Networks Transfer Learning")
    print("=" * 70)

    results = run_section3_experiments(config)

    # Convert to serializable format
    serializable_results = {}
    for key, stats in results.items():
        serializable_results[key] = {
            'total_episodes': stats.total_episodes,
            'total_steps': stats.total_steps,
            'training_time': stats.training_time,
            'final_avg_reward': stats.final_avg_reward,
            'convergence_episode': stats.convergence_episode,
        }

    return serializable_results


def print_comparison_table(section1_results: dict, section2_results: dict, section3_results: dict):
    """Print a comparison table of all results."""
    print("\n" + "=" * 90)
    print("RESULTS COMPARISON")
    print("=" * 90)

    print("\n--- Section 1: Individual Training ---")
    print(f"{'Environment':<35} {'Episodes':<12} {'Time (s)':<12} {'Avg Reward':<15}")
    print("-" * 75)
    for env, stats in section1_results.items():
        print(f"{env:<35} {stats['total_episodes']:<12} {stats['training_time']:<12.2f} {stats['final_avg_reward']:<15.2f}")

    if section2_results:
        print("\n--- Section 2: Fine-tuning ---")
        print(f"{'Transfer':<35} {'Episodes':<12} {'Time (s)':<12} {'Avg Reward':<15}")
        print("-" * 75)
        for transfer, stats in section2_results.items():
            print(f"{transfer:<35} {stats['total_episodes']:<12} {stats['training_time']:<12.2f} {stats['final_avg_reward']:<15.2f}")

    if section3_results:
        print("\n--- Section 3: Progressive Networks ---")
        print(f"{'Transfer':<35} {'Episodes':<12} {'Time (s)':<12} {'Avg Reward':<15}")
        print("-" * 75)
        for transfer, stats in section3_results.items():
            print(f"{transfer:<35} {stats['total_episodes']:<12} {stats['training_time']:<12.2f} {stats['final_avg_reward']:<15.2f}")

    print("\n" + "=" * 90)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_results(results: dict, filename: str):
    """Save results to JSON file."""
    report_dir = os.path.join(os.path.dirname(__file__), "report")
    os.makedirs(report_dir, exist_ok=True)

    filepath = os.path.join(report_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"Results saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="DRL Assignment 3: Meta and Transfer Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train.py --section all       Run all sections sequentially
    python train.py --section 1         Train individual networks only
    python train.py --section 2         Run fine-tuning experiments
    python train.py --section 3         Run progressive networks experiments
    python train.py --episodes 1000     Set max episodes
    python train.py --lr 0.001          Set learning rate

TensorBoard visualization:
    tensorboard --logdir logs/
        """
    )

    parser.add_argument('--section', type=str, default='all',
                        choices=['all', '1', '2', '3'],
                        help='Which section to run (default: all)')
    parser.add_argument('--episodes', type=int, default=2000,
                        help='Maximum number of episodes (default: 2000)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--hidden', type=int, default=256,
                        help='Hidden layer dimension (default: 256, matches pre-trained models)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')

    args = parser.parse_args()

    # Configuration
    config = TrainingConfig(
        max_episodes=args.episodes,
        lr_actor=args.lr,
        lr_critic=args.lr,
        hidden_dim=args.hidden,
        seed=args.seed,
        gamma=args.gamma,
    )

    print("=" * 70)
    print("DRL Assignment 3: Meta and Transfer Learning")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Section: {args.section}")
    print(f"Max Episodes: {config.max_episodes}")
    print(f"Learning Rate: {config.lr_actor}")
    print(f"Hidden Dim: {config.hidden_dim}")
    print(f"Seed: {config.seed}")
    print("=" * 70)

    all_results = {
        'config': {
            'max_episodes': config.max_episodes,
            'lr': config.lr_actor,
            'hidden_dim': config.hidden_dim,
            'seed': config.seed,
            'gamma': config.gamma,
        },
        'timestamp': datetime.now().isoformat(),
    }

    section1_results = {}
    section2_results = {}
    section3_results = {}

    # Run requested sections
    if args.section in ['all', '1']:
        section1_results = run_section1(config)
        all_results['section1'] = section1_results

    if args.section in ['all', '2']:
        if args.section == '2' and not section1_results:
            print("\nNote: Section 2 requires trained models from Section 1.")
            print("Make sure to run Section 1 first or have saved models available.\n")
        section2_results = run_section2(config)
        all_results['section2'] = section2_results

    if args.section in ['all', '3']:
        if args.section == '3' and not section1_results:
            print("\nNote: Section 3 requires trained models from Section 1.")
            print("Make sure to run Section 1 first or have saved models available.\n")
        section3_results = run_section3(config)
        all_results['section3'] = section3_results

    # Print comparison and save results
    if section1_results:
        print_comparison_table(section1_results, section2_results, section3_results)
        save_results(all_results, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    print("\nTraining complete!")
    print("View TensorBoard logs with: tensorboard --logdir logs/")


if __name__ == "__main__":
    main()
