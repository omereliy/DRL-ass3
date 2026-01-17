# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep Reinforcement Learning Assignment 3: Meta and Transfer Learning using Actor-Critic. Implements:
- Section 1: Individual actor-critic training for CartPole-v1, Acrobot-v1, MountainCarContinuous-v0
- Section 2: Fine-tuning pre-trained models (Acrobot→CartPole, CartPole→MountainCar)
- Section 3: Progressive Networks transfer learning from multiple sources

## Commands

### Run Training
```bash
# Run all sections
python train.py --section all

# Run individual sections
python train.py --section 1    # Train individual networks
python train.py --section 2    # Fine-tuning experiments
python train.py --section 3    # Progressive networks

# With custom parameters
python train.py --section 1 --episodes 1000 --lr 0.003 --hidden 256
```

### TensorBoard
```bash
tensorboard --logdir logs/
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Architecture

### Standardized Dimensions
All environments use standardized input/output dimensions for transfer learning:
- `STANDARDIZED_OBS_DIM = 6` (max observation dimension, from Acrobot)
- `STANDARDIZED_ACT_DIM = 3` (max action dimension, from Acrobot)
- Smaller environments are zero-padded; actions are masked to valid indices

### Core Components

**`src/utils.py`**: Configuration (`TrainingConfig`), statistics tracking (`TrainingStats`), device setup, TensorBoard logging, and helper functions.

**`src/environments.py`**: `StandardizedEnv` wrapper that pads observations and handles action conversion. MountainCarContinuous is discretized to 3 actions.

**`src/actor_critic.py`**: `ActorCritic` model with separate Actor/Critic networks (2 hidden layers each). `ActorCriticTrainer` handles training with advantage normalization.

**`src/fine_tuning.py`**: `FineTuningTrainer` loads pre-trained source model, re-initializes output layers, trains on target environment.

**`src/progressive_networks.py`**: `ProgressiveActorCritic` with frozen source networks. Lateral connections combine source hidden features with target network.

### Model Flow
1. Individual training creates models in `models/actor_critic_{env_name}.pt`
2. Fine-tuning loads source model, reinitializes output layers
3. Progressive networks load multiple frozen sources, add lateral connections to trainable target

## Key Files

- `train.py` - Main entry point with CLI
- `src/actor_critic.py` - Core Actor-Critic implementation
- `src/fine_tuning.py` - Section 2 transfer learning
- `src/progressive_networks.py` - Section 3 progressive networks
- `DRL_Assignment3_Colab.ipynb` - Google Colab notebook for GPU training
