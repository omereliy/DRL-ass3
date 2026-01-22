# DRL Assignment 3 - Submission Package

## Team Members
- **Omer Eliyahu** (206510828)
- **[STUDENT_NAME]** ([STUDENT_ID])

> **Note**: Replace `[STUDENT_NAME]` and `[STUDENT_ID]` in both this file and `DRL_Assignment3_Report.md` before submission.

---

## Submission Contents

```
submission/
├── DRL_Assignment3_Report.pdf     # Main report (required)
├── DRL_Assignment3_Report.md      # Report source (markdown)
├── learning_curves.png            # Training visualizations
├── SUBMISSION_README.md           # This file
├── src/                           # Solution scripts
│   ├── actor_critic.py            # Actor-Critic implementation
│   ├── environments.py            # Standardized environment wrappers
│   ├── fine_tuning.py             # Section 2: Fine-tuning
│   ├── progressive_networks.py    # Section 3: Progressive Networks
│   └── utils.py                   # Utilities and configuration
├── train.py                       # Main training script
├── requirements.txt               # Python dependencies
└── models/                        # Trained model checkpoints
    ├── actor_critic_CartPole-v1.pt
    ├── actor_critic_Acrobot-v1.pt
    ├── actor_critic_MountainCarContinuous-v0.pt
    ├── actor_critic_finetuned_*.pt
    └── actor_critic_progressive_*.pt
```

---

## Running Instructions

### 1. Setup Environment

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Training

```bash
# Run all sections (takes several hours)
python train.py --section all

# Run individual sections
python train.py --section 1    # Section 1: Individual training
python train.py --section 2    # Section 2: Fine-tuning
python train.py --section 3    # Section 3: Progressive Networks

# Custom hyperparameters
python train.py --section 1 --episodes 2000 --lr 0.003 --hidden 256
```

### 3. View Training Progress

```bash
# Start TensorBoard
tensorboard --logdir logs/

# Open browser to http://localhost:6006
```

### 4. Generated Outputs

- **Models**: Saved to `models/` directory
- **Logs**: TensorBoard logs in `logs/` directory
- **Results**: JSON summary in `report/results.json`
- **Figures**: Learning curves in `report/learning_curves.png`

---

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Gymnasium
- TensorBoard
- NumPy, Matplotlib

See `requirements.txt` for complete list.

---

## Assignment Checklist

- [x] Section 1: Individual Actor-Critic training (3 environments)
- [x] Section 2: Fine-tuning experiments (2 transfer pairs)
- [x] Section 3: Progressive Networks implementation (documented debugging)
- [x] Standardized input/output dimensions across environments
- [x] Training statistics and convergence analysis
- [x] TensorBoard visualizations
- [x] Report with explanations and analysis (max 6 pages)
