# Assignment 3: Meta and Transfer Learning

**Institution:** Ben-Gurion University of the Negev, Faculty of Engineering Sciences, Department of Software and Information Systems  
**Course:** Deep Reinforcement Learning  
**Date Published:** 24/12/2025  
**Due Date:** 16/01/2024

---

## General Instructions

* **Submission Format:** The assignment should be submitted in pairs via the Moodle course site.

* **Zip File Contents:** The submission must include a zip file containing:
  * A report in PDF format containing answers to questions and requested code outputs.
  * Detailed explanations and analysis.
  * Short instructions for running the scripts.
  * The scripts of your solutions.

* **Technical Requirements:**
  * Scripts must be written in Python using the PyTorch library for Neural Networks.
  * Use TensorBoard for visualization and graphs.

* **Report Guidelines:**
  * The report can be written in English or Hebrew.
  * The length must not exceed six pages.
  * Include your names and IDs in the report.

---

## Introduction

While humans and animals can learn new tasks in just a few trials, deep reinforcement learning algorithms usually require a large number of trials. Standard tools require re-collecting large datasets and training from scratch for new tasks. Intuitively, knowledge from one task should facilitate learning related tasks more quickly.

In this assignment, you will design a reinforcement learning algorithm that leverages prior experience to solve new tasks quickly, a method referred to in literature as meta-reinforcement learning.

---

## Section 1 – Training Individual Networks (25%)

In this section, you will implement the actor-critic architecture (from HW2) for two additional small control problems: **Acrobot-v1** and **MountainCarContinuous-v0**.

**Goals:**

* Achieve the respective goals: reaching the mountain top and bringing the acrobot to a pre-specified height.

* **Standardization for Transfer Learning:** The size of the input and output for *all* tasks must be identical.
  * For problems with smaller inputs, pad with 0.
  * For problems with smaller outputs, create "empty" actions that are never used.

**Requirements:**

* You must retrain the architecture for the **cartpole** problem.
* You may use different architectures for each problem, but each must include at least one hidden layer (Input -> Hidden -> Output).
* Provide statistics for running time and the number of training iterations required for convergence.

---

## Section 2 – Fine-tune an Existing Model (25%)

You will fine-tune a model trained on a source problem and apply it to a target problem.

**Tasks:**
Apply the following to two pairs (Source -> Target):

1. **Acrobot -> Cartpole**
2. **Cartpole -> MountainCar**

**Procedure:**

* Take the model fully trained on the **source**.
* Re-initialize the weights of the output layer.
* Train the new network on the **target**.

**Analysis:**

* Provide statistics on running time and training iterations.
* Compare results to Section 1. Did fine-tuning lead to faster convergence?

---

## Section 3 – Transfer Learning (50%)

In this section, you will implement a simplified version of **Progressive Networks**.

**Tasks:**
Apply the following settings (Sources -> Target):

1. **{Acrobot, MountainCar} -> Cartpole**
2. **{Cartpole, Acrobot} -> MountainCar**

**Procedure:**

* Use the fully-trained **source** networks created in Section 1 and connect them to the un-trained **target** network.

* **Frozen Sources:** The source networks remain frozen throughout the process.

* **Adapters:** Implementing adapters (marked with 'a' in diagrams) is optional.

**Connections:**

* **Single Hidden Layer:** Connect the hidden layers of the sources to the output of the target network.

* **Multiple Hidden Layers:** Connect the top hidden layer in the source to the target output, then connect layer to until you run out of hidden layers in one of the architectures.

**Analysis:**

* Train until convergence. Did transfer learning improve training?
* Provide statistics on running time and training iterations.

**Important Note:** Transfer learning is tricky. If you do not succeed in showing improvement, you must document your efforts and explain how you attempted to get the architectures to work.

---

## Project Structure

```
DRL-ass3/
├── README.md
├── src/
│   ├── actor_critic.py
│   ├── environments.py
│   ├── fine_tuning.py
│   ├── progressive_networks.py
│   └── utils.py
├── models/
├── logs/
└── report/
```
