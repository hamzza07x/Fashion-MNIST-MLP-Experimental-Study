# Deep Learning for Perception — Assignment 1

## Fashion-MNIST MLP Experimental Study

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An empirical study of **Multi-Layer Perceptrons (MLPs)** using Fashion-MNIST, covering manual backpropagation, activation functions, loss functions, optimizers, overfitting, regularization, and hyperparameter tuning.

---

## Project Information

|                     | Details                                                                          |
| ------------------- | -------------------------------------------------------------------------------- |
| **Course**          | Deep Learning for Perception (BCS-7E)                                            |
| **Assignment**      | Assignment 1 — MLP Mechanics, Optimizers, Regularization & Hyperparameter Search |
| **Submission Date** | September 13, 2026                                                               |
| **Framework**       | PyTorch 2.0+                                                                     |
| **Python**          | 3.10+                                                                            |
| **Compute**         | NVIDIA Tesla T4 GPU                                                              |
| **Random Seed**     | 42                                                                               |

### Contributors

| Contributor  | Student ID | Section | Contribution                                                                  |
| ------------ | ---------- | ------- | ----------------------------------------------------------------------------- |
| **M. Hamza** | 23P-XXX    | BCS-7E  | Architecture Engineering, NumPy Autograd, Regularization & Cross-Validation   |
| **M. Talha** | 23F-0562   | BCS-7E  | Activation Dynamics, Optimizer Benchmarks, Hyperparameter Tuning & Evaluation |

---

## Table of Contents

* [Executive Summary](#executive-summary)
* [Dataset & Partitioning](#dataset--partitioning)
* [Experimental Results](#experimental-results)

  * [Part 1 — NumPy MLP & Gradient Verification](#part-1--numpy-mlp--gradient-verification)
  * [Part 2 — Activation Functions](#part-2--activation-functions)
  * [Part 3 — Loss Functions & Regression](#part-3--loss-functions--regression)
  * [Part 4 — Optimizers & Learning Rate Sensitivity](#part-4--optimizers--learning-rate-sensitivity)
  * [Part 5 — Controlled Overfitting](#part-5--controlled-overfitting)
  * [Part 6 — Regularization Ablation](#part-6--regularization-ablation)
  * [Part 7 — Hyperparameter Tuning](#part-7--hyperparameter-tuning)
* [Repository Structure](#repository-structure)
* [Installation](#installation)
* [Reproduction Guide](#reproduction-guide)
* [Citation](#citation)

---

# Executive Summary

This repository contains the implementation, experiment scripts, and results for an empirical study of **Multi-Layer Perceptrons (MLPs)** on the **Fashion-MNIST** classification benchmark and the **Diabetes** tabular regression dataset.

### Key Results

* **Gradient Verification:** NumPy backpropagation matched PyTorch autograd with a maximum absolute difference of **5.04 × 10⁻⁸**.
* **Best Activation:** Leaky ReLU achieved **89.48%** validation accuracy, slightly outperforming ReLU at **89.25%**.
* **Dead ReLU Units:** **8.85%** of evaluated ReLU hidden units remained inactive across the entire validation set.
* **Vanishing Gradients:** Sigmoid produced a mean first-layer gradient magnitude of **6.40 × 10⁻⁵**, compared with **7.28 × 10⁻⁴** for ReLU.
* **Loss Comparison:** Cross-Entropy and MSE achieved **89.25%** and **89.27%** validation accuracy, respectively.
* **Best Optimizer:** Adam reached **85%+ validation accuracy in 2 epochs** and achieved **89.25%** validation accuracy.
* **Overfitting:** A 5-layer MLP trained on only 2,000 samples reached **99.10% training accuracy** but only **82.05% validation accuracy**, producing a **17.05 percentage-point generalization gap**.
* **Regularization:** Dropout with `p = 0.5` reduced the generalization gap to **0.65 pp**, while L1 regularization reduced it to **1.69 pp**.
* **Best Regularization Accuracy:** Increasing the training set to 20,000 samples achieved **86.82% validation accuracy**.
* **Best Tuned Model:** 5-fold cross-validation selected:

  * Hidden widths: `(256, 256)`
  * Learning rate: `0.001`
  * Dropout: `0.2`
  * Weight decay: `0.0`
* **Cross-Validation:** **87.86% ± 0.25%**
* **Final Test Accuracy:** **89.26%**
* **Macro Precision:** **0.8922**
* **Macro Recall:** **0.8926**
* **Macro F1:** **0.8919**

---

# Dataset & Partitioning

The primary dataset is **Fashion-MNIST**, containing 70,000 grayscale images of size `28 × 28`.

Each image is flattened into a **784-dimensional vector** and normalized to `[0, 1]`.

To prevent test-set leakage, the dataset was partitioned as follows:

| Split      |    Samples |     Percentage |
| ---------- | ---------: | -------------: |
| Training   |     48,000 | 68.6% of total |
| Validation |     12,000 | 17.1% of total |
| Test       |     10,000 | 14.3% of total |
| **Total**  | **70,000** |       **100%** |

The original 60,000-image development set was split using stratification:

* **4,800 training samples per class**
* **1,200 validation samples per class**
* **1,000 test samples per class**

The final 10,000-image test set remains untouched until final evaluation.

---

# Experimental Results

## Part 1 — NumPy MLP & Gradient Verification

A two-layer MLP was implemented from scratch using NumPy:

```text
784 → 64 → 10
```

### Implementation

The model uses:

* Vectorized NumPy operations
* He/Kaiming weight initialization
* ReLU activation
* Numerically stable Softmax
* Cross-Entropy loss
* Manual forward and backward propagation
* Mini-batch gradient descent

### Training

Training loss decreased from:

```text
Epoch 1  →  1.2994
Epoch 10 →  0.4919
```

### Gradient Verification

The analytical NumPy gradients were compared against PyTorch's automatic differentiation using identical tensors.

**Maximum absolute difference:**

```text
5.04 × 10⁻⁸
```

This confirms that the manually derived gradients match PyTorch autograd to floating-point precision.

---

## Part 2 — Activation Functions

Four activation functions were evaluated using the architecture:

```text
784 → 128 → 64 → 10
```

with:

* Adam optimizer
* Learning rate: `0.001`
* Batch size: `256`

### Validation Accuracy

| Activation     | Validation Accuracy |
| -------------- | ------------------: |
| **Leaky ReLU** |          **89.48%** |
| ReLU           |              89.25% |
| Tanh           |              88.47% |
| Sigmoid        |              88.28% |

### Gradient Flow

The first-layer mean absolute gradient magnitudes were:

| Activation | Mean Absolute Gradient |
| ---------- | ---------------------: |
| Sigmoid    |          `6.40 × 10⁻⁵` |
| ReLU       |          `7.28 × 10⁻⁴` |

The substantially smaller gradient magnitude for Sigmoid demonstrates its susceptibility to **vanishing gradients**, particularly in saturated activation regions.

### Dead ReLU Analysis

Approximately **8.85%** of evaluated ReLU hidden units produced zero output for every validation sample.

This demonstrates the potential **dying ReLU** problem and explains why Leaky ReLU can provide more robust gradient flow.

---

## Part 3 — Loss Functions & Regression

### Fashion-MNIST Classification

Cross-Entropy and Mean Squared Error were compared under the same training conditions.

| Loss Function | Validation Accuracy |
| ------------- | ------------------: |
| Cross-Entropy |              89.25% |
| MSE           |              89.27% |

Although MSE performed similarly in this experiment, Cross-Entropy remains the more appropriate standard loss for multi-class classification because its gradient with Softmax simplifies to:

```text
∇z L = p - y
```

This provides a more direct error signal than applying MSE through the Softmax derivative.

### Diabetes Regression

A separate MLP was trained on the Diabetes tabular regression dataset.

Architecture:

```text
10 → 32 → 16 → 1
```

Training used MSE loss for 100 epochs.

| Metric   |      Result |
| -------- | ----------: |
| **MSE**  | **3390.37** |
| **RMSE** |   **58.23** |
| **MAE**  |   **49.30** |

---

## Part 4 — Optimizers & Learning Rate Sensitivity

Four optimization strategies were evaluated on the baseline architecture:

```text
784 → 128 → 64 → 10
```

### Optimizer Comparison

| Optimizer | Accuracy @ 0.001 | Epochs to ≥85% | Tuned LR | Tuned Final Accuracy |
| --------- | ---------------: | -------------: | -------: | -------------------: |
| **Adam**  |       **89.25%** |          **2** |    0.001 |               87.04% |
| RMSprop   |           88.53% |              4 |    0.001 |               87.29% |
| Momentum  |           81.03% |         Failed |    0.010 |           **88.07%** |
| SGD       |           49.89% |         Failed |    0.050 |           **85.68%** |

Adam demonstrated the fastest convergence using the common learning rate of `0.001`.

Learning-rate tuning significantly improved both SGD and Momentum.

---

## Part 5 — Controlled Overfitting

To deliberately induce high variance, an overparameterized five-layer MLP was trained using only **2,000 training samples**.

Architecture:

```text
784 → 512 → 512 → 512 → 512 → 10
```

The model was trained without regularization for 53 epochs.

### Results

| Metric              |       Result |
| ------------------- | -----------: |
| Training Accuracy   |   **99.10%** |
| Validation Accuracy |   **82.05%** |
| Generalization Gap  | **17.05 pp** |

The divergence became noticeable around **Epoch 45**.

The large gap demonstrates strong memorization and high model variance caused by the combination of limited training data and excessive model capacity.

---

# Part 6 — Regularization Ablation

Regularization techniques were evaluated using the 2,000-sample training regime.

The experiments were limited to 20 epochs with early stopping patience of 3.

| Method              | Setting      | Train Acc. |  Val. Acc. |         Gap |
| ------------------- | ------------ | ---------: | ---------: | ----------: |
| Baseline            | None         |     88.55% |     81.59% |     6.96 pp |
| L2                  | λ = 10⁻⁴     |     85.85% |     79.62% |     6.23 pp |
| L2                  | λ = 10⁻³     |     86.65% |     81.15% |     5.50 pp |
| L2                  | λ = 10⁻²     |     75.90% |     72.49% |     3.41 pp |
| **L1**              | **λ = 10⁻⁵** | **82.05%** | **80.36%** | **1.69 pp** |
| Dropout             | p = 0.2      |     87.75% |     81.33% |     6.42 pp |
| **Dropout**         | **p = 0.5**  | **81.35%** | **80.70%** | **0.65 pp** |
| Dropout             | p = 0.7      |     63.70% |     69.20% |    -5.50 pp |
| Batch Normalization | BatchNorm1d  |     96.80% |     81.05% |    15.75 pp |
| Early Stopping      | Patience = 3 |     88.55% |     81.59% |     6.96 pp |
| Data Augmentation   | Crop + Flip  |     81.25% |     80.04% |     1.21 pp |
| Data Scaling        | N = 10,000   |     87.61% |     83.97% |     3.63 pp |
| Data Scaling        | N = 20,000   |     91.81% | **86.82%** |     4.99 pp |

### Key Findings

* **Dropout `p = 0.5`** produced the smallest positive generalization gap at **0.65 pp**.
* **L1 regularization** achieved a strong balance between training and validation performance with a **1.69 pp gap**.
* L1 produced approximately **28.02% parameter sparsity**.
* **Dropout `p = 0.7`** over-regularized the model and caused underfitting.
* Batch normalization accelerated convergence but still produced a large **15.75 pp** generalization gap.
* Increasing the training set to **20,000 samples** produced the highest validation accuracy in this ablation study at **86.82%**.

---

# Part 7 — Hyperparameter Tuning

A random search was performed over **12 configurations** using 5-fold stratified cross-validation.

The search explored:

* Learning rates from `10⁻⁴` to `10⁻²`
* Hidden-layer configurations:

  * `(128, 64)`
  * `(256, 256)`
  * `(512, 256)`
* Dropout rates from `0.0` to `0.5`
* Weight decay from `0.0` to `10⁻³`

### Best Configuration

```text
Learning Rate : 0.001
Hidden Widths: (256, 256)
Dropout      : 0.2
Weight Decay : 0.0
```

### Cross-Validation

```text
87.86% ± 0.25%
```

The selected architecture was then retrained using the complete 48,000-sample training partition and evaluated once on the untouched 10,000-sample test set.

### Final Test Results

| Metric              |     Result |
| ------------------- | ---------: |
| **Accuracy**        | **89.26%** |
| **Macro Precision** | **0.8922** |
| **Macro Recall**    | **0.8926** |
| **Macro F1**        | **0.8919** |

The final tuned model improved upon the Part 2 baseline by **0.97 percentage points**.

---

# Repository Structure

```text
.
├── README.md
├── requirements.txt
├── environment.yml
│
├── data/
│   ├── fashion-mnist_train.csv
│   └── fashion-mnist_test.csv
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── numpy_mlp.py
│   ├── models.py
│   ├── trainer.py
│   └── utils.py
│
├── experiments/
│   ├── run_part1_grad_check.py
│   ├── run_part2_activations.py
│   ├── run_part3_loss_regression.py
│   ├── run_part4_optimizers.py
│   ├── run_part5_overfitting_gap.py
│   ├── run_part6_regularization.py
│   └── run_part7_kfold_tuning.py
│
└── notebooks/
    └── DLP_Assignment01.ipynb
```

### Source Modules

| File             | Purpose                                                     |
| ---------------- | ----------------------------------------------------------- |
| `config.py`      | Global configuration, random seed, and device selection     |
| `data_loader.py` | Dataset loading and stratified train/validation splitting   |
| `numpy_mlp.py`   | NumPy MLP and analytical gradient verification              |
| `models.py`      | PyTorch MLP architectures                                   |
| `trainer.py`     | Training, validation, L1 regularization, and early stopping |
| `utils.py`       | Metrics, dead-neuron analysis, and visualization utilities  |

---

# Installation

## Option 1 — Conda

```bash
git clone https://github.com/<your-username>/dlp-fashion-mnist-mlp.git
cd dlp-fashion-mnist-mlp

conda env create -f environment.yml
conda activate dlp-assignment1
```

## Option 2 — Python Virtual Environment

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Windows

```powershell
python -m venv venv
venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dependencies

```text
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0,<2.0.0
pandas>=2.0.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
scipy>=1.10.0
tqdm>=4.65.0
```

---

# Reproduction Guide

All experiment scripts should be executed from the repository root.

## 1. Gradient Verification

```bash
python experiments/run_part1_grad_check.py
```

Expected:

```text
Epoch 1 loss  ≈ 1.2994
Epoch 10 loss ≈ 0.4919
Max gradient difference ≤ 5.04 × 10⁻⁸
```

## 2. Activation Analysis

```bash
python experiments/run_part2_activations.py
```

Expected:

```text
Leaky ReLU ≈ 89.48%
ReLU       ≈ 89.25%
Dead ReLU  ≈ 8.85%
```

## 3. Loss & Regression Experiments

```bash
python experiments/run_part3_loss_regression.py
```

Expected:

```text
Cross-Entropy ≈ 89.25%
MSE           ≈ 89.27%

Diabetes:
MSE  = 3390.37
RMSE = 58.23
MAE  = 49.30
```

## 4. Optimizer Experiments

```bash
python experiments/run_part4_optimizers.py
```

Expected:

```text
Adam      → 89.25%
RMSprop   → 88.53%
Momentum  → 88.07% (tuned)
SGD       → 85.68% (tuned)
```

## 5. Overfitting Experiment

```bash
python experiments/run_part5_overfitting_gap.py
```

Expected:

```text
Training Accuracy   ≈ 99.10%
Validation Accuracy ≈ 82.05%
Generalization Gap  ≈ 17.05 pp
```

## 6. Regularization Experiments

```bash
python experiments/run_part6_regularization.py
```

Expected key results:

```text
Dropout p=0.5 → 0.65 pp gap
L1 λ=1e-5     → 1.69 pp gap
```

## 7. Hyperparameter Search

```bash
python experiments/run_part7_kfold_tuning.py
```

Expected:

```text
Best Architecture: (256, 256)
Learning Rate:     0.001
Dropout:            0.2
CV Accuracy:        87.86% ± 0.25%
Test Accuracy:      89.26%
```

---

# Citation

If you use this repository in academic work, please cite:

```bibtex
@misc{hamza_talha_dlp_assignment1_2026,
  author       = {M. Hamza and M. Talha},
  title        = {Deep Learning for Perception Assignment 1:
                  Fashion-MNIST MLP Experimental Study},
  year         = {2026},
  month        = {September},
  institution  = {FAST-NUCES},
  note         = {Course: Deep Learning for Perception, Section BCS-7E}
}
```

---

## License

This project is released under the **MIT License**.
