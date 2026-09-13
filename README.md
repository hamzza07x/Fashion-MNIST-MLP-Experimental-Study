# Deep Learning for Perception — Assignment 1: Fashion-MNIST MLP Experimental Study[cite: 3]

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

### Project & Contributor Metadata[cite: 3]
* **Course:** Deep Learning for Perception (BCS-7E)[cite: 3]
* **Assignment:** Assignment No. 1 — Multi-Layer Perceptron Mechanics, Optimizers, Regularization & Hyperparameter Search[cite: 3]
* **Submission Date:** September 13, 2026[cite: 3]
* **Compute Environment:** NVIDIA Tesla T4 GPU, Fixed Random Seed = 42[cite: 1, 3]

| Contributor Name | Student Roll / Registration ID | Section | Contribution Role |
| :--- | :--- | :--- | :--- |
| **M. Hamza**[cite: 3] | 23P-[cite: 3] | BCS-7E[cite: 3] | Architecture Engineering, NumPy Autograd, Regularization & Cross-Validation[cite: 3] |
| **M. Talha**[cite: 3] | 23F-0562[cite: 3] | BCS-7E[cite: 3] | Activation Dynamics, Optimizer Benchmarks, Hyperparameter Tuning & Evaluation[cite: 3] |

---

## Table of Contents
1. [Executive Summary & Key Report Results](#executive-summary--key-report-results)
2. [Dataset Pipeline & Partitioning](#dataset-pipeline--partitioning)
3. [Experimental Suite & Empirical Findings](#experimental-suite--empirical-findings)
   - [Part 1: NumPy MLP & Gradient Verification](#part-1-numpy-mlp-and-gradient-verification)
   - [Part 2: Activation Functions & Gradient Flow Analysis](#part-2-activation-functions-and-gradient-flow-analysis)
   - [Part 3: Loss Function Comparison & Tabular Regression](#part-3-loss-function-comparison-and-tabular-regression)
   - [Part 4: Optimizer Trajectory & Learning Rate Sensitivity](#part-4-optimizer-trajectory-and-learning-rate-sensitivity)
   - [Part 5: Controlled Overfitting & Variance Analysis](#part-5-controlled-overfitting-and-variance-analysis)
   - [Part 6: Regularization Ablation Study](#part-6-regularization-ablation-study)
   - [Part 7: Hyperparameter Tuning via 5-Fold Cross-Validation](#part-7-hyperparameter-tuning-via-5-fold-cross-validation)
4. [Repository Architecture](#repository-architecture)
5. [Environment Setup & Installation](#environment-setup--installation)
6. [Step-by-Step Reproduction Guide](#step-by-step-reproduction-guide)
7. [Citation](#citation)

---

## Executive Summary & Key Report Results

This repository contains the official codebase, experiment logs, and reproduction scripts for our empirical evaluation of Multilayer Perceptrons (MLPs) on the **Fashion-MNIST** benchmark and the **Diabetes** tabular regression dataset[cite: 3].

### Summary of Empirical Metrics[cite: 3]
* **Analytical Gradient Equivalence:** Manual backpropagation in pure NumPy matched PyTorch `autograd` tensor gradients with a maximum absolute discrepancy of **$5.04 \times 10^{-8}$**[cite: 3].
* **Activation Performance & Dead Neurons:** Leaky ReLU attained the highest validation accuracy of **$89.48\%$**, outperforming standard ReLU (**$89.25\%$**)[cite: 3]. Standard ReLU exhibited an **$8.85\%$ permanent dead neuron rate** across validation evaluation[cite: 3]. Sigmoid displayed severe gradient vanishing ($6.40 \times 10^{-5}$ mean absolute gradient versus $7.28 \times 10^{-4}$ for ReLU)[cite: 3].
* **Loss Function Comparison:** Cross-entropy ($89.25\%$) and Mean Squared Error ($89.27\%$) achieved nearly identical classification validation accuracy[cite: 3]. A continuous tabular regression baseline on the Diabetes dataset yielded **MSE = 3390.37, RMSE = 58.23, and MAE = 49.30**[cite: 3].
* **Optimization Speed:** At $\eta = 0.001$, Adam was the fastest optimizer, achieving $\ge 85\%$ validation accuracy within **2 epochs** and peaking at **$89.25\%$**[cite: 3]. Tuned SGD ($\eta = 0.05$) improved from an initial $49.89\%$ up to $85.68\%$, while tuned Momentum ($\eta = 0.01$) improved from $81.03\%$ to $88.07\%$[cite: 3].
* **Generalization Gap Induction:** By training a deep 5-layer MLP ($784 \to 512 \times 4 \to 10$) on an intentionally restricted subset of $N=2,000$ samples, the model achieved **$99.10\%$** training accuracy against **$82.05\%$** validation accuracy—producing a massive **$17.05$ percentage-point generalization gap** indicative of high variance[cite: 3].
* **Regularization Impact:** Evaluated on the $N=2,000$ sample regime, Dropout ($p=0.5$) and Data Augmentation reduced the baseline generalization gap of $6.96$ pp down to **$0.65$ pp** and **$1.21$ pp**, respectively[cite: 3]. $L_1$ regularization ($\lambda = 10^{-5}$) provided the most balanced trade-off, lowering the gap to **$1.69$ pp** with only a modest decrease in training accuracy[cite: 3].
* **Optimal Model & Test Performance:** 5-fold cross-validation over 12 configurations selected an optimal architecture of `width = (256, 256)`, `lr = 0.001`, and `dropout = 0.2` (scoring a CV mean of **$87.86\% \pm 0.25\%$**)[cite: 3]. When retrained on the full 48,000-sample training partition and evaluated on the untouched 10,000-sample test vault, the model achieved **$89.26\%$ Test Accuracy**, **$0.8922$ Macro Precision**, **$0.8926$ Macro Recall**, and **$0.8919$ Macro F1-score**, exceeding the Part 2 baseline by **$+0.97$ percentage points**[cite: 3].

---

## Dataset Pipeline & Partitioning

Fashion-MNIST inputs ($28 \times 28$ grayscale pixels flattened to 784 dimensions) are normalized to the range $[0, 1]$[cite: 3]. To prevent data leakage, the partition splits are strictly held as:

$$\text{Fashion-MNIST (70,000)} = \begin{cases} \mathcal{D}_{\text{train}}: & 48,000 \text{ samples (80\% of dev set)} \text{[cite: 1, 3]} \\ \mathcal{D}_{\text{val}}: & 12,000 \text{ samples (20\% of dev set)} \text{[cite: 1, 3]} \\ \mathcal{D}_{\text{test}}: & 10,000 \text{ samples (held-out evaluation vault)} \text{[cite: 1, 3]} \end{cases}$$

Class balance across all 10 target categories is preserved with exact stratification[cite: 1]:
* 4,800 training samples per class[cite: 1]
* 1,200 validation samples per class[cite: 1]
* 1,000 test samples per class[cite: 1]

---

## Experimental Suite & Empirical Findings

### Part 1: NumPy MLP and Gradient Verification[cite: 3]
* **Implementation:** A 2-layer MLP ($784 \to 64 \to 10$) written from scratch in NumPy using vectorized matrix operations, He (Kaiming) weight initialization, ReLU hidden activations, numerically stable Softmax, and Cross-Entropy loss[cite: 1, 3].
* **Training Dynamics:** Mini-batch gradient descent reduced training loss from **1.2994** in Epoch 1 down to **0.4919** in Epoch 10[cite: 3].
* **Gradient Sanity Check:** NumPy analytical Jacobians were compared against PyTorch automatic differentiation on identical tensors[cite: 3]:
  * Maximum absolute discrepancy across parameter tensors: **$5.04 \times 10^{-8}$**[cite: 3]
  * Validates exact mathematical derivation to floating-point precision[cite: 3].

### Part 2: Activation Functions and Gradient Flow Analysis[cite: 3]
* **Evaluated Functions:** Sigmoid, Tanh, ReLU, and Leaky ReLU ($\alpha = 0.01$) on a $784 \to 128 \to 64 \to 10$ architecture using Adam ($\eta = 0.001$, batch size 256)[cite: 1, 3].
* **Vanishing Gradient Verification:**
  * First-layer mean absolute gradient norm for Sigmoid: **$6.40 \times 10^{-5}$**[cite: 3]
  * First-layer mean absolute gradient norm for ReLU: **$7.28 \times 10^{-4}$**[cite: 3]
  * Sigmoid experienced an order-of-magnitude reduction in gradient magnitude due to saturation at high and low activation regimes[cite: 3].
* **Validation Accuracy & Dying ReLUs:**
  * Leaky ReLU: **$89.48\%$** (Best performer)[cite: 3]
  * ReLU: **$89.25\%$**[cite: 3]
  * Tanh: **$88.47\%$**[cite: 1]
  * Sigmoid: **$88.28\%$**[cite: 1]
  * **$8.85\%$** of evaluated ReLU hidden units output zero across every single sample in the validation dataset, demonstrating dead neuron vulnerability[cite: 3].

### Part 3: Loss Function Comparison and Tabular Regression[cite: 3]
* **Classification Loss Benchmark (Fashion-MNIST):**
  * Cross-Entropy: **$89.25\%$** validation accuracy[cite: 3]
  * Mean Squared Error: **$89.27\%$** validation accuracy[cite: 3]
  * While MSE performed on par under Adam on this balanced benchmark, Cross-Entropy provides linear error-proportional gradient flow ($\nabla_z \mathcal{L} = p - y$), avoiding the vanishing gradient traps caused by Softmax derivative terms in MSE[cite: 3].
* **Continuous Tabular Regression (Diabetes Dataset):**
  * Architecture: $10 \to 32 \to 16 \to 1$ MLP trained using MSE loss for 100 epochs[cite: 1].
  * **Mean Squared Error (MSE):** `3390.37`[cite: 3]
  * **Root Mean Squared Error (RMSE):** `58.23`[cite: 3]
  * **Mean Absolute Error (MAE):** `49.30`[cite: 3]

### Part 4: Optimizer Trajectory and Learning Rate Sensitivity[cite: 3]
Evaluated across 8 epochs on the baseline architecture ($784 \to 128 \to 64 \to 10$)[cite: 1]:

| Optimizer | Common $\eta = 0.001$ Acc | Epochs to $\ge 85\%$ (Common $\eta$) | Tuned $\eta$ | Tuned Final Val Acc | Epochs to $\ge 85\%$ (Tuned $\eta$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Adam**[cite: 3] | **$89.25\%$**[cite: 3] | **2**[cite: 3] | 0.001[cite: 1] | **$87.04\%$**[cite: 1] | **3**[cite: 1] |
| **RMSprop**[cite: 3] | $88.53\%$[cite: 3] | 4[cite: 3] | 0.001[cite: 1] | $87.29\%$[cite: 1] | 6[cite: 1] |
| **Momentum (0.9)**[cite: 1, 3] | $81.03\%$[cite: 3] | Failed[cite: 1] | 0.010[cite: 1] | $88.07\%$[cite: 3] | 7[cite: 1] |
| **SGD**[cite: 3] | $49.89\%$[cite: 3] | Failed[cite: 1] | 0.050[cite: 1, 3] | $85.68\%$[cite: 3] | >8[cite: 1] |

### Part 5: Controlled Overfitting and Variance Analysis[cite: 3]
* **Experimental Condition:** Subsampled dataset ($N=2,000$ training points) paired with an overparameterized 5-layer MLP ($784 \to 512 \to 512 \to 512 \to 512 \to 10$) trained without regularization for 53 epochs[cite: 1, 3].
* **Results:**
  * Training Accuracy: **$99.10\%$**[cite: 3]
  * Validation Accuracy: **$82.05\%$**[cite: 3]
  * Generalization Gap: **$17.05$ percentage points**[cite: 3]
* **Diagnostic Finding:** Divergence occurred around Epoch 45[cite: 1]. The massive generalization gap confirms extreme high variance and memorization of training instances[cite: 3].

### Part 6: Regularization Ablation Study[cite: 3]
Benchmarking 12 regularization runs on the $N=2,000$ training set (20 epochs max, early stopping patience = 3)[cite: 1, 3]:

| Regularization Method | Setting / Hyperparameter | Train Accuracy | Validation Accuracy | Generalization Gap | Status / Observation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Baseline**[cite: 3] | No regularization[cite: 1] | $88.55\%$[cite: 1] | $81.59\%$[cite: 1] | $6.96$ pp[cite: 3] | Reference model[cite: 1] |
| **$L_2$ Weight Decay**[cite: 1] | $\lambda = 10^{-4}$[cite: 1] | $85.85\%$[cite: 1] | $79.62\%$[cite: 1] | $6.23$ pp[cite: 1] | Stopped at epoch 16[cite: 1] |
| **$L_2$ Weight Decay**[cite: 1] | $\lambda = 10^{-3}$[cite: 1] | $86.65\%$[cite: 1] | $81.15\%$[cite: 1] | $5.50$ pp[cite: 1] | Stopped at epoch 18[cite: 1] |
| **$L_2$ Weight Decay**[cite: 3] | $\lambda = 10^{-2}$[cite: 3] | $75.90\%$[cite: 1] | $72.49\%$[cite: 1] | $3.41$ pp[cite: 1] | Over-penalized capacity; least helpful[cite: 3] |
| **$L_1$ Penalty (Lasso)**[cite: 1, 3] | $\lambda = 10^{-5}$[cite: 1, 3] | $82.05\%$[cite: 1] | $80.36\%$[cite: 1] | **$1.69$ pp**[cite: 3] | **Best trade-off**; induced $28.02\%$ parameter sparsity[cite: 1, 3] |
| **Dropout**[cite: 1] | $p = 0.2$[cite: 1] | $87.75\%$[cite: 1] | $81.33\%$[cite: 1] | $6.42$ pp[cite: 1] | Mild variance reduction[cite: 1] |
| **Dropout**[cite: 3] | $p = 0.5$[cite: 3] | $81.35\%$[cite: 1] | $80.70\%$[cite: 1] | **$0.65$ pp**[cite: 3] | **Lowest positive gap**; sacrificed some train accuracy[cite: 3] |
| **Dropout**[cite: 3] | $p = 0.7$[cite: 3] | $63.70\%$[cite: 1] | $69.20\%$[cite: 1] | **$-5.50$ pp**[cite: 1] | Over-regularized; severe signal destruction[cite: 3] |
| **Batch Normalization**[cite: 3] | BatchNorm1d per layer[cite: 1] | $96.80\%$[cite: 1] | $81.05\%$[cite: 1] | $15.75$ pp[cite: 1] | Accelerated convergence (stopped @ ep 7), but overfit[cite: 1, 3] |
| **Early Stopping**[cite: 1] | Patience = 3 epochs[cite: 1] | $88.55\%$[cite: 1] | $81.59\%$[cite: 1] | $6.96$ pp[cite: 1] | Prevented late-stage degradation[cite: 1] |
| **Data Augmentation**[cite: 3] | Random crop & flip[cite: 1] | $81.25\%$[cite: 1] | $80.04\%$[cite: 1] | **$1.21$ pp**[cite: 3] | Effective synthetic variance reduction[cite: 3] |
| **Data Scaling (Mid)**[cite: 1] | $N = 10,000$[cite: 1] | $87.61\%$[cite: 1] | $83.97\%$[cite: 1] | $3.63$ pp[cite: 1] | Clear variance reduction via data scaling[cite: 1] |
| **Data Scaling (High)**[cite: 1] | $N = 20,000$[cite: 1] | $91.81\%$[cite: 1] | **$86.82\%$**[cite: 1] | $4.99$ pp[cite: 1] | **Highest accuracy** among all regularization regimes[cite: 1] |

### Part 7: Hyperparameter Tuning via 5-Fold Cross-Validation[cite: 3]
* **Methodology:** Random search across 12 hyperparameter configurations spanning learning rates ($\eta \in [10^{-2}, 10^{-4}]$), layer architectures (widths: `(128, 64)`, `(256, 256)`, `(512, 256)`), dropout rates ($p \in [0.0, 0.5]$), and weight decay ($\lambda \in [0.0, 10^{-3}]$)[cite: 3].
* **Cross-Validation Result:** Evaluated on the 48,000 training partition using 5-fold stratified cross-validation[cite: 3].
  * **Top Configuration:** `learning_rate = 0.001`, `hidden_widths = (256, 256)`, `dropout = 0.2`, `weight_decay = 0.0`[cite: 3]
  * **5-Fold Cross-Validation Mean Accuracy:** **$87.86\% \pm 0.25\%$**[cite: 3]
* **Held-Out Test Set Evaluation:** The optimal model was retrained on all 48,000 training examples and evaluated once on the untouched 10,000-sample test vault[cite: 3]:
  * **Test Accuracy:** **$89.26\%$**[cite: 3]
  * **Macro Precision:** **$0.8922$**[cite: 3]
  * **Macro Recall:** **$0.8926$**[cite: 3]
  * **Macro F1-Score:** **$0.8919$**[cite: 3]
  * **Improvement:** Outperformed the Part 2 baseline by **$+0.97$ percentage points**[cite: 3].

---

## Repository Architecture

```text
.
├── README.md                           # Master project documentation
├── requirements.txt                    # Pinned Python package dependencies
├── environment.yml                     # Conda reproducible environment definition
├── data/                               # Local dataset folder (auto-download fallback)
│   ├── fashion-mnist_train.csv         # Train split (optional CSV path)[cite: 1]
│   └── fashion-mnist_test.csv          # Test split (optional CSV path)[cite: 1]
├── src/                                # Modular source code library
│   ├── __init__.py
│   ├── config.py                       # Global parameters, seeds (42), device selector[cite: 1, 3]
│   ├── data_loader.py                  # Stratified 80/20 train-val pipeline and test loader[cite: 1, 3]
│   ├── numpy_mlp.py                    # Hand-coded 2-layer NumPy MLP with autograd gradient check[cite: 1, 3]
│   ├── models.py                       # Modular PyTorch MLPClassifier supporting Dropout & BatchNorm[cite: 1]
│   ├── trainer.py                      # Training loop with validation tracking, L1 penalty & early stopping[cite: 1]
│   └── utils.py                        # Metrics calculator, dead unit analyzer, and plotting functions[cite: 1]
├── experiments/                        # Standalone execution scripts for assignment parts
│   ├── run_part1_grad_check.py         # NumPy backprop parity vs PyTorch autograd[cite: 1, 3]
│   ├── run_part2_activations.py        # Activation comparison & dead ReLU diagnostic[cite: 1, 3]
│   ├── run_part3_loss_regression.py    # CE vs MSE classification & Diabetes tabular regression[cite: 1, 3]
│   ├── run_part4_optimizers.py         # Optimizer convergence and learning rate sensitivity[cite: 1, 3]
│   ├── run_part5_overfitting_gap.py    # Sample reduction (N=2000) & generalization gap study[cite: 1, 3]
│   ├── run_part6_regularization.py     # 12-run regularization ablation suite[cite: 1, 3]
│   └── run_part7_kfold_tuning.py       # 5-fold CV hyperparameter search and test evaluation[cite: 3]
└── notebooks/
    └── DLP_Assignment01.ipynb          # Original Jupyter notebook with complete cell execution history[cite: 1]
```

---

## Environment Setup & Installation

### Option 1: Using Conda (Recommended)
```bash
git clone [https://github.com/](https://github.com/)<your-repo>/dlp-fashion-mnist-mlp.git
cd dlp-fashion-mnist-mlp

conda env create -f environment.yml
conda activate dlp-assignment1
```

### Option 2: Using standard Python venv
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Dependency Manifest (`requirements.txt`)
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

## Step-by-Step Reproduction Guide

Execute each experiment script from the project root to reproduce the numerical results from the report[cite: 3]:

### 1. Verify Analytical Gradient Parity
```bash
python experiments/run_part1_grad_check.py
```
*Expected Result:* NumPy loss drops from $1.2994$ to $0.4919$, and max absolute difference vs PyTorch autograd is $\le 5.04 \times 10^{-8}$[cite: 3].

### 2. Run Activation Dynamics & Dead Neuron Diagnostic
```bash
python experiments/run_part2_activations.py
```
*Expected Result:* Leaky ReLU reaches $89.48\%$ validation accuracy, ReLU reaches $89.25\%$, and dead ReLU unit percentage outputs $\approx 8.85\%$[cite: 3].

### 3. Compare Loss Functions & Run Diabetes Regression
```bash
python experiments/run_part3_loss_regression.py
```
*Expected Result:* CE val accuracy is $89.25\%$, MSE val accuracy is $89.27\%$, and Diabetes regression outputs MSE = $3390.37$, RMSE = $58.23$, MAE = $49.30$[cite: 3].

### 4. Evaluate Optimizer Convergence
```bash
python experiments/run_part4_optimizers.py
```
*Expected Result:* Adam reaches $\ge 85\%$ in 2 epochs (ending at $89.25\%$), RMSprop in 4 epochs ($88.53\%$), and tuned Momentum ($\eta=0.01$) reaches $88.07\%$[cite: 3].

### 5. Reproduce Generalization Gap (Overfitting Stress-Test)
```bash
python experiments/run_part5_overfitting_gap.py
```
*Expected Result:* Trains for 53 epochs on $N=2,000$ points, achieving $99.10\%$ train accuracy and $82.05\%$ val accuracy ($17.05$ pp generalization gap)[cite: 3].

### 6. Run Full Regularization Ablation
```bash
python experiments/run_part6_regularization.py
```
*Expected Result:* Re-evaluates all 12 regularization configurations; confirms Dropout ($p=0.5$) gap at $0.65$ pp and $L_1$ gap at $1.69$ pp[cite: 3].

### 7. Run 5-Fold Cross-Validation & Held-Out Test Evaluation
```bash
python experiments/run_part7_kfold_tuning.py
```
*Expected Result:* 5-fold CV selects `(256, 256)` width with `dropout = 0.2` ($87.86\% \pm 0.25\%$ CV score) and achieves $89.26\%$ final test accuracy[cite: 3].

---

## Citation

```bibtex
@misc{hamza_talha_dlp_assignment1_2026,
  author = {M. Hamza and M. Talha},
  title = {Deep Learning for Perception Assignment 1: Fashion-MNIST MLP Experimental Study},
  year = {2026},
  month = {September},
  institution = {FAST-NUCES},
  note = {Course: Deep Learning for Perception, Section BCS-7E}
}
```
