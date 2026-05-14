# An Explainable System for Mitotic Cell Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![Run on Kaggle](https://img.shields.io/badge/Run%20on-Kaggle-20BEFF?logo=kaggle&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Model-Virchow2-FFD21E?logo=huggingface&logoColor=black)
![PEFT](https://img.shields.io/badge/Fine--tuning-LoRA%20%2F%20PEFT-7C3AED?logo=python&logoColor=white)
![W&B](https://img.shields.io/badge/Tracked%20with-W%26B-FFBE00?logo=weightsandbiases&logoColor=black)
![Balanced Accuracy](https://img.shields.io/badge/Balanced%20Accuracy-0.861-22C55E)
![ROC AUC](https://img.shields.io/badge/ROC--AUC-0.945-22C55E)
![License](https://img.shields.io/badge/License-MIT-16A34A)

> Binary classification of **Atypical Mitotic Figures (AMF)** vs **Normal Mitotic Figures (NMF)** in H&E-stained breast histopathology images, using a parameter-efficiently fine-tuned pathology foundation model with built-in explainability.

---

## Overview

Atypical mitotic figures are an independent prognostic marker in breast cancer, yet their manual identification is time-consuming, subjective, and subject to high inter-pathologist variability (~78% agreement). This project builds an automated binary classifier that distinguishes AMF from NMF, achieves **0.861 Balanced Accuracy** on a held-out test set, and produces visual explanations for every prediction using Attention Rollout and Grad-CAM++.

The pipeline fine-tunes [**Virchow2**](https://huggingface.co/paige-ai/Virchow2) - a ViT-H/14 pathology foundation model - using **Low-Rank Adaptation (LoRA)**, which keeps 99.2% of the model frozen and trains only 5.3 million parameters. This makes the full pipeline trainable on a single Kaggle T4 GPU (16 GB VRAM) within a 12-hour session.

---

## Exploratory Data Analysis

A comprehensive exploration of the combined MIDOG25 + AMI-BR dataset before any model training. Covers:

| Section | What it shows |
|---|---|
| Dataset Overview | Class distribution, dataset source breakdown, split sizes |
| Augmentation Analysis | Original vs augmented ratio, augmentation type breakdown per class |
| Cross-Tabulation Heatmaps | Split × Class, Dataset × Class, Dataset × Split interactions |
| Class Balance per Split | Stacked bar showing AMF/NMF percentage per split |
| Augmentation Type × Class | Which augmentation types were applied to which classes in training |
| Sample Images | Side-by-side NMF vs AMF original patches |
| Augmentation Showcase | One original image next to all four of its augmented variants |
| Summary Table | Full numeric breakdown of counts, AMF%, original vs augmented per split |

**Key finding highlighted:** The training set is approximately 48% AMF / 52% NMF (after 4× augmentation of the minority class), while the validation and test sets reflect the natural distribution of ~16% AMF / 84% NMF - a distribution mismatch that directly motivated the threshold tuning.

---

## Training, Evaluation & Explainability

The full end-to-end machine learning pipeline. Runs sequentially, top to bottom, on a Kaggle GPU notebook. Organized into the following sections:

| Section | Description |
|---|---|
| Setup & Imports | Environment setup, deterministic seeds, GPU info, HuggingFace + W&B authentication |
| Configuration | Single `CFG` class - all paths and hyperparameters in one place |
| Data Loading & Verification | CSV parsing, file existence checks, class balance validation, sample visualization |
| Custom Dataset Class | PyTorch `Dataset` returning `(image_tensor, label, filename)` |
| Augmentation Pipelines | Training: D4 symmetry, HED stain jitter, RandStainNA-lite, color jitter, random erasing. Val/Test: resize + normalize only |
| Model: Virchow2 + LoRA | Backbone loading, CLS⊕mean(patch) pooling, LoRA wrapping, gradient checkpointing |
| Loss, Optimizer, Scheduler | Focal Loss, two-group AdamW, cosine LR with warmup |
| MixUp | Standard MixUp implementation for regularization |
| Training Loop | fp16 AMP, gradient accumulation, per-epoch validation, best-checkpoint saving |
| 4-Fold Cross-Validation | StratifiedKFold driver, WeightedRandomSampler, W&B per-fold logging |
| Training Curves | Loss and balanced accuracy curves per fold |
| TTA + Ensemble Inference | 8-view D4 test-time augmentation averaged across all 4 fold checkpoints |
| Threshold Tuning | Sweep P(AMF) threshold on validation set, select maximum balanced accuracy |
| Final Test Evaluation | One-shot evaluation on held-out test set with full metric suite |
| Calibration | Temperature scaling via LBFGS, ECE before/after, reliability diagrams |
| Explainability Setup | Register-aware `reshape_transform` for Virchow2, GradCAM++ and AttentionRollout initialization |
| Representative Case Selection | 3 TP / 3 TN / 3 FP / 3 FN selected by confidence from test set |
| Visualization Grid | Side-by-side: Original - Attention Rollout - Grad-CAM++ AMF - Grad-CAM++ NMF |

---

## Dataset

**MIDOG25 + AMI-BR (MIDOG21 + TUPAC16)**

| Split | Total patches | AMF | NMF | AMF % |
|---|---|---|---|---|
| Training (originals) | 8,362 | 1,306 | 7,056 | 15.6% |
| Training (with augmentation) | ~14,000 | ~7,000 | 7,056 | ~50% |
| Validation | 2,788 | 435 | 2,353 | 15.6% |
| Test | 2,788 | 435 | 2,353 | 15.6% |

- **Image format:** 224 × 224 RGB PNG, H&E stained, centered on a single mitotic figure
- **Sources:** MIDOG25 atypical training set, MIDOG21, TUPAC16
- **No bounding boxes or slide identifiers available**
- Dataset available on Kaggle: [`lostluinor/mitoticfigure-spiltandaugmenteddataset`](https://www.kaggle.com/datasets/lostluinor/mitoticfigure-spiltandaugmenteddataset)

**Augmentation types (training AMF only, pre-baked in dataset):**

| Type | Transformation |
|---|---|
| 1 | Rotation 15–30° with white fill |
| 2 | Horizontal flip + brightness ±10% |
| 3 | Vertical flip + contrast ±10% |
| 4 | Rotation 5–15° + sharpness ±10% |

---

## Model Architecture

```
Input (224×224 H&E patch)
    ↓
Virchow2 ViT-H/14  [FROZEN — 627M params]
    ↓ 261 tokens: 1 CLS + 4 register + 256 patch
LoRA adapters on qkv / proj / fc1 / fc2  [TRAINABLE - 5.3M params]
    ↓
Pooling: CLS ⊕ mean(patch tokens)  →  2,560-d vector
    ↓
Linear head  →  2 logits  →  P(NMF), P(AMF)
```

**Virchow2** is a ViT-H/14 pretrained by [Paige AI](https://huggingface.co/paige-ai/Virchow2) on 3.1 million whole-slide images using DINOv2. Access is gated - apply via HuggingFace before running the notebook.

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Loss | Focal Loss (γ=2, α=0.25) |
| Optimizer | AdamW (wd=0.05) |
| LR - LoRA params | 1e-4 |
| LR - classification head | 1e-5 |
| LR schedule | Cosine with 10% warmup |
| Epochs per fold | 10 |
| Batch size | 16 (×4 grad. accum. → effective 64) |
| Cross-validation | 4-fold StratifiedKFold |
| Precision | fp16 AMP + gradient checkpointing |
| Sampler | WeightedRandomSampler (inverse class freq.) |
| MixUp | α=0.2, p=0.5 |
| Seed | 42 |

**Runtime augmentations (training only):**
D4 dihedral symmetry (HFlip + VFlip + RandomRotate90), HED stain jitter (σ=0.05), RandStainNA-style luminance perturbation, ColorJitter (brightness/contrast/saturation/hue), Random Erasing (p=0.25).

---

## Results

### Test set - evaluated exactly once

| Metric | Value |
|---|---|
| **Balanced Accuracy** | **0.861** |
| ROC-AUC | 0.945 |
| F1 - AMF (minority) | 0.627 |
| F1 - Macro | 0.759 |
| MCC | 0.576 |
| AMF Recall | 0.903 |
| AMF Precision | 0.480 |
| ECE (after temperature scaling) | 0.099 |

**Published Virchow2 + LoRA single-model benchmark (Banerjee et al., 2025):** 0.814 BA on AMi-Br. The ensemble + threshold tuning recipe here adds approximately 5 BA points.

### Confusion matrix (test, n = 2,788)

|  | Predicted NMF | Predicted AMF |
|---|---|---|
| **True NMF** | 1,927  | 426  |
| **True AMF** | 42  | 393  |

90.3% of atypical mitoses correctly identified. The high recall / modest precision trade-off is intentional - in a screening setting, a missed atypical mitosis is more costly than a false alarm that a pathologist can quickly reject.

### Cross-validation (4-fold, val BA per fold)

| Fold | Best Val BA |
|---|---|
| 0 | 0.844 |
| 1 | 0.877 |
| 2 | 0.852 |
| 3 | 0.835 |
| **Mean ± Std** | **0.852 ± 0.015** |

---

## Explainability

Two complementary methods are applied post-hoc on the frozen ensemble:

### Attention Rollout *(Abnar & Zuidema, 2020)*
Aggregates the self-attention matrices across all 32 transformer blocks, propagated through the residual connections. The CLS token's attention to the 256 patch tokens is extracted, register tokens are discarded, and the result is reshaped to a 16×16 saliency map and upscaled to 224×224. Shows **where the model attended**, class-agnostic.

### Grad-CAM++
Computes the gradient of the predicted class logit with respect to the normalized features of the last transformer block. The higher-order weighting of Grad-CAM++ yields cleaner localisation than vanilla Grad-CAM on Vision Transformers. Generates **separate heatmaps for AMF and NMF**, making it class-discriminative.

### Visualization grid
A 12-row grid of representative test cases (3 × TP, 3 × TN, 3 × FP, 3 × FN) is produced, each row showing:

```
Original patch | Attention Rollout overlay | Grad-CAM++ (AMF) | Grad-CAM++ (NMF) | Pred/True/Confidence
```

**Qualitative observations:**
- Both methods consistently focus on the central mitotic figure, not on surrounding stroma or background
- AMF and NMF Grad-CAM++ maps are distinct (not inverses), indicating the model has learned class-specific features
- High-confidence predictions produce tighter, more focused heatmaps; low-confidence ones spread across multiple structures

---

## Inference Pipeline

```
Test image
    ↓  × 8 views (D4: 4 rotations × 2 flips)
Forward pass through each fold model (4 total)
    ↓  average softmax probabilities across 8 views × 4 folds = 32 predictions
Ensemble probability P(AMF)
    ↓
Apply tuned threshold (0.37)
    ↓
Final prediction: AMF or NMF
```

Threshold 0.37 was chosen by sweeping [0.10, 0.90] in steps of 0.01 on the validation set, selecting the value that maximises balanced accuracy. This corrects for the train/test distribution mismatch (50/50 training vs 16/84 test).

---

## Setup & Requirements

### Environment
- Kaggle GPU notebook (Tesla T4, 16 GB VRAM)
- Python 3.10+
- CUDA 12.x

### Key dependencies

```txt
torch>=2.2.0
timm>=1.0.11
transformers>=4.41.0
peft>=0.11.0
albumentations>=1.4.0
grad-cam==1.5.4
scikit-image
scikit-learn
wandb
pandas
numpy
matplotlib
seaborn
```

Install in a Kaggle notebook:

```bash
pip install -q --upgrade timm>=1.0.11 peft>=0.11.0 grad-cam==1.5.4 \
    albumentations>=1.4.0 transformers>=4.41.0 scikit-image
```

### HuggingFace access (required)

Virchow2 is a gated model.

1. Create a HuggingFace account
2. Go to [`paige-ai/Virchow2`](https://huggingface.co/paige-ai/Virchow2) and submit an access request using an **institutional email**
3. Once approved, create a HuggingFace token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Add it to your Kaggle notebook as a Secret named `HF_TOKEN` (Add-ons → Secrets)

### W&B experiment tracking (optional)

1. Create a [Weights & Biases](https://wandb.ai) account
2. Add your API key to Kaggle Secrets as `WANDB_API_KEY`
3. The notebook will automatically log training curves, metrics, confusion matrices, and model checkpoints

---

## Limitations

**No slide-level grouping in cross-validation.** The dataset provides no slide identifier, so patches from the same whole-slide image may appear in different folds. This makes CV scores slightly optimistic. The held-out test set is the only fully honest performance estimate.

**Modest AMF precision (48%).** The model is tuned for high recall, which means roughly half of its atypical flags are false positives. In a clinical workflow, a pathologist must verify each positive prediction. The system assists, it does not replace expert review.

**Imperfect calibration.** Expected calibration error of 0.099 remains after temperature scaling. Probability scores should be treated as rankings rather than exact likelihoods.

---

## References

> - Banerjee et al. (2025). *Benchmarking Deep Learning and Vision Foundation Models for Atypical vs. Normal Mitosis Classification with Cross-Dataset Evaluation.* arXiv:2506.21444
> - Zimmermann et al. (2024). *Virchow2: Scaling Self-Supervised Mixed Magnification Models in Pathology.* arXiv:2408.00738
> - Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
> - Abnar & Zuidema (2020). *Quantifying Attention Flow in Transformers.* ACL 2020.
> - Lashen et al. (2022). *The Characteristics and Clinical Significance of Atypical Mitosis in Breast Cancer.* Modern Pathology, 35, 1341–1348.
> - Ochi & Bae (2025). *Ensemble of Pathology Foundation Models for MIDOG 2025 Track 2.* arXiv:2509.02591

---

## License

Code: MIT License.  
Virchow2 model weights: [CC BY-NC-ND 4.0](https://huggingface.co/paige-ai/Virchow2) - non-commercial, academic use only.  
Dataset: see [MIDOG25](https://midog2025.grand-challenge.org/) and [AMI-BR](https://github.com/DeepMicroscopy/AMi-Br_Benchmark) licensing terms.
