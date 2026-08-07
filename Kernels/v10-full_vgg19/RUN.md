# v10-full_vgg19

**Slug:** `farhantahsinkhan/aqua20-v10-full-vgg19`

**Pushed:** 2026-08-07  |  **Status:** ✅ COMPLETE

URL: https://www.kaggle.com/code/farhantahsinkhan/aqua20-v10-full-vgg19

## Purpose

Full 25+75 epoch training run for **VGG19**.

VGG19 (143M). Largest model in the roster by param count.

Paper baseline for this architecture: **77.98%**

## Config

| | |
|---|---|
| Model | `vgg19` |
| Sampler | WeightedRandomSampler (on) |
| Epochs | 25 (head) + 75 (fine-tune) |
| Selection metric | macro_f1 |
| Seed | 42 |
| GPU | NvidiaTeslaT4 |
| `TIME_BUDGET_SEC` | `7.0 * 3600` |
| Estimated wall clock | ~5 h |
| min_lr (Stage 1) | 1e-7 |
| min_lr (Stage 2) | 1e-7 |

## Result

| Metric | Value |
|---|---|
| Top-1 accuracy | **66.07%** (Paper baseline: 77.98%) |
| Top-3 accuracy | 90.45% |
| Macro F1 | **0.5352** |
| Macro Precision | 0.5104 |
| Macro Recall | 0.6105 |
| Best checkpoint epoch | Stage 2, Epoch 56 |
| Best val macro-F1 | 0.5722 |
| Errors count | 547 / 1612 (33.9%) |
| Errors where true was rank 2 | 245 (44.79%) |
| Stage-2 epochs run | 75 / 75 |
