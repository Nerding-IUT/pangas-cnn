# v9-full_inceptionv3

**Slug:** `farhantahsinkhan/aqua20-v9-full-inceptionv3`

**Pushed:** 2026-08-07  |  **Status:** ✅ COMPLETE

URL: https://www.kaggle.com/code/farhantahsinkhan/aqua20-v9-full-inceptionv3

## Purpose

Full 25+75 epoch training run for **InceptionV3**.

InceptionV3 (27M). CAM map is 5x5 at 224px (documented caveat, kept for cross-model comparability).

Paper baseline for this architecture: **76.36%**

## Config

| | |
|---|---|
| Model | `inceptionv3` |
| Sampler | WeightedRandomSampler (on) |
| Epochs | 25 (head) + 75 (fine-tune) |
| Selection metric | macro_f1 |
| Seed | 42 |
| GPU | NvidiaTeslaT4 |
| `TIME_BUDGET_SEC` | `3.0 * 3600` |
| Estimated wall clock | ~1.5 h |
| min_lr (Stage 1) | 1e-7 |
| min_lr (Stage 2) | 1e-7 |

## Result

| Metric | Value |
|---|---|
| Top-1 accuracy | **83.68%** (+7.32 pp over paper's 76.36%) |
| Top-3 accuracy | 96.84% |
| Macro F1 | **0.7863** |
| Macro Precision | 0.7709 |
| Macro Recall | 0.8188 |
| Best checkpoint epoch | Stage 2, Epoch 34 |
| Best val macro-F1 | 0.7849 |
| Errors count | 263 / 1612 (16.3%) |
| Errors where true was rank 2 | 151 (57.41%) |
| Stage-2 epochs run | 75 / 75 |
