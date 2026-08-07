# v8-full_swin

**Slug:** `farhantahsinkhan/aqua20-v8-full-swin`

**Pushed:** 2026-08-07  |  **Status:** ✅ COMPLETE

URL: https://www.kaggle.com/code/farhantahsinkhan/aqua20-v8-full-swin

## Purpose

Full 25+75 epoch training run for **swin** (SwinV2-B, 88M params).

Paper baseline for this architecture: **88.65%**

## Config

| | |
|---|---|
| Model | `swin` |
| Sampler | WeightedRandomSampler (on) |
| Epochs | 25 (head) + 75 (fine-tune) |
| Selection metric | macro_f1 |
| Seed | 42 |
| GPU | NvidiaTeslaT4 |
| `TIME_BUDGET_SEC` | `8.5 * 3600` |
| Estimated wall clock | ~6-7 h |
| min_lr (Stage 1) | 1e-7 |
| min_lr (Stage 2) | 1e-7 |

## Result

| Metric | Value |
|---|---|
| Top-1 accuracy | **89.83%** (+1.18 pp over paper's 88.65%) |
| Top-3 accuracy | **99.26%** |
| Macro F1 | **0.8631** |
| Macro Precision | 0.8650 |
| Macro Recall | 0.8735 |
| Best checkpoint epoch | Stage 2, Epoch 32 |
| Best val macro-F1 | 0.8709 |
| Errors count | 164 / 1612 (10.2%) |
| Errors where true was rank 2 | 124 (75.61%) |
| Stage-2 epochs run | 75 / 75 |
