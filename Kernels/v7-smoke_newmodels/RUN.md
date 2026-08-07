# v7-smoke_newmodels

**Slug:** `farhantahsinkhan/aqua20-v7-smoke-newmodels`

**Pushed:** TBD · **Status:** not yet pushed

URL: https://www.kaggle.com/code/farhantahsinkhan/aqua20-v7-smoke-newmodels

## Purpose

Integration smoke for the three new architectures: **SwinV2-B, InceptionV3, VGG19**.

This is NOT a training run. The purpose is to catch integration traps in ~10 minutes
rather than discovering them hours into a 6–7 h training run. Catches:

| Model | Trap | Check |
|---|---|---|
| InceptionV3 | `aux_logits=True` makes `forward()` return a namedtuple in train mode, breaking `outputs.max(1)` | asserts `model.aux_logits == False` after build |
| VGG19 | `features[-1]` is MaxPool (7×7), not a feature layer | asserts CAM target is `ReLU` |
| Swin | `channels_last=True` path has **never executed** in this project | asserts CAM is non-degenerate after the permute |

Exit criteria (all three must pass before pushing v8/v9/v10):
- Each model builds, forward+backward runs without error
- Output shape is (batch, 20) for all three
- Grad-CAM heatmap is non-zero and non-uniform for all three
- Architecture-specific assertions above all pass
- Script prints "ALL MODELS PASSED"

## Config

| | |
|---|---|
| Kernel type | script (`smoke_newmodels.py`) — not a notebook |
| GPU | NvidiaTeslaT4 |
| Internet | on (downloads pretrained ImageNet weights) |
| `kernel_sources` | none |

## Result

*(to be filled after run)*
