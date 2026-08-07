# v12-xai_convnext_base

**Slug:** `farhantahsinkhan/aqua20-v12-xai-convnext-base`

**Pushed:** TBD  |  **Status:** not yet pushed

URL: https://www.kaggle.com/code/farhantahsinkhan/aqua20-v12-xai-convnext-base

## Purpose

Full XAI battery (**GradCAM + LIME + SHAP + 5 Faithfulness & Agreement metrics**) for **convnext_base** over the 39 fixed manifest images (`v5-manifest`).

## Config

| | |
|---|---|
| Model | `convnext_base` |
| Checkpoint kernel | `aqua20-v4-full-convnext-base` |
| Manifest kernel | `aqua20-v5-manifest` (39 images) |
| `SMOKE` | False (full run) |
| LIME samples | 1000 |
| SHAP samples | 200 (GradientExplainer, 50-img background) |
| GPU | NvidiaTeslaT4 |
| Estimated runtime | ~8 min |

## Result

*(to be filled after run)*
