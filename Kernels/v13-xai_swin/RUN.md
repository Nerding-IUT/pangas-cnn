# v13-xai_swin

**Slug:** `farhantahsinkhan/aqua20-v13-xai-swin`

**Pushed:** TBD  |  **Status:** not yet pushed

URL: https://www.kaggle.com/code/farhantahsinkhan/aqua20-v13-xai-swin

## Purpose

Full XAI battery (**GradCAM + LIME + SHAP + 5 Faithfulness & Agreement metrics**) for **swin** over the 39 fixed manifest images (`v5-manifest`).

## Config

| | |
|---|---|
| Model | `swin` |
| Checkpoint kernel | `aqua20-v8-full-swin` |
| Manifest kernel | `aqua20-v5-manifest` (39 images) |
| `SMOKE` | False (full run) |
| LIME samples | 1000 |
| SHAP samples | 200 (GradientExplainer, 50-img background) |
| GPU | NvidiaTeslaT4 |
| Estimated runtime | ~8 min |

## Result

*(to be filled after run)*
