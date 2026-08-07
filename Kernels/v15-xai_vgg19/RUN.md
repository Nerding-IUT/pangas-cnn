# v15-xai_vgg19

**Slug:** `farhantahsinkhan/aqua20-v15-xai-vgg19`

**Pushed:** TBD  |  **Status:** not yet pushed

URL: https://www.kaggle.com/code/farhantahsinkhan/aqua20-v15-xai-vgg19

## Purpose

Full XAI battery (**GradCAM + LIME + SHAP + 5 Faithfulness & Agreement metrics**) for **vgg19** over the 39 fixed manifest images (`v5-manifest`).

## Config

| | |
|---|---|
| Model | `vgg19` |
| Checkpoint kernel | `aqua20-v10-full-vgg19` |
| Manifest kernel | `aqua20-v5-manifest` (39 images) |
| `SMOKE` | False (full run) |
| LIME samples | 1000 |
| SHAP samples | 200 (GradientExplainer, 50-img background) |
| GPU | NvidiaTeslaT4 |
| Estimated runtime | ~8 min |

## Result

*(to be filled after run)*
