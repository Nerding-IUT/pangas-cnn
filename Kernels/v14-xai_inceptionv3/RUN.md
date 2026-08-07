# v14-xai_inceptionv3

**Slug:** `farhantahsinkhan/aqua20-v14-xai-inceptionv3`

**Pushed:** TBD  |  **Status:** not yet pushed

URL: https://www.kaggle.com/code/farhantahsinkhan/aqua20-v14-xai-inceptionv3

## Purpose

Full XAI battery (**GradCAM + LIME + SHAP + 5 Faithfulness & Agreement metrics**) for **inceptionv3** over the 39 fixed manifest images (`v5-manifest`).

## Config

| | |
|---|---|
| Model | `inceptionv3` |
| Checkpoint kernel | `aqua20-v9-full-inceptionv3` |
| Manifest kernel | `aqua20-v5-manifest` (39 images) |
| `SMOKE` | False (full run) |
| LIME samples | 1000 |
| SHAP samples | 200 (GradientExplainer, 50-img background) |
| GPU | NvidiaTeslaT4 |
| Estimated runtime | ~8 min |

## Result

*(to be filled after run)*
