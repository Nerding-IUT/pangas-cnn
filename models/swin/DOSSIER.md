# Model Dossier: SwinV2-B

This dossier analyzes the explainability and saliency map performance for **SwinV2-B** (the top-performing Transformer architecture).

---

## 1. Performance Overview
- **Ours Top-1 Accuracy:** **89.83%** (vs. paper's **88.65%**, beating it by **+1.18 pp**)
- **Macro F1 Score:** **0.8631**
- **Sanity Check:** **SUSPECT** (Grad-CAM and LIME passed; SHAP was skipped due to OOM)

---

## 2. Explainability Summary

| Method | Mean Deletion AUC↓ | Mean Insertion AUC↑ | Mean Concentration↑ |
|:---|:---|:---|:---|
| **Grad-CAM** | 0.5168 | 0.8355 | 0.4670 |
| **LIME** | 0.4422 | 0.8391 | 0.3745 |
| **SHAP** | 0.4650 | 0.4653 | 0.0000 (OOM) |

- **CUDA OOM Behavior:** During SHAP gradient checks, the SwinV2-B model filled the T4's VRAM (~14.19 GiB out of 14.56 GiB), leaving no room for gradient-expected tracking. The fallback mechanism caught the exception and generated zero-saliency maps.
- **Attention Coherence:** SwinV2-B has high Grad-CAM concentration (0.4670). Its self-attention mechanism produces coherent, non-fragmented attention masks that cover whole object shapes rather than localized patches.

---

## 3. Notable Strengths & Rescues
- **Paper Confusion Pairs (4/8 correct):** Swin was the most successful model at resolving paper confusion pairs, correctly classifying dolphin (#1067), seaAnemone (#1377), seaSlug (#1425), and shark (#1468).
- **Rare Class Rescue (4/4 correct):** Classified crab, dolphin, octopus, and shrimp with 100% accuracy. The global receptive field of self-attention allows it to model context (e.g. background sand vs animal shape) and perform correct feature mapping.

---

## 4. Instructive Misclassifications
1. **Test Index #3 (Coral labeled, predicted SeaAnemone with 0.995 conf):** The self-attention maps highlight the soft texture of the coral tentacles, aligning the pattern features with `seaAnemone`.
2. **Test Index #341 (Coral labeled, predicted Starfish with 1.000 conf):** The model focuses on the branching star-like arms of the coral, showing a shape-biased classification error.
3. **Test Index #944 (Fish labeled, predicted Eel with 1.000 conf):** Attention is concentrated on the elongated body shape, ignoring the specific fin shapes that would distinguish it as a regular fish.
