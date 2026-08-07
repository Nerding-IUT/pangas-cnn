# Model Dossier: InceptionV3

This dossier analyzes the explainability and saliency map performance for **InceptionV3** (the bottom-tier CNN).

---

## 1. Performance Overview
- **Ours Top-1 Accuracy:** **83.68%** (vs. paper's **76.36%**, beating it by **+7.32 pp**)
- **Macro F1 Score:** **0.7863**
- **Sanity Check:** **PASSED** (all saliency methods beat random baseline)

---

## 2. Explainability Summary

| Method | Mean Deletion AUC↓ | Mean Insertion AUC↑ | Mean Concentration↑ |
|:---|:---|:---|:---|
| **Grad-CAM** | 0.2815 | 0.6186 | 0.2554 |
| **LIME** | 0.2292 | 0.6518 | 0.4328 |
| **SHAP** | 0.1929 | 0.6872 | 0.5769 |

- **Coarse Feature Maps:** Due to the input size constraint (224x224 instead of its native 299x299), the final convolutional layer emits a very coarse **5x5 grid**. This results in highly blocky and pixelated Grad-CAM heatmaps, yielding a low concentration score (0.2554).
- **Fragile Representation:** Has a very low Deletion AUC (~0.28 for Grad-CAM, ~0.22 for LIME), showing that the model's confidence collapses immediately when its local salient regions are blurred.

---

## 3. Notable Weaknesses
- **Paper Confusion Pairs (0/8 correct):** Fails on all 8 paper confusion pair images. Saliency maps show highly scattered and coarse attention block shapes.
- **Challenging Conditions Failure (2/4 correct):** Fails on octopus (#1075) and fish (#725) under poor lighting and contrast. Its low-resolution feature representation cannot resolve details in low-contrast scenes.

---

## 4. Instructive Misclassifications
1. **Test Index #1075 (Octopus labeled, predicted SeaSlug with 0.855 conf):** Saliency maps focus on a blurry shape on the reef. The coarse features confuse the octopus's camouflaged shape with a sea slug.
2. **Test Index #725 (Fish labeled, predicted FishInGroups with 0.999 conf):** Replaces single fish with `fishInGroups` because the coarse receptive fields blur the individual fish boundary, mimicking a group texture.
3. **Test Index #354 (Crab labeled, predicted Rayfish with 0.949 conf):** Attention shifts entirely to the sand context surrounding the crab, mistaking it for the sandy backdrop typical of a rayfish.
