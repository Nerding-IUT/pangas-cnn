# Model Dossier: VGG-19

This dossier analyzes the explainability and saliency map performance for **VGG-19** (the bottom-tier CNN).

---

## 1. Performance Overview
- **Ours Top-1 Accuracy:** **66.07%** (vs. paper's **77.98%**, falling behind by **-11.91 pp**)
- **Macro F1 Score:** **0.5352**
- **Sanity Check:** **PASSED** (all saliency methods beat random baseline)

---

## 2. Explainability Summary

| Method | Mean Deletion AUC↓ | Mean Insertion AUC↑ | Mean Concentration↑ |
|:---|:---|:---|:---|
| **Grad-CAM** | 0.3220 | 0.6485 | 0.5450 |
| **LIME** | 0.2527 | 0.6942 | 0.4069 |
| **SHAP** | 0.1673 | 0.4056 | 0.5847 |

- **High Concentration:** VGG-19 exhibits the highest Grad-CAM Concentration (0.5450). However, this is not a sign of better accuracy, but of **overfitting**. The model focuses intensely on extremely small, localized patches of high contrast (e.g. highlights, edges of rocks), completely ignoring the overall animal structure.
- **Low Insertion AUC:** Its insertion AUC is extremely low (Grad-CAM: 0.6485, SHAP: 0.4056). Adding back salient pixels does not restore confidence efficiently because the model's features are highly fragmented.

---

## 3. Notable Weaknesses
- **Rare Class Failure (0/4 correct):** Fails on all rare class images in the manifest. Saliency maps show it focuses on high-contrast backgrounds rather than the animals.
- **Overfitting Failure:** With 140M+ parameters, the network overfitted on specific training background textures, rendering it fragile on unseen test images.

---

## 4. Instructive Misclassifications
1. **Test Index #319 (Coral labeled, predicted FishInGroups with 0.457 conf):** Focuses on a localized pattern on the coral surface, confusing its repeating texture with a school of fish.
2. **Test Index #1025 (Flatworm labeled, predicted SeaCucumber with 0.638 conf):** Completely misses the flatworm body, focusing on a background rock shape resembling a sea cucumber.
3. **Test Index #350 (Crab labeled, predicted Rayfish with 0.215 conf):** The model attends to the texture of the sandy floor, leading to a weak prediction of `rayfish`.
