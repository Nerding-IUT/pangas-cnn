# Model Dossier: ResNet-50

This dossier analyzes the explainability and saliency map performance for **ResNet-50** (the mid-tier CNN baseline).

---

## 1. Performance Overview
- **Ours Top-1 Accuracy:** **85.61%** (vs. paper's **82.69%**, beating it by **+2.92 pp**)
- **Macro F1 Score:** **0.7588**
- **Sanity Check:** **PASSED** (all saliency methods beat random baseline)

---

## 2. Explainability Summary

| Method | Mean Deletion AUC↓ | Mean Insertion AUC↑ | Mean Concentration↑ |
|:---|:---|:---|:---|
| **Grad-CAM** | 0.3633 | 0.7401 | 0.2527 |
| **LIME** | 0.3441 | 0.7216 | 0.3361 |
| **SHAP** | 0.1929 | 0.5156 | 0.5475 |

- **Diffuse Grad-CAM maps:** ResNet-50 exhibits a very low concentration (0.2527) for Grad-CAM. Saliency maps are broad and diffuse, often highlighting surrounding water and background reef structures rather than focusing strictly on the subject animal.
- **Sensitivity:** Because its representations are less robust than ConvNeXt's, its Deletion AUC is lower (~0.36), showing that deleting salient regions results in a faster confidence drop.

---

## 3. Notable Weaknesses & Rare Class Failure
- **Rare Class Failure (0/4 correct):** Fails on all rare class images in the manifest (crab -> predicted turtle, dolphin -> predicted rayfish, octopus -> predicted fish, shrimp -> predicted crab).
- Saliency maps for these rare classes show that ResNet-50's attention completely bypasses the animals, focusing instead on background sand ripples or water textures. The model has not learned features specifically mapping to these rare categories.
- **Paper Confusion Pairs (1/8 correct):** Only correct on seaSlug (#1425), failing on all other difficult confusion pairs.

---

## 4. Instructive Misclassifications
1. **Test Index #1070 (Dolphin labeled, predicted Rayfish with 0.993 conf):** The model focuses on the flat, sandy bottom rather than the dolphin swimming above. It confuses the scene context with the flat, bottom-dwelling habitat of a rayfish.
2. **Test Index #1478 (Shrimp labeled, predicted Crab with 0.999 conf):** Attention is concentrated on the sandy seafloor around the shrimp, leading to a confident misclassification of `crab`.
3. **Test Index #319 (Coral correct, but predicted VGG19 failed):** ResNet-50 correctly predicted coral, but its Grad-CAM map is very diffuse, indicating it relied on the overall background reef texture rather than local animal features.
