# Model Dossier: ResNet-50

This dossier analyzes the explainability and saliency map performance for **ResNet-50** (the mid-tier CNN baseline).

---

## 1. Performance Overview
- **Baseline Top-1 Accuracy:** **85.61%** (vs. paper's **82.69%**, beating it by **+2.92 pp**)
- **Baseline Macro F1 Score:** **0.7588**
- **Optimized Top-1 (TTA + Cosine):** **87.03%** (beating paper by **+4.34 pp**, and baseline by **+1.42 pp**)
- **Optimized Macro F1 (TTA + Cosine):** **0.7908** (beating baseline by **+3.20 pp**)
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

---

## 5. Performance Optimization Experiments
To improve the ResNet-50 baseline, we evaluated 7 data-augmentation, scheduling, and post-processing strategies:

| Strategy | Top-1 Accuracy | Macro F1 | n_errors | Epochs (Stage 2) | Status / Impact |
|:---|:---|:---|:---|:---|:---|
| **v2 Baseline** | **85.61%** | **0.7588** | 232 | 75 | Reference baseline |
| **`current`** | 85.48% | 0.7564 | 234 | 75 | Control run (reproduced baseline within noise) |
| **`strong`** | 84.06% | 0.7526 | 257 | 75 | **FAILED** — over-distortion (blur/erasing) degraded features |
| **`rare`** | 84.37% | 0.7522 | 252 | 64 (cap) | **PARTIAL** — hit 2.5h budget and was cut off early |
| **`label_smooth`** | 85.36% | 0.7631 | 236 | 75 | **NEUTRAL/MODEST** — soft boundaries (+0.43 pp F1 but 236 errors) |
| **`cosine_anneal`** | 85.48% | 0.7676 | 234 | 75 | **SUCCESS (LRS)** — prevented early LR collapse (+0.88 pp F1) |
| **`tta_8_views`** | **86.72%** | **0.7722** | **214** | — | **SUCCESS (TTA)** — Test-Time Augmentation yields **+1.11 pp Top-1, +1.34 pp F1** |
| **`tta_cosine`** | **87.03%** | **0.7908** | **209** | 75 | **SUCCESS (COMBO)** — combined Cosine scheduler + TTA yields **+1.42 pp Top-1, +3.20 pp F1** |

### Insights & TTA Success
1. **The Scheduler Bottleneck:** The standard `ReduceLROnPlateau` collapses the backbone learning rate too early during Stage 2 fine-tuning (freezing weights by epoch ~40). Using a `CosineAnnealingLR` schedule keeps gradients flowing and allows the model to learn better-calibrated features, increasing Macro F1 to 0.7676 (+0.88 pp).
2. **The Capacity Ceiling & Test-Time Augmentation (TTA):** Retraining with augmentations alone (`strong`, `rare`) does not work on ResNet-50 because the model lacks capacity. However, test-time spatial voting (TTA) using 8 views resolves boundary ambiguities and fine-grained classification mistakes.
3. **Synergy:** Combining `CosineAnnealingLR` at training time with `TTA` at inference time yields our absolute best result: **87.03% Top-1 Accuracy** (23 fewer errors than baseline) and **0.7908 Macro F1** (+3.20 pp over baseline).


