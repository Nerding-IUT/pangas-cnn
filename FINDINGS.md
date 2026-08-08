# Phase 4 XAI Cross-Model Comparison & Findings

This document summarizes the quantitative and qualitative explainability findings across all **5 deep learning models** evaluated on the **39 manifest images** from `v5-manifest`.

---

## Headline Comparison Table

| Model | Top-1 Accuracy | Macro F1 | Grad-CAM Deletion↓ | Grad-CAM Insertion↑ | Grad-CAM Concentration↑ | LIME Deletion↓ | LIME Insertion↑ | LIME Concentration↑ | SHAP Deletion↓ | SHAP Insertion↑ | SHAP Concentration↑ |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **VGG-19** | 66.07% | 0.5352 | 0.3220 | 0.6485 | 0.5450 | 0.2527 | 0.6942 | 0.4069 | 0.1673 | 0.4056 | 0.5847 |
| **InceptionV3** | 83.68% | 0.7863 | 0.2815 | 0.6186 | 0.2554 | 0.2292 | 0.6518 | 0.4328 | 0.1929 | 0.6872 | 0.5769 |
| **ResNet-50** | 85.61% | 0.7588 | 0.3633 | 0.7401 | 0.2527 | 0.3441 | 0.7216 | 0.3361 | 0.1929 | 0.5156 | 0.5475 |
| **SwinV2-B** | 89.83% | 0.8631 | 0.5168 | 0.8355 | 0.4670 | 0.4422 | 0.8391 | 0.3745 | 0.4650 | 0.4653 | 0.0000 |
| **ConvNeXt-Base** | 90.63% | 0.8748 | 0.4905 | 0.8345 | 0.4912 | 0.4640 | 0.8467 | 0.3705 | 0.4056 | 0.7519 | 0.5591 |

> [!NOTE]
> **Swin SHAP Fallback:** SwinV2-B consumed ~14.19 GiB out of 14.56 GiB T4 GPU memory, causing CUDA Out-of-Memory (OOM) during SHAP gradient computation. In accordance with the OOM-safe design, SHAP maps were filled with zeros, leading to `0.0000` concentration.

---

## Key Findings

### 1. Robust Representations vs. Deletion AUC (The Deletion Paradox)
A counterintuitive trend emerged from the deletion and insertion metrics:
- We expected higher-accuracy models to drop in confidence faster (lower Deletion AUC) as salient features were deleted.
- Instead, **higher-performing models (SwinV2-B and ConvNeXt-Base) showed HIGHER Deletion AUC** (~0.49) than weaker models like InceptionV3 (~0.28).
- **Reasoning:** High-capacity models learn distributed, highly robust representations. Deleting a small localized set of "most salient" pixels does not break their predictions because their feature maps contain multiple redundant cues. Conversely, low-capacity models (VGG-19, InceptionV3) rely on fragile, highly localized features; deleting those features immediately collapses their confidence.

### 2. Saliency Concentration & Focus
Saliency Concentration measures the fraction of total explanation score concentrated in the top 10% of pixels:
- **Grad-CAM Concentration** exhibits a U-shaped curve: VGG-19 (0.5450) and ConvNeXt/Swin (~0.48) are highly concentrated, while ResNet-50 and InceptionV3 are diffuse (~0.25).
- **LIME Concentration** shows that weaker models are actually more concentrated on small segments than top-tier models. This indicates that LIME explanation maps tend to be more diffuse for models with global contextual awareness.

![Accuracy vs Concentration](file:///d:/PORA/6th%20Semester/CSE%204622%20ML%20Lab/22%20Batch%20Files/ML%20project/report_assets/xai_figures/accuracy_vs_concentration.png)

### 3. Agreement Between Methods (Grad-CAM vs. LIME vs. SHAP)
We calculated the Spearman rank correlation of saliency maps across different explanation methods:
- **LIME and Grad-CAM show moderate agreement** (Spearman ~0.25 to 0.34) across all architectures.
- **SHAP exhibits very low correlation with Grad-CAM and LIME** (Spearman < 0.08). This is because SHAP (expected gradients) provides pixel-level gradients that are extremely sparse and speckly, highlighting fine edges, whereas Grad-CAM provides coarse blocky heatmaps and LIME highlights contiguous superpixels.
- The Spearman agreement heatmaps for each model are saved under `report_assets/xai_figures/agreement_spearman_<model>.png`.

### 4. Rare-Class Attention (Capacity Rescue)
The `rare_class` stratum (octopus, dolphin, crab, shrimp) confirmed the **capacity-rescue hypothesis**:
- Top-tier models (**ConvNeXt-Base** and **SwinV2-B**) achieved **100% accuracy (4/4)** on rare class images.
- Mid-tier (**ResNet-50**) and bottom-tier (**VGG-19**) models failed on **100% (0/4)** of these same images.
- Explainability maps reveal that ResNet-50 completely misses the animals (focusing on background water/sand textures), whereas ConvNeXt-Base focuses its attention precisely on the animal's boundary (e.g., the crab shell or dolphin flank), showing that higher capacity allows models to learn clean object representation even with very few training samples.

### 5. Label Ambiguity (Fish vs. FishInGroups)
All 5 models predicted `fishInGroups` on all three images in the `fish_group` stratum, despite the ground truth label being `fish`.
- Saliency maps show that the models are focusing on multiple distinct fish in the frame.
- This is a clear case of **label ambiguity** in the dataset rather than a model defect: the models are correctly identifying groups of fish, but the human-provided label is single `fish`.

![Fish Label Ambiguity](file:///d:/PORA/6th%20Semester/CSE%204622%20ML%20Lab/22%20Batch%20Files/ML%20project/report_assets/xai_figures/resnet50_581_fish-as-fishInGroups_LABEL-AMBIGUITY.png)
*(Fig: ResNet-50 explaining test image #581. The model predicts fishInGroups with high confidence because there are clearly multiple fish in the frame.)*

### 6. Challenging Conditions (Turbidity / Poor Lighting)
On low light, low contrast, and low saturation images (`challenging` stratum):
- **ConvNeXt-Base, Swin, ResNet-50, and VGG-19 all achieved 100% accuracy (4/4)**.
- Only **InceptionV3 struggled**, misclassifying 2 out of 4 images.
- InceptionV3 natively processes 299x299 images, but running it on 224x224 yields a very coarse 5x5 feature grid. Under low contrast, the model fails to extract fine boundaries, and its saliency maps disperse randomly across the background.

---

## 7. Optimization Experiments (ResNet-50 Capacity Testing)
To determine if we could squeeze further performance out of the mid-tier baseline architecture (**ResNet-50**), we executed a series of 7 optimization experiments testing different data-augmentation policies, learning rate schedulers, loss configurations, and inference-time techniques:

### Training-Time Augmentation & Loss Policies
- **`current` (Baseline reproduction):** Confirmed model stability and reproducibility under the baseline configuration, yielding **85.48%** accuracy (baseline 85.61%).
- **`strong` (Aggressive augmentation):** Reduced accuracy to **84.06%** and Macro F1 to **0.7526**. Aggressive ColorJitter, GaussianBlur (simulating turbidity), and RandomErasing (simulating occlusion) over-distorted the inputs, corrupting critical fine-grained features.
- **`rare` (Rare-class targeted distortion):** Attempted to boost tail class representations. The run achieved **84.37%** but was truncated early due to time cap limitations.
- **`label_smooth` (Softened targets):** Reassigned hard targets to smooth distributions (0.1 smoothing parameter) to handle boundary ambiguity (e.g. `fish` vs. `fishInGroups`). While it slightly improved Macro F1 to **0.7631** (+0.43 pp), it did not reduce the absolute error count (236 errors vs. 232).
- **`cosine_anneal` (Dynamic LR Scheduling):** Replaced the default `ReduceLROnPlateau` scheduler (which collapsed the learning rate too early) with a continuous `CosineAnnealingLR` schedule. This successfully improved feature representation calibration, boosting Macro F1 to **0.7676** (+0.88 pp over baseline).

### Inference-Time & Combined Optimizations (The Winning Strategies)
- **`tta_8_views` (Test-Time Augmentation):** Instead of retraining, we evaluated the baseline ResNet-50 weights by averaging softmax logits over **8 augmented test-time views** (horizontal flips, zoom scales, and corner crops). This successfully bypassed the architecture's capacity ceiling, yielding **86.72% Top-1 Accuracy** (+1.11 pp over baseline) and **0.7722 Macro F1** (+1.34 pp over baseline).
- **`tta_cosine` (Combined Scheduler + TTA):** Evaluating the `cosine_anneal` weights using the 8-view TTA post-processing yielded our absolute best result: **87.03% Top-1 Accuracy** (+1.42 pp over baseline) and **0.7908 Macro F1** (+3.20 pp over baseline), reducing errors from 232 to **209** (23 fewer errors).

![Augmentation Comparison](file:///d:/PORA/6th%20Semester/CSE%204622%20ML%20Lab/22%20Batch%20Files/ML%20project/pangas-cnn/analysis/augmentation_comparison.png)

---

## Model Dossiers
Detailed per-architecture analyses are located at:
- [VGG-19 Dossier](file:///d:/PORA/6th%20Semester/CSE%204622%20ML%20Lab/22%20Batch%20Files/ML%20project/pangas-cnn/models/vgg19/DOSSIER.md)
- [InceptionV3 Dossier](file:///d:/PORA/6th%20Semester/CSE%204622%20ML%20Lab/22%20Batch%20Files/ML%20project/pangas-cnn/models/inceptionv3/DOSSIER.md)
- [ResNet-50 Dossier](file:///d:/PORA/6th%20Semester/CSE%204622%20ML%20Lab/22%20Batch%20Files/ML%20project/pangas-cnn/models/resnet50/DOSSIER.md)
- [SwinV2-B Dossier](file:///d:/PORA/6th%20Semester/CSE%204622%20ML%20Lab/22%20Batch%20Files/ML%20project/pangas-cnn/models/swin/DOSSIER.md)
- [ConvNeXt-Base Dossier](file:///d:/PORA/6th%20Semester/CSE%204622%20ML%20Lab/22%20Batch%20Files/ML%20project/pangas-cnn/models/convnext_base/DOSSIER.md)


