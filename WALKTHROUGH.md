# Walkthrough — Project Completion & Final Optimization

We have successfully executed the complete Explainable AI (XAI) pipeline, completed the comparative cross-model analysis, and implemented a highly successful performance optimization combining a Cosine learning rate scheduler with Test-Time Augmentation (TTA) on ResNet-50.

---

## 1. Summary of Accomplishments

### Git Conflict Resolution & Integration
- Resolved the merge conflict in `notebook.ipynb` by opting for Arian's branch (`--theirs`), integrating his augmented ResNet-50 dataset setup and LFS checkpoints into the main project repository.

### ResNet-50 Performance Optimization Experiments
We conducted 7 experiments to push ResNet-50 past its baseline limits:
1. **`current` (Baseline reproduction):** Re-trained ResNet-50 to verify model replication stability, yielding **85.48%** accuracy (baseline 85.61%).
2. **`strong` (Aggressive augmentation):** Re-trained ResNet-50 with intensive augmentation (blur, rotation, color, erasing), which degraded accuracy to **84.06%** due to over-distortion.
3. **`rare` (Rare-class targeted distortion):** Targeted minority classes with extra rotations and occlusion transforms, achieving **84.37%** before hitting the time budget cap.
4. **`label_smooth` (v16):** Set up and ran ResNet-50 with a label smoothing parameter of 0.1 to address boundary label ambiguity. This yielded a soft improvement in Macro F1 (**0.7631**, +0.43 pp over baseline) but dropped Top-1 accuracy slightly (**85.36%**).
5. **`cosine_anneal` (v18 - SUCCESS):** Trained ResNet-50 using a continuous `CosineAnnealingLR` scheduler instead of the baseline `ReduceLROnPlateau` scheduler. This prevented early backbone freezing, yielding:
   - **Top-1 Accuracy:** **85.48%**
   - **Macro F1:** **0.7676** (**+0.88 pp** over baseline)
6. **`tta_8_views` (v17 - SUCCESS):** Created and ran a post-processing evaluation kernel averaging softmax outputs across **8 test-time views** (crops, flips, and zoom scales) on the baseline weights, yielding:
   - **Top-1 Accuracy:** **86.72%** (**+1.11 pp** over baseline)
   - **Macro F1:** **0.7722** (**+1.34 pp** over baseline)
7. **`tta_cosine` (v19 - BEST COMBO):** Combined the scheduler and post-processing optimizations by running the 8-view TTA evaluation on the `cosine_anneal` weights, yielding:
   - **Top-1 Accuracy:** **87.03%** (**+1.42 pp** over baseline, **+4.34 pp** over paper)
   - **Macro F1:** **0.7908** (**+3.20 pp** over baseline)
   - **Absolute Errors:** **209** (down from 232 baseline, **23 fewer errors**)

---

## 2. Final ResNet-50 Performance Matrix

| Strategy | Top-1 Accuracy | Macro F1 | n_errors | Stage 2 Epochs | Status |
|:---|:---|:---|:---|:---|:---|
| **v2 Baseline** | **85.61%** | **0.7588** | 232 | 75 | Reference baseline |
| **`current`** | 85.48% | 0.7564 | 234 | 75 | Reproduced control |
| **`strong`** | 84.06% | 0.7526 | 257 | 75 | Failed (over-distorted) |
| **`rare`** | 84.37% | 0.7522 | 252 | 64 (cap) | Partial (time cap hit) |
| **`label_smooth` (v16)** | 85.36% | 0.7631 | 236 | 75 | Neutral (better calibration) |
| **`cosine_anneal` (v18)** | 85.48% | 0.7676 | 234 | 75 | **SUCCESS (LRS)** |
| **`tta_8_views` (v17)** | **86.72%** | **0.7722** | **214** | — | **SUCCESS (TTA)** |
| **`tta_cosine` (v19)** | **87.03%** | **0.7908** | **209** | 75 | **SUCCESS (COMBO)** |

---

## 3. Location of Deliverables

- **Final Analysis & Optimization Summary:** [`FINDINGS.md`](file:///d:/PORA/6th%20Semester/CSE%204622%20ML%20Lab/22%20Batch%20Files/ML%20project/FINDINGS.md)
- **Model-Specific Dossiers:**
  - [VGG-19 Dossier](file:///d:/PORA/6th%20Semester/CSE%204622%20ML%20Lab/22%20Batch%20Files/ML%20project/pangas-cnn/models/vgg19/DOSSIER.md)
  - [InceptionV3 Dossier](file:///d:/PORA/6th%20Semester/CSE%204622%20ML%20Lab/22%20Batch%20Files/ML%20project/pangas-cnn/models/inceptionv3/DOSSIER.md)
  - [ResNet-50 Dossier](file:///d:/PORA/6th%20Semester/CSE%204622%20ML%20Lab/22%20Batch%20Files/ML%20project/pangas-cnn/models/resnet50/DOSSIER.md)
  - [SwinV2-B Dossier](file:///d:/PORA/6th%20Semester/CSE%204622%20ML%20Lab/22%20Batch%20Files/ML%20project/pangas-cnn/models/swin/DOSSIER.md)
  - [ConvNeXt-Base Dossier](file:///d:/PORA/6th%20Semester/CSE%204622%20ML%20Lab/22%20Batch%20Files/ML%20project/pangas-cnn/models/convnext_base/DOSSIER.md)
- **Comparison CSV:** [`augmentation_comparison.csv`](file:///d:/PORA/6th%20Semester/CSE%204622%20ML%20Lab/22%20Batch%20Files/ML%20project/pangas-cnn/analysis/augmentation_comparison.csv)
- **Comparison Figure:** [`augmentation_comparison.png`](file:///d:/PORA/6th%20Semester/CSE%204622%20ML%20Lab/22%20Batch%20Files/ML%20project/pangas-cnn/analysis/augmentation_comparison.png)
