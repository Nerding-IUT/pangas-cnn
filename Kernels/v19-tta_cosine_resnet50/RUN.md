# Run v19-tta_cosine_resnet50

## Purpose
Evaluate ResNet-50 weights trained with CosineAnnealingLR (from v18) using Test-Time Augmentation (TTA).
By combining training-time schedule optimization and test-time spatial voting, we seek the maximum possible performance from the ResNet-50 architecture.

## Settings
- Model: ResNet-50 (CosineAnnealingLR weights from v18)
- TTA: 8 views

## Status
Queued on Kaggle.
