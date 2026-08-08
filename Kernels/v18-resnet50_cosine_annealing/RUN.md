# Run v18-resnet50_cosine_anneal

## Purpose
Train ResNet-50 using CosineAnnealingLR scheduler instead of ReduceLROnPlateau.
ReduceLROnPlateau collapsed the backbone learning rate too early in stage 2 training. CosineAnnealingLR allows continuous learning rate adjustment without collapsing to 1e-8.

## Settings
- Model: ResNet-50
- Epochs: 25 (head) + 75 (fine-tune)
- Sampler: WeightedRandomSampler
- Augmentation: baseline ("current")
- Scheduler: Stage 1 ReduceLROnPlateau, Stage 2 CosineAnnealingLR

## Status
Queued on Kaggle.
