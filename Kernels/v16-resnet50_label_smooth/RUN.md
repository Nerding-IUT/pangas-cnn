# Run v16-resnet50_label_smooth

## Purpose
Train ResNet-50 using label smoothing (0.1) on the baseline data augmentation policy ("current"). 
This experiment aims to soften target distributions, addressing boundary ambiguity (e.g. fish vs fishInGroups) and confusion pairs (e.g. coral vs seaAnemone) without changing the network capacity.

## Settings
- Model: ResNet-50
- Epcohs: 25 (head) + 75 (fine-tune)
- Sampler: WeightedRandomSampler
- Augmentation: baseline ("current")
- Loss: CrossEntropyLoss with `label_smoothing=0.1`

## Status
Queued on Kaggle.
