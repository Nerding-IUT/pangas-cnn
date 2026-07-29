import torch
import os

NUM_CLASSES = 20
BATCH_SIZE = 32
EPOCHS_STAGE1 = 25
EPOCHS_STAGE2 = 75
LR_HEAD = 1e-3
LR_BACKBONE = 1e-5
MODEL_NAME = "resnet50"  # resnet50 | convnext | swin
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_NAME = "AQUA20"
CHECKPOINT_PATH = "best_model.pth"
import multiprocessing as _mp
try:
    _mp.set_start_method("fork")
except RuntimeError:
    pass
NUM_WORKERS = os.cpu_count()
