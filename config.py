import torch
import os
import multiprocessing as _mp

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
CHECKPOINT_PATH = f"best_{MODEL_NAME}.pth"
STATE_PATH = f"training_state_{MODEL_NAME}.json"

try:
    _mp.set_start_method("fork")
    FORK_AVAILABLE = True
except RuntimeError:  # already set by an earlier import
    FORK_AVAILABLE = _mp.get_start_method() == "fork"
except ValueError:  # Windows has no fork
    FORK_AVAILABLE = False

# Without fork, workers use spawn and cannot pickle the lambda transforms in
# data/dataset.py, so stay single-process off Linux.
NUM_WORKERS = os.cpu_count() if FORK_AVAILABLE else 0
