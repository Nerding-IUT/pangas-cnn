import torch.nn as nn
import torchvision
from torchvision.models import Inception_V3_Weights
from config import NUM_CLASSES

CLASSIFIER_ATTR = "fc"
# InceptionV3 stages emit standard (N, C, H, W).
GRADCAM_CHANNELS_LAST = False


def get_gradcam_layer(model):
    # Mixed_7c is the last inception block before global average pooling.
    # At 224×224 input this yields a 5×5 feature map (vs 8×8 at the native
    # 299×299). The coarser resolution is a documented caveat; 224 is kept
    # for cross-model comparability. If the 5×5 CAM is unusable in the XAI
    # harness, fall back to Mixed_6e (12×12 at 224px).
    return model.Mixed_7c


def build_model(num_classes=NUM_CLASSES):
    model = torchvision.models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    # ⚠️  aux_logits=True (the pretrained default) makes forward() return an
    # InceptionOutputs namedtuple in train mode, breaking `outputs.max(1)` in
    # train_one_epoch. Disable immediately — the pretrained AuxLogits head is
    # irrelevant to fine-tuning on AQUA20.
    model.aux_logits = False
    model.AuxLogits = None
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    nn.init.kaiming_normal_(model.fc.weight, mode="fan_out", nonlinearity="relu")
    nn.init.zeros_(model.fc.bias)
    return model


def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
