import torch
import torch.nn as nn
import torchvision
from torchvision.models import Swin_V2_B_Weights
from config import NUM_CLASSES

CLASSIFIER_ATTR = "head"
# Swin stages emit (N, H, W, C), not (N, C, H, W).
GRADCAM_CHANNELS_LAST = True


def get_gradcam_layer(model):
    # The only Conv2d in swin is the patch-embed stem, so an automatic
    # "last conv" search would target the very first layer. Use the final
    # stage output instead.
    return model.features[-1]


def build_model(num_classes=NUM_CLASSES):
    model = torchvision.models.swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    nn.init.kaiming_normal_(model.head.weight, mode="fan_out", nonlinearity="relu")
    nn.init.zeros_(model.head.bias)
    return model


def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
