import torch
import torch.nn as nn
import torchvision
from torchvision.models import ConvNeXt_Tiny_Weights
from config import NUM_CLASSES

CLASSIFIER_ATTR = "classifier"


def build_model(num_classes=NUM_CLASSES):
    model = torchvision.models.convnext_tiny(
        weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    )
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    nn.init.kaiming_normal_(
        model.classifier[-1].weight, mode="fan_out", nonlinearity="relu"
    )
    nn.init.zeros_(model.classifier[-1].bias)
    return model


def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
