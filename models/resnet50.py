import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights
from config import NUM_CLASSES, DEVICE

CLASSIFIER_ATTR = "fc"
GRADCAM_CHANNELS_LAST = False


def get_gradcam_layer(model):
    return model.layer4[-1]


def build_model(num_classes=NUM_CLASSES):
    model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
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
