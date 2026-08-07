import torch.nn as nn
import torchvision
from torchvision.models import VGG19_Weights
from config import NUM_CLASSES

CLASSIFIER_ATTR = "classifier"
# VGG stages emit standard (N, C, H, W).
GRADCAM_CHANNELS_LAST = False


def get_gradcam_layer(model):
    # features[-1] is MaxPool2d (output 7×7).
    # features[-2] is the last ReLU before that pool (output 14×14) —
    # higher resolution, and more meaningful gradients than the pool itself.
    return model.features[-2]


def build_model(num_classes=NUM_CLASSES):
    model = torchvision.models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
    # Disable inplace activations so PyTorch autograd hooks (GradCAM) don't error
    # on inplace modification of hooked tensor views.
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = False
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    nn.init.kaiming_normal_(model.classifier[-1].weight, mode="fan_out", nonlinearity="relu")
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
