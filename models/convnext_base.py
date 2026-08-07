import torch.nn as nn
import torchvision
from torchvision.models import ConvNeXt_Base_Weights
from config import NUM_CLASSES

CLASSIFIER_ATTR = "classifier"
# ConvNeXt stages emit standard (N, C, H, W).
GRADCAM_CHANNELS_LAST = False


def get_gradcam_layer(model):
    return model.features[-1]


def build_model(num_classes=NUM_CLASSES):
    """ConvNeXt-Base — 87.6M params.

    This is the size the AQUA20 paper reports as its best model (90.69%).
    ConvNeXt-Tiny (28M, models/convnext.py) is a different, smaller model;
    do not report Tiny's numbers as the paper's ConvNeXt result.
    """
    model = torchvision.models.convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
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
