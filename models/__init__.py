from models.resnet50 import build_model as build_resnet50
from models.resnet50 import freeze_backbone as freeze_resnet50
from models.resnet50 import unfreeze_all as unfreeze_resnet50
from models.resnet50 import CLASSIFIER_ATTR as RESNET_CLASSIFIER

from models.convnext import build_model as build_convnext
from models.convnext import freeze_backbone as freeze_convnext
from models.convnext import unfreeze_all as unfreeze_convnext
from models.convnext import CLASSIFIER_ATTR as CONVNEXT_CLASSIFIER

from models.swin import build_model as build_swin
from models.swin import freeze_backbone as freeze_swin
from models.swin import unfreeze_all as unfreeze_swin
from models.swin import CLASSIFIER_ATTR as SWIN_CLASSIFIER

MODEL_REGISTRY = {
    "resnet50": {
        "build": build_resnet50,
        "freeze": freeze_resnet50,
        "unfreeze": unfreeze_resnet50,
        "classifier_attr": RESNET_CLASSIFIER,
    },
    "convnext": {
        "build": build_convnext,
        "freeze": freeze_convnext,
        "unfreeze": unfreeze_convnext,
        "classifier_attr": CONVNEXT_CLASSIFIER,
    },
    "swin": {
        "build": build_swin,
        "freeze": freeze_swin,
        "unfreeze": unfreeze_swin,
        "classifier_attr": SWIN_CLASSIFIER,
    },
}
