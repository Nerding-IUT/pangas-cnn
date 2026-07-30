from models import convnext, resnet50, swin


def _entry(module):
    return {
        "build": module.build_model,
        "freeze": module.freeze_backbone,
        "unfreeze": module.unfreeze_all,
        "classifier_attr": module.CLASSIFIER_ATTR,
        "gradcam_layer": module.get_gradcam_layer,
        "gradcam_channels_last": module.GRADCAM_CHANNELS_LAST,
    }


MODEL_REGISTRY = {
    "resnet50": _entry(resnet50),
    "convnext": _entry(convnext),
    "swin": _entry(swin),
}
