from models import convnext, convnext_base, resnet50, swin, vgg19, inceptionv3


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
    "resnet50":      _entry(resnet50),
    "convnext":      _entry(convnext),
    "convnext_base": _entry(convnext_base),  # was missing — scripts path couldn't build v4
    "swin":          _entry(swin),
    "vgg19":         _entry(vgg19),
    "inceptionv3":   _entry(inceptionv3),
}
