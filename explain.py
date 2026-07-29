import os
import multiprocessing as _mp

try:
    _mp.set_start_method("fork")
except RuntimeError:
    pass

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset_builder
from torchvision import transforms
from torch.utils.data import DataLoader

from config import DEVICE, CHECKPOINT_PATH, MODEL_NAME, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, NUM_WORKERS
from models import MODEL_REGISTRY


def find_last_conv(model):
    last_conv = None
    last_conv_name = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
            last_conv_name = name
    if last_conv is None:
        raise ValueError("No Conv2d layer found in model")
    return last_conv_name, last_conv


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    @torch.no_grad()
    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor.unsqueeze(0))
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.clamp(cam, min=0)
        cam = cam.squeeze().cpu().numpy()
        h, w = input_tensor.shape[1:]
        cam = np.maximum(cam, 0)
        cam = (
            (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            if cam.max() > cam.min()
            else cam
        )
        import cv2

        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        return cam, class_idx


def unnormalize(tensor, mean, std):
    img = tensor.cpu().clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img.permute(1, 2, 0).numpy()


def overlay_heatmap(img, cam, alpha=0.5):
    import cv2

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * 255 * img).astype(np.uint8)
    return overlay


def get_class_names():
    builder = load_dataset_builder("taufiktrf/AQUA20")
    return builder.info.features["label"].names


def main():
    class_names = get_class_names()
    print(f"Classes ({len(class_names)}): {class_names}\n")

    model = MODEL_REGISTRY[MODEL_NAME]["build"]().to(DEVICE)
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Find the last convolutional layer automatically
    last_conv_name, last_conv = find_last_conv(model)
    print(f"Target layer: {last_conv_name} ({type(last_conv).__name__})")

    from datasets import load_dataset

    dataset = load_dataset("taufiktrf/AQUA20")
    test_split = dataset["test"]

    val_transform = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE + 32),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    def transform_fn(examples):
        examples["image"] = [
            val_transform(img.convert("RGB")) for img in examples["image"]
        ]
        return examples

    test_split.set_transform(transform_fn)
    test_loader = DataLoader(
        test_split, batch_size=1, shuffle=False, num_workers=min(4, NUM_WORKERS or 0)
    )

    gradcam = GradCAM(model, last_conv)

    os.makedirs("outputs/gradcam", exist_ok=True)
    num_examples = min(40, len(test_split))

    for i, batch in enumerate(test_loader):
        if i >= num_examples:
            break
        img_tensor = batch["image"].to(DEVICE)
        label = batch["label"].item()

        cam, pred = gradcam.generate(img_tensor.squeeze(0))
        img_np = unnormalize(img_tensor.squeeze(0), IMAGENET_MEAN, IMAGENET_STD)
        img_np = np.clip(img_np, 0, 1)

        overlay = overlay_heatmap(img_np, cam)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_np)
        axes[0].set_title(f"True: {class_names[label]}")
        axes[0].axis("off")

        axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(f"Pred: {class_names[pred]}")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(
            f"outputs/gradcam/{i:04d}_{class_names[label]}_pred_{class_names[pred]}.png",
            dpi=150,
        )
        plt.close(fig)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{num_examples}] Grad-CAM images saved")

    print(f"\nSaved {num_examples} Grad-CAM visualizations to outputs/gradcam/")


if __name__ == "__main__":
    main()
