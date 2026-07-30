import os

import cv2
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset_builder
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from config import DEVICE, CHECKPOINT_PATH, MODEL_NAME, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, NUM_WORKERS
from models import MODEL_REGISTRY

# How many Grad-CAM figures to produce, split between errors and successes.
# Misclassifications get the larger share — they are what the analysis is about.
N_WRONG_EXAMPLES = 20
N_RIGHT_EXAMPLES = 20


class GradCAM:
    def __init__(self, model, target_layer, channels_last=False):
        self.model = model
        self.target_layer = target_layer
        self.channels_last = channels_last
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

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()

        # Grad-CAM needs the backward pass, so autograd must stay enabled here.
        # requires_grad on the input guarantees a graph even if every parameter
        # is frozen.
        with torch.enable_grad():
            input_tensor = input_tensor.unsqueeze(0).requires_grad_(True)
            output = self.model(input_tensor)
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
            one_hot = torch.zeros_like(output)
            one_hot[0, class_idx] = 1
            self.model.zero_grad()
            output.backward(gradient=one_hot)

        with torch.no_grad():
            grads, acts = self.gradients, self.activations
            if self.channels_last:  # (N, H, W, C) -> (N, C, H, W)
                grads = grads.permute(0, 3, 1, 2)
                acts = acts.permute(0, 3, 1, 2)
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = (weights * acts).sum(dim=1)
            cam = torch.relu(cam).squeeze(0).cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        h, w = input_tensor.shape[2:]
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        return cam, class_idx


def unnormalize(tensor, mean, std):
    img = tensor.cpu().clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img.permute(1, 2, 0).numpy()


def overlay_heatmap(img, cam, alpha=0.5):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * 255 * img).astype(np.uint8)
    return overlay


def get_class_names():
    builder = load_dataset_builder("taufiktrf/AQUA20")
    return builder.info.features["label"].names


@torch.no_grad()
def collect_predictions(model, loader):
    model.eval()
    labels, preds = [], []
    for batch in loader:
        outputs = model(batch["image"].to(DEVICE))
        preds.append(outputs.argmax(dim=1).cpu())
        labels.append(batch["label"])
    return torch.cat(labels).numpy(), torch.cat(preds).numpy()


def select_gradcam_indices(labels, preds, n_wrong, n_right):
    """Pick a class-diverse sample of test indices, weighted toward errors.

    labels/preds come from a shuffle=False loader, so position i in those
    arrays is index i in the test dataset.
    """

    def spread(pool, n):
        # Round-robin over true classes so one dominant class (fish, coral)
        # cannot swallow the whole quota.
        by_class = {}
        for idx in pool:
            by_class.setdefault(int(labels[idx]), []).append(int(idx))
        picked, classes = [], sorted(by_class)
        while len(picked) < n and any(by_class[c] for c in classes):
            for c in classes:
                if by_class[c] and len(picked) < n:
                    picked.append(by_class[c].pop(0))
        return picked

    wrong = spread(np.flatnonzero(labels != preds), n_wrong)
    right = spread(np.flatnonzero(labels == preds), n_right)
    return wrong, right


def main():
    class_names = get_class_names()
    print(f"Classes ({len(class_names)}): {class_names}\n")

    model_cfg = MODEL_REGISTRY[MODEL_NAME]
    model = model_cfg["build"]().to(DEVICE)
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    target_layer = model_cfg["gradcam_layer"](model)
    print(f"Target layer: {type(target_layer).__name__}")

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
    num_workers = min(4, NUM_WORKERS or 0)

    # Score the test set first so the sample can be chosen from the errors.
    scoring_loader = DataLoader(
        test_split, batch_size=32, shuffle=False, num_workers=num_workers
    )
    labels, preds = collect_predictions(model, scoring_loader)

    wrong_idx, right_idx = select_gradcam_indices(
        labels, preds, N_WRONG_EXAMPLES, N_RIGHT_EXAMPLES
    )
    selected = wrong_idx + right_idx
    n_classes = len({int(labels[i]) for i in selected})
    print(
        f"Selected {len(wrong_idx)} misclassified + {len(right_idx)} correct "
        f"images, spanning {n_classes} classes"
    )

    gradcam_loader = DataLoader(
        Subset(test_split, selected), batch_size=1, shuffle=False,
        num_workers=num_workers,
    )

    gradcam = GradCAM(
        model, target_layer, channels_last=model_cfg["gradcam_channels_last"]
    )

    out_dir = f"outputs/gradcam/{MODEL_NAME}"
    os.makedirs(out_dir, exist_ok=True)

    for i, batch in enumerate(gradcam_loader):
        ds_idx = selected[i]
        label, pred = int(labels[ds_idx]), int(preds[ds_idx])
        img_tensor = batch["image"].to(DEVICE)

        # Explain the prediction that was actually recorded, not a fresh argmax.
        cam, _ = gradcam.generate(img_tensor.squeeze(0), class_idx=pred)
        img_np = unnormalize(img_tensor.squeeze(0), IMAGENET_MEAN, IMAGENET_STD)
        img_np = np.clip(img_np, 0, 1)

        overlay = overlay_heatmap(img_np, cam)
        correct = label == pred

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_np)
        axes[0].set_title(f"True: {class_names[label]}")
        axes[0].axis("off")

        axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(f"Pred: {class_names[pred]}" + ("" if correct else "  X"))
        axes[2].axis("off")

        plt.tight_layout()
        tag = "ok" if correct else "wrong"
        plt.savefig(
            f"{out_dir}/{tag}_{i:03d}_"
            f"{class_names[label]}-as-{class_names[pred]}.png",
            dpi=150,
        )
        plt.close(fig)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(selected)}] Grad-CAM images saved")

    print(f"\nSaved {len(selected)} Grad-CAM visualizations to {out_dir}/")
    print(f"  misclassified -> {out_dir}/wrong_*.png")


if __name__ == "__main__":
    main()
