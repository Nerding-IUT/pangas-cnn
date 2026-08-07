"""
smoke_newmodels.py — v7 integration smoke for SwinV2-B, InceptionV3, VGG19.

NOT a training run. Purpose: catch every known integration trap in ~10 minutes
rather than discovering them hours into a full training run.

What it does for each new model:
  1. Build and print param count
  2. Single forward + backward pass (verifies no crash, correct output shape)
  3. Single Grad-CAM (verifies target layer hook fires, heatmap is non-degenerate)
  4. Print diagnostics specific to each architecture's known traps

Known traps checked:
  - InceptionV3: aux_logits must be False (else forward() returns namedtuple,
    breaking outputs.max(1) in train_one_epoch)
  - VGG19: Grad-CAM target must be features[-2] (ReLU, 14x14), NOT features[-1]
    (MaxPool, 7x7)
  - Swin: channels_last=True path has never executed — verify heatmap is non-zero
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2

# ── pip install (Kaggle already has these, but belt-and-suspenders) ───────────
os.system(
    "pip install -q "
    '"torch>=2.0.0" "torchvision>=0.15.0" '
    '"datasets>=2.14.0" "numpy>=1.24.0" "opencv-python-headless>=4.7.0"'
)

from torchvision.models import (
    VGG19_Weights,
    Inception_V3_Weights,
    Swin_V2_B_Weights,
)

NUM_CLASSES = 20
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

os.makedirs("smoke_outputs", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# GradCAM (identical to notebook.ipynb cell 23)
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, target_layer, channels_last=False):
        self.model = model
        self.target_layer = target_layer
        self.channels_last = channels_last
        self.gradients = self.activations = None
        # Disable inplace activations so PyTorch autograd hooks don't error on view modification
        for m in self.model.modules():
            if hasattr(m, "inplace"):
                m.inplace = False
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, m, inp, out): self.activations = out.detach()
    def _bwd_hook(self, m, gin, gout): self.gradients = gout[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        with torch.enable_grad():
            inp = input_tensor.unsqueeze(0).requires_grad_(True)
            output = self.model(inp)
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
        h, w = input_tensor.shape[1:]
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        return cam, class_idx


# ─────────────────────────────────────────────────────────────────────────────
# Smoke routine
# ─────────────────────────────────────────────────────────────────────────────

def run_smoke(name, model, target_layer_fn, channels_last, classifier_attr):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    # ── Forward pass ─────────────────────────────────────────────────────────
    dummy = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    out = model(dummy)
    assert out.shape == (2, NUM_CLASSES), (
        f"FAIL: expected output (2, {NUM_CLASSES}), got {out.shape}"
    )
    print(f"  Forward pass: OK  output shape {out.shape}")

    # ── Backward pass ────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    labels = torch.zeros(2, dtype=torch.long).to(DEVICE)
    model.train()
    loss = criterion(model(dummy), labels)
    loss.backward()
    print(f"  Backward pass: OK  loss={loss.item():.4f}")

    # ── Grad-CAM ─────────────────────────────────────────────────────────────
    target_layer = target_layer_fn(model)
    layer_type = type(target_layer).__name__
    print(f"  CAM target layer: {target_layer.__class__.__name__}  (channels_last={channels_last})")

    gradcam = GradCAM(model, target_layer, channels_last=channels_last)
    single = torch.randn(3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    cam, pred_class = gradcam.generate(single)
    cam_h, cam_w = cam.shape
    cam_min, cam_max = float(cam.min()), float(cam.max())
    print(f"  CAM shape: {cam_h}x{cam_w}  range=[{cam_min:.4f}, {cam_max:.4f}]")

    # Non-degenerate check
    assert cam_max > 0.01, f"FAIL: CAM is near-zero (max={cam_max:.6f}) — check target layer"
    assert cam_max > cam_min, "FAIL: CAM is uniform (all same value)"
    print(f"  CAM non-degenerate: OK")

    # ── Architecture-specific checks ─────────────────────────────────────────
    if name == "inceptionv3":
        assert not model.aux_logits, "FAIL: aux_logits is True — will break train_one_epoch"
        assert model.AuxLogits is None, "FAIL: AuxLogits is not None"
        print(f"  InceptionV3 aux_logits=False: OK")
        if cam_h < 6:
            print(f"  NOTE: CAM is {cam_h}x{cam_w} at 224px (expected; documented caveat).")
            print( "        Fallback layer is model.Mixed_6e if 5x5 is too coarse for figures.")

    if name == "vgg19":
        assert layer_type == "ReLU", (
            f"FAIL: VGG19 CAM target should be ReLU (14x14), got {layer_type}"
        )
        print(f"  VGG19 target is ReLU (not MaxPool): OK")

    if name == "swin":
        assert channels_last, "FAIL: Swin should have channels_last=True"
        print(f"  Swin channels_last path: OK")

    print(f"  [{name}] ALL CHECKS PASSED")
    return {"name": name, "n_params": n_params, "cam_h": cam_h, "cam_w": cam_w,
            "cam_max": cam_max, "status": "OK"}


# ─────────────────────────────────────────────────────────────────────────────
# Build and smoke each model
# ─────────────────────────────────────────────────────────────────────────────

results = []

# 1. VGG19
model_vgg = torchvision.models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
model_vgg.classifier[-1] = nn.Linear(model_vgg.classifier[-1].in_features, NUM_CLASSES)
nn.init.kaiming_normal_(model_vgg.classifier[-1].weight, mode="fan_out", nonlinearity="relu")
nn.init.zeros_(model_vgg.classifier[-1].bias)
results.append(run_smoke(
    "vgg19", model_vgg,
    target_layer_fn=lambda m: m.features[-2],
    channels_last=False,
    classifier_attr="classifier",
))

# 2. InceptionV3
model_inc = torchvision.models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
model_inc.aux_logits = False
model_inc.AuxLogits = None
model_inc.fc = nn.Linear(model_inc.fc.in_features, NUM_CLASSES)
nn.init.kaiming_normal_(model_inc.fc.weight, mode="fan_out", nonlinearity="relu")
nn.init.zeros_(model_inc.fc.bias)
results.append(run_smoke(
    "inceptionv3", model_inc,
    target_layer_fn=lambda m: m.Mixed_7c,
    channels_last=False,
    classifier_attr="fc",
))

# 3. SwinV2-B
model_swin = torchvision.models.swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
model_swin.head = nn.Linear(model_swin.head.in_features, NUM_CLASSES)
nn.init.kaiming_normal_(model_swin.head.weight, mode="fan_out", nonlinearity="relu")
nn.init.zeros_(model_swin.head.bias)
results.append(run_smoke(
    "swin", model_swin,
    target_layer_fn=lambda m: m.features[-1],
    channels_last=True,
    classifier_attr="head",
))

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("SMOKE SUMMARY")
print("="*60)
all_ok = True
for r in results:
    status = r["status"]
    if status != "OK":
        all_ok = False
    print(f"  {r['name']:15} params={r['n_params']:>12,}  "
          f"CAM={r['cam_h']}x{r['cam_w']}  status={status}")

print()
if all_ok:
    print("ALL MODELS PASSED — safe to push v8/v9/v10 full training runs")
else:
    print("SOME MODELS FAILED — fix before pushing full runs")
    sys.exit(1)
