# ============================================================
#  v17 — Test-Time Augmentation (TTA) on ResNet-50 v2
#  No training. Loads existing v2 checkpoint, evaluates with TTA.
# ============================================================
import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import (precision_recall_fscore_support,
                             confusion_matrix)
import pandas as pd

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 20
IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_WORKERS = os.cpu_count()
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# v2 checkpoint path (mounted from Kaggle dataset/kernel sources)
import glob
ckpt_hits = glob.glob("/kaggle/input/**/weights_resnet50.pth", recursive=True)
if not ckpt_hits:
    raise FileNotFoundError("Could not find weights_resnet50.pth under /kaggle/input")
CKPT = ckpt_hits[0]


# TTA variants to average over (N=7 for good coverage / reasonable speed)
TTA_TRANSFORMS = [
    # 1. Standard centre-crop (same as val)
    T.Compose([T.Resize(IMG_SIZE + 32), T.CenterCrop(IMG_SIZE),
               T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
    # 2. Horizontal flip
    T.Compose([T.Resize(IMG_SIZE + 32), T.CenterCrop(IMG_SIZE),
               T.RandomHorizontalFlip(p=1.0),
               T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
    # 3. Slightly larger crop → zoom out
    T.Compose([T.Resize(IMG_SIZE + 64), T.CenterCrop(IMG_SIZE),
               T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
    # 4. Zoom out + flip
    T.Compose([T.Resize(IMG_SIZE + 64), T.CenterCrop(IMG_SIZE),
               T.RandomHorizontalFlip(p=1.0),
               T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
    # 5. Five-crop top-left
    T.Compose([T.Resize(IMG_SIZE + 32), T.FiveCrop(IMG_SIZE),
               T.Lambda(lambda crops: crops[0]),  # top-left
               T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
    # 6. Five-crop top-right
    T.Compose([T.Resize(IMG_SIZE + 32), T.FiveCrop(IMG_SIZE),
               T.Lambda(lambda crops: crops[1]),  # top-right
               T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
    # 7. Five-crop bottom-left
    T.Compose([T.Resize(IMG_SIZE + 32), T.FiveCrop(IMG_SIZE),
               T.Lambda(lambda crops: crops[2]),  # bottom-left
               T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
    # 8. Five-crop bottom-right
    T.Compose([T.Resize(IMG_SIZE + 32), T.FiveCrop(IMG_SIZE),
               T.Lambda(lambda crops: crops[3]),  # bottom-right
               T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
]
N_TTA = len(TTA_TRANSFORMS)

# ---- Model ----
def build_resnet50(n_classes):
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m

print(f"Device: {DEVICE}")
print(f"Loading checkpoint: {CKPT}")
model = build_resnet50(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"Loaded ResNet-50: {n_params:,} params")

# ---- Dataset ----
dataset    = load_dataset("taufiktrf/AQUA20")
test_split = dataset["test"]
class_names = dataset["test"].features["label"].names
print(f"Test set: {len(test_split)} samples | {NUM_CLASSES} classes")

# ---- TTA Inference ----
print(f"\nRunning TTA with {N_TTA} views per image...")

def _apply(ex, tfm):
    ex["image"] = [tfm(img.convert("RGB")) for img in ex["image"]]
    return ex

all_probs  = []   # list of (n_samples, n_classes) per TTA view
all_labels = None

t0 = time.time()
for view_idx, tfm in enumerate(TTA_TRANSFORMS):
    test_split.set_transform(lambda ex, t=tfm: _apply(ex, t))
    loader = DataLoader(test_split, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS)
    view_probs = []
    view_labels = []
    with torch.no_grad():
        for batch in loader:
            imgs   = batch["image"].to(DEVICE)
            labels = batch["label"]
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            view_probs.append(probs)
            view_labels.extend(labels.numpy().tolist())
    view_probs = np.concatenate(view_probs, axis=0)
    all_probs.append(view_probs)
    if all_labels is None:
        all_labels = np.array(view_labels)
    elapsed = time.time() - t0
    print(f"  View {view_idx+1}/{N_TTA} done ({elapsed:.1f}s)")

# ---- Aggregate ----
avg_probs  = np.mean(all_probs, axis=0)          # (n_samples, n_classes)
tta_preds  = np.argmax(avg_probs, axis=1)
baseline_preds = np.argmax(all_probs[0], axis=1)  # view 0 = standard centre-crop

def metrics(labels, preds, probs, tag):
    top1  = float(np.mean(labels == preds))
    # top-3
    top3_idx = np.argsort(probs, axis=1)[:, -3:]
    top3  = float(np.mean([labels[i] in top3_idx[i] for i in range(len(labels))]))
    p, r, f, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0)
    _, _, wf, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0)
    n_err = int(np.sum(labels != preds))
    conf_ok  = float(np.max(probs[labels == preds], axis=1).mean()) if (labels==preds).any() else 0
    conf_bad = float(np.max(probs[labels != preds], axis=1).mean()) if (labels!=preds).any() else 0
    print(f"\n=== [{tag}] ===")
    print(f"  Top-1: {top1:.4f} | Top-3: {top3:.4f}")
    print(f"  Macro P: {p:.4f} | R: {r:.4f} | F1: {f:.4f}")
    print(f"  Weighted F1: {wf:.4f}")
    print(f"  Errors: {n_err} / {len(labels)} ({100*n_err/len(labels):.1f}%)")
    print(f"  Mean conf correct: {conf_ok:.4f} | wrong: {conf_bad:.4f}")
    return dict(tag=tag, top1=round(top1,4), top3=round(top3,4),
                macro_p=round(p,4), macro_r=round(r,4), macro_f1=round(f,4),
                weighted_f1=round(wf,4), n_errors=n_err,
                mean_conf_correct=round(conf_ok,4),
                mean_conf_wrong=round(conf_bad,4))

res_baseline = metrics(all_labels, baseline_preds, all_probs[0], "baseline (centre-crop only)")
res_tta      = metrics(all_labels, tta_preds,      avg_probs,    f"TTA ({N_TTA} views)")

delta_top1 = (res_tta["top1"] - res_baseline["top1"]) * 100
delta_f1   = res_tta["macro_f1"] - res_baseline["macro_f1"]
print(f"\n=== DELTA (TTA vs baseline) ===")
print(f"  Top-1: {delta_top1:+.2f} pp | Macro F1: {delta_f1:+.4f}")

# ---- Per-class breakdown ----
p_c, r_c, f_c, sup = precision_recall_fscore_support(
    all_labels, tta_preds, labels=range(NUM_CLASSES), zero_division=0)
per_class = pd.DataFrame([
    {"class": class_names[c], "support": int(sup[c]),
     "precision": round(float(p_c[c]),4),
     "recall":    round(float(r_c[c]),4),
     "f1":        round(float(f_c[c]),4)}
    for c in range(NUM_CLASSES)]).sort_values("f1")
os.makedirs("analysis/tta", exist_ok=True)
per_class.to_csv("analysis/tta/per_class.csv", index=False)
print("\nBottom-5 classes (TTA):")
print(per_class.head(5).to_string(index=False))

# ---- Confusion pairs ----
cmx = confusion_matrix(all_labels, tta_preds)
pairs = [{"true": class_names[t], "predicted": class_names[p],
          "count": int(cmx[t,p]),
          "pct_of_true_class": round(100*cmx[t,p]/max(cmx[t].sum(),1),2)}
         for t in range(NUM_CLASSES) for p in range(NUM_CLASSES)
         if t != p and cmx[t,p] > 0]
pairs_df = pd.DataFrame(pairs).sort_values("count", ascending=False)
pairs_df.to_csv("analysis/tta/confusion_pairs.csv", index=False)
print("\nTop confusion pairs (TTA):")
print(pairs_df.head(8).to_string(index=False))

# ---- Save summary ----
summary = {**res_tta, "n_tta_views": N_TTA,
           "ckpt": CKPT, "delta_top1_pp": round(delta_top1,4),
           "delta_macro_f1": round(delta_f1,4)}
with open("analysis/tta/summary.json","w") as fh:
    json.dump(summary, fh, indent=2)

np.save("analysis/tta/avg_probs.npy", avg_probs.astype(np.float32))
print(f"\nDone in {(time.time()-t0)/60:.1f} min. Results in analysis/tta/")
