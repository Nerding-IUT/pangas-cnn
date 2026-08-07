# xai.py — GradCAM + LIME + SHAP over the frozen shared manifest.
#
# One script, one MODEL_NAME. Every model in the roster is explained on the
# SAME images (from v5-manifest), which is what makes the cross-model
# comparison in direction #3 valid.
#
# Deliberately a .py script, not a notebook: KAGGLE_RUN_GUIDE.md §2 prefers
# scripts, and CLAUDE.md §8 records that editing .ipynb by cell index has
# already silently written code into a markdown cell in this project. This
# kernel has no interactive value, so there is no reason to accept that risk.
#
# Inputs  (all via kernel_sources):
#   <CHECKPOINT_KERNEL>/  weights_<model>.pth
#   aqua20-v5-manifest/   xai_manifest.csv
# Outputs (/kaggle/working IS the output dir — never copy into a subfolder):
#   metrics/       <- small, downloads first
#   figures/
#   saliency/

import os
# Reduces CUDA memory fragmentation; must be set before torch initialises CUDA.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import glob
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import spearmanr
from torchvision import transforms
from torchvision.transforms import functional as TF
from datasets import load_dataset, load_dataset_builder

# ─────────────────────────── CONFIG ───────────────────────────
MODEL_NAME = "vgg19"          # resnet50 | convnext_base | swin | inceptionv3 | vgg19
CHECKPOINT_KERNEL = "aqua20-v10-full-vgg19"
MANIFEST_KERNEL = "aqua20-v5-manifest"

SMOKE = False                     # True -> only N_SMOKE manifest rows
N_SMOKE = 5

LIME_SAMPLES = 1000              # perturbations per image
SHAP_NSAMPLES = 200              # expected-gradient samples per image
SHAP_BACKGROUND = 20             # reference images drawn from train
FAITH_STEPS = 50                 # deletion/insertion curve resolution
TOPK_FRAC = 0.10                 # for concentration + IoU

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

METRICS_DIR, FIG_DIR, SAL_DIR = "metrics", "figures", "saliency"
# ──────────────────────────────────────────────────────────────

random_state = np.random.RandomState(SEED)
torch.manual_seed(SEED)

# np.trapz was renamed np.trapezoid in numpy 2.0. The Kaggle image's version is
# not guaranteed, so bind whichever exists rather than assuming.
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ══════════════════ model builders ══════════════════
# Mirrors notebook.ipynb cell 8 so the two stay recognisably the same code.

def build_resnet50(n):
    m = torchvision.models.resnet50()
    m.fc = nn.Linear(m.fc.in_features, n)
    return m, (lambda x: x.layer4[-1]), False


def build_convnext_base(n):
    m = torchvision.models.convnext_base()
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n)
    return m, (lambda x: x.features[-1]), False


def build_swin(n):
    m = torchvision.models.swin_v2_b()
    m.head = nn.Linear(m.head.in_features, n)
    # Swin stages emit (N, H, W, C), not (N, C, H, W) -> channels_last=True.
    return m, (lambda x: x.features[-1]), True


def build_inceptionv3(n):
    # aux_logits must be True to load the pretrained-shaped state dict, but the
    # aux head makes forward() return a namedtuple in train mode. Disabling it
    # after construction gives a plain tensor everywhere.
    m = torchvision.models.inception_v3(init_weights=False)
    m.fc = nn.Linear(m.fc.in_features, n)
    m.aux_logits = False
    m.AuxLogits = None
    # At 224 (not its native 299) the last block emits 5x5 — a coarse CAM.
    # Kept at 224 for cross-model comparability; recorded as a caveat.
    return m, (lambda x: x.Mixed_7c), False


def build_vgg19(n):
    m = torchvision.models.vgg19()
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n)
    # features[-1] is a MaxPool (7x7); features[-2] is the last ReLU (14x14),
    # which gives twice the CAM resolution for free.
    return m, (lambda x: x.features[-2]), False


BUILDERS = {
    "resnet50": build_resnet50,
    "convnext_base": build_convnext_base,
    "swin": build_swin,
    "inceptionv3": build_inceptionv3,
    "vgg19": build_vgg19,
}


# ══════════════════ saliency methods ══════════════════

class GradCAM:
    """Ported from notebook.ipynb cell 23, which is the version proven on v2/v4."""

    def __init__(self, model, target_layer, channels_last=False):
        self.model, self.channels_last = model, channels_last
        self.gradients = self.activations = None
        # Disable inplace activations so PyTorch autograd hooks don't error on view modification
        for m in self.model.modules():
            if hasattr(m, "inplace"):
                m.inplace = False
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.activations = o.detach()

    def _bwd(self, m, gi, go):
        self.gradients = go[0].detach()

    def __call__(self, x, class_idx):
        self.model.eval()
        # Grad-CAM needs the backward pass, so autograd must stay enabled.
        with torch.enable_grad():
            x = x.unsqueeze(0).requires_grad_(True)
            out = self.model(x)
            one_hot = torch.zeros_like(out)
            one_hot[0, class_idx] = 1
            self.model.zero_grad()
            out.backward(gradient=one_hot)
        with torch.no_grad():
            g, a = self.gradients, self.activations
            if self.channels_last:  # (N,H,W,C) -> (N,C,H,W)
                g, a = g.permute(0, 3, 1, 2), a.permute(0, 3, 1, 2)
            cam = torch.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1))
        cam = cam.squeeze(0).cpu().numpy()
        return resize_map(cam)


def lime_saliency(model, raw01, class_idx):
    from lime import lime_image

    def classifier_fn(images):
        # LIME hands back unnormalized [0,1] float images; normalization has to
        # happen here, not in the transform pipeline.
        out = []
        for i in range(0, len(images), 32):
            batch = torch.from_numpy(
                np.asarray(images[i:i + 32], dtype=np.float32)
            ).permute(0, 3, 1, 2)
            batch = TF.normalize(batch, IMAGENET_MEAN, IMAGENET_STD).to(DEVICE)
            with torch.no_grad():
                out.append(torch.softmax(model(batch), 1).cpu().numpy())
        return np.concatenate(out)

    explainer = lime_image.LimeImageExplainer(random_state=SEED)
    exp = explainer.explain_instance(
        raw01.astype(np.double), classifier_fn,
        labels=(class_idx,), top_labels=None,
        hide_color=None, num_samples=LIME_SAMPLES,
        random_seed=SEED,
    )
    # If the label key is missing, every segment silently maps to 0.0 and the
    # result is indistinguishable from "LIME found nothing". Fail loudly instead.
    if class_idx not in exp.local_exp:
        raise KeyError(
            f"LIME produced no explanation for class {class_idx}; "
            f"got {sorted(exp.local_exp)}"
        )
    weights = dict(exp.local_exp[class_idx])
    sal = np.vectorize(lambda s: weights.get(s, 0.0))(exp.segments)
    if not np.any(sal):
        print("    !! LIME map is all zeros (no segment carried weight)")
    return sal.astype(np.float32)


def _to_numpy(v):
    """shap returns torch tensors on the model's device when run on GPU.

    np.asarray() on a CUDA tensor raises rather than transferring, and a
    CPU-only local test never hits it — this cost one Kaggle push cycle.
    """
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    return np.asarray(v)


def shap_saliency(explainer, x, class_idx):
    # ranked_outputs=1 explains the top-scoring class, which is the prediction
    # we are already analysing — 20x cheaper than explaining every class.
    sv, idx = explainer.shap_values(
        x.unsqueeze(0), nsamples=SHAP_NSAMPLES, ranked_outputs=1
    )
    arr = _to_numpy(sv[0] if isinstance(sv, list) else sv)
    while arr.ndim > 3:  # drop batch and/or trailing output axis
        arr = arr[0] if arr.shape[0] == 1 else arr[..., 0]
    if arr.shape[0] == 3:  # (C,H,W) -> (H,W)
        arr = arr.sum(0)
    explained = int(_to_numpy(idx).ravel()[0])
    return arr.astype(np.float32), explained


# ══════════════════ shared post-processing ══════════════════

def resize_map(m):
    """Bring any saliency map to IMG_SIZE x IMG_SIZE."""
    if m.shape != (IMG_SIZE, IMG_SIZE):
        m = np.asarray(
            Image.fromarray(m.astype(np.float32), mode="F").resize(
                (IMG_SIZE, IMG_SIZE), Image.BILINEAR
            )
        )
    return m.astype(np.float32)


def normalize_saliency(m):
    """ReLU then min-max to [0,1].

    GradCAM is already non-negative; LIME and SHAP are signed. Making the three
    comparable means discarding negative evidence, which is a real loss — the
    signed maps are saved separately so nothing is thrown away.
    """
    m = resize_map(np.nan_to_num(m))
    m = np.maximum(m, 0)
    rng = m.max() - m.min()
    return (m - m.min()) / rng if rng > 0 else np.zeros_like(m)


# ══════════════════ faithfulness ══════════════════

def faithfulness(model, x, sal, class_idx, blurred):
    """Deletion and insertion AUC (Petsiuk et al. 2018).

    Deletion: replace the most-salient pixels with a blurred version and watch
    p(class) fall — a *lower* AUC means the map found what mattered.
    Insertion: the reverse, starting fully blurred — *higher* is better.

    A Gaussian-blur baseline is used rather than black, which would be far out
    of distribution for underwater imagery and would inflate both scores.
    """
    # LIME assigns one identical value to every pixel of a superpixel, so a
    # plain argsort breaks those ties in raster order and deletion eats each
    # segment top-left-first instead of by importance. Measured locally: that
    # alone made LIME score *worse than random* on deletion. Seeded jitter
    # (far below the saliency scale) randomises tie order without perturbing
    # any genuine ranking.
    flat = sal.ravel().astype(np.float64)
    jitter = np.random.RandomState(SEED).rand(flat.size) * 1e-9
    order = np.argsort(-(flat + jitter))
    n_px = order.size
    per_step = max(1, n_px // FAITH_STEPS)

    del_batch, ins_batch = [], []
    for step in range(FAITH_STEPS + 1):
        k = min(step * per_step, n_px)
        mask = torch.zeros(n_px, device=DEVICE)
        if k:
            mask[torch.from_numpy(order[:k].copy()).to(DEVICE)] = 1.0
        mask = mask.view(1, IMG_SIZE, IMG_SIZE)
        del_batch.append(x * (1 - mask) + blurred * mask)
        ins_batch.append(blurred * (1 - mask) + x * mask)

    curves = []
    for batch in (del_batch, ins_batch):
        probs = []
        stacked = torch.stack(batch)
        with torch.no_grad():
            for i in range(0, len(stacked), 32):
                p = torch.softmax(model(stacked[i:i + 32]), 1)[:, class_idx]
                probs.append(p.cpu().numpy())
        curves.append(np.concatenate(probs))

    xs = np.linspace(0, 1, FAITH_STEPS + 1)
    # curves[1][0] is p(class) on the fully-blurred image. If the blur does not
    # actually destroy the prediction, insertion starts near its ceiling and
    # has no headroom to discriminate — worth recording, not assuming.
    return (float(_trapz(curves[0], xs)), float(_trapz(curves[1], xs)),
            float(curves[1][0]))


def concentration(sal):
    """Share of total saliency mass inside the top 10% of pixels.

    A proxy for how *focused* attention is. NOT a measure of whether it is
    focused on the right thing — AQUA20 ships no segmentation masks, so
    attention-on-animal cannot be computed. See FINDINGS.md.
    """
    flat = np.sort(sal.ravel())[::-1]
    total = flat.sum()
    if total <= 0:
        return 0.0
    return float(flat[: max(1, int(TOPK_FRAC * flat.size))].sum() / total)


def agreement(a, b):
    rho, _ = spearmanr(a.ravel(), b.ravel())  # tuple form works on every scipy
    k = max(1, int(TOPK_FRAC * a.size))
    ma = np.zeros(a.size, bool); ma[np.argsort(-a.ravel())[:k]] = True
    mb = np.zeros(b.size, bool); mb[np.argsort(-b.ravel())[:k]] = True
    union = (ma | mb).sum()
    return (0.0 if np.isnan(rho) else float(rho)), float((ma & mb).sum() / union)


# ══════════════════ io helpers ══════════════════

def find_one(slug, filename, what):
    """Locate a file inside a mounted kernel output.

    kernel_sources does NOT mount at /kaggle/input/<slug>/ as you would expect
    — v5-manifest proved it lands at /kaggle/input/notebooks/<user>/<slug>/.
    Both layouts are tried rather than hard-coding either.
    """
    for pattern in (
        f"/kaggle/input/**/{slug}/**/{filename}",
        f"/kaggle/input/{slug}/**/{filename}",
    ):
        hits = glob.glob(pattern, recursive=True)
        if hits:
            print(f"  {what}: {hits[0]}")
            return hits[0]
    raise FileNotFoundError(f"{what} ({filename}) not found under any {slug} mount")


def main():
    t_start = time.time()
    for d in (METRICS_DIR, FIG_DIR, SAL_DIR):
        os.makedirs(d, exist_ok=True)

    print("=" * 68)
    print(f"XAI — {MODEL_NAME}   device={DEVICE}   smoke={SMOKE}")
    print("=" * 68)
    for entry in sorted(os.listdir("/kaggle/input")):
        print(f"  mounted: {entry}/")

    ckpt_path = find_one(CHECKPOINT_KERNEL, "weights_*.pth", "checkpoint")
    manifest_path = find_one(MANIFEST_KERNEL, "xai_manifest.csv", "manifest")

    manifest = pd.read_csv(manifest_path)
    if SMOKE:
        # The manifest is sorted by test_index, so head(N) is NOT a sample —
        # the first three rows are all coral. Coral fills the whole frame, so
        # restoring any random pixels recovers the prediction and random's
        # insertion score is inflated, which made the sanity check misfire.
        # Take one image per stratum instead, which also spreads the classes.
        manifest = (manifest.groupby("stratum", sort=False)
                    .head(1).head(N_SMOKE).reset_index(drop=True))
    print(f"  explaining {len(manifest)} images\n")

    names = load_dataset_builder("taufiktrf/AQUA20").info.features["label"].names
    ds = load_dataset("taufiktrf/AQUA20")
    test_split, train_split = ds["test"], ds["train"]

    # Two transforms sharing geometry: LIME and the figures need the
    # unnormalized image, the model needs the normalized one.
    geom = transforms.Compose([
        transforms.Resize(IMG_SIZE + 32),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
    ])
    to_raw = lambda img: geom(img.convert("RGB"))
    to_net = lambda raw: TF.normalize(raw, IMAGENET_MEAN, IMAGENET_STD)

    model, target_fn, channels_last = BUILDERS[MODEL_NAME](len(names))
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(DEVICE).eval()
    print(f"Loaded {MODEL_NAME}: {sum(p.numel() for p in model.parameters()):,} params")

    target_layer = target_fn(model)
    print(f"Grad-CAM target: {type(target_layer).__name__} "
          f"(channels_last={channels_last})")
    gradcam = GradCAM(model, target_layer, channels_last)

    import shap
    bg_idx = random_state.choice(len(train_split), SHAP_BACKGROUND, replace=False)
    background = torch.stack(
        [to_net(to_raw(train_split[int(i)]["image"])) for i in bg_idx]
    ).to(DEVICE)
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    # Some large models (e.g. Swin) consume almost all VRAM, leaving no room
    # for the GradientExplainer init forward pass.  In that case we fall back
    # to zero SHAP maps so GradCAM + LIME can still complete.
    try:
        shap_explainer = shap.GradientExplainer(model, background)
        print(f"SHAP background: {tuple(background.shape)}\n")
    except (torch.cuda.OutOfMemoryError, RuntimeError) as _shap_err:
        print(f"WARNING: SHAP init failed ({_shap_err.__class__.__name__}): "
              f"will use zero maps for this model.")
        shap_explainer = None
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    faith_rows, agree_rows, pred_rows = [], [], []

    for n, row in enumerate(manifest.itertuples(), 1):
        idx = int(row.test_index)
        raw = to_raw(test_split[idx]["image"])
        raw01 = raw.permute(1, 2, 0).numpy()
        x = to_net(raw).to(DEVICE)

        with torch.no_grad():
            probs = torch.softmax(model(x.unsqueeze(0)), 1)[0].cpu().numpy()
        pred = int(probs.argmax())
        true = int(row.class_idx)
        rank = int(np.where(np.argsort(-probs) == true)[0][0]) + 1
        print(f"[{n}/{len(manifest)}] #{idx} {names[true]} -> {names[pred]} "
              f"({probs[pred]:.3f})", flush=True)

        maps_signed = {}
        t0 = time.time()
        maps_signed["gradcam"] = gradcam(x, pred)
        t_gc = time.time() - t0

        t0 = time.time()
        maps_signed["lime"] = lime_saliency(model, raw01, pred)
        t_lime = time.time() - t0

        # Flush fragmented CUDA memory left by LIME before SHAP gradient pass.
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        t0 = time.time()
        if shap_explainer is not None:
            # shap_values() can OOM even after a successful init (e.g. Swin)
            # because LIME fills VRAM.  Fall back to zeros per image.
            try:
                sv, explained = shap_saliency(shap_explainer, x, pred)
                if explained != pred:
                    print(f"    !! SHAP explained class {explained}, not {pred}")
            except (torch.cuda.OutOfMemoryError, RuntimeError) as _e:
                print(f"    !! SHAP OOM on image {idx} ({_e.__class__.__name__}): zero map")
                sv = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
                explained = pred
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
        else:
            sv = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
            explained = pred
        maps_signed["shap"] = sv
        t_shap = time.time() - t0

        # A random map is carried as a fourth "method" so the sanity check
        # (a real method must beat noise) is permanent, not a one-off.
        maps_signed["random"] = random_state.rand(IMG_SIZE, IMG_SIZE).astype(np.float32)

        maps = {k: normalize_saliency(v) for k, v in maps_signed.items()}
        print(f"    gradcam {t_gc:.1f}s | lime {t_lime:.1f}s | shap {t_shap:.1f}s")

        blurred = TF.gaussian_blur(x.unsqueeze(0), 51, 11.0).squeeze(0)
        for method, sal in maps.items():
            d_auc, i_auc, p_blur = faithfulness(model, x, sal, pred, blurred)
            faith_rows.append({
                "test_index": idx, "model": MODEL_NAME, "method": method,
                "deletion_auc": round(d_auc, 5), "insertion_auc": round(i_auc, 5),
                "concentration": round(concentration(sal), 5),
                "p_blurred": round(p_blur, 5),
                "correct": pred == true, "stratum": row.stratum,
            })
            np.save(f"{SAL_DIR}/{method}_{idx:04d}.npy", maps_signed[method])

        for a, b in (("gradcam", "lime"), ("gradcam", "shap"), ("lime", "shap"),
                     ("gradcam", "random")):
            rho, iou = agreement(maps[a], maps[b])
            agree_rows.append({
                "test_index": idx, "model": MODEL_NAME, "method_a": a, "method_b": b,
                "spearman": round(rho, 5), "iou_top10": round(iou, 5),
            })

        pred_rows.append({
            "test_index": idx, "model": MODEL_NAME, "true": names[true],
            "pred": names[pred], "correct": pred == true,
            "confidence": round(float(probs[pred]), 5),
            "true_prob": round(float(probs[true]), 5), "true_rank": rank,
            "stratum": row.stratum,
        })

        fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
        axes[0].imshow(raw01)
        axes[0].set_title(f"True: {names[true]}")
        for ax, key in zip(axes[1:], ("gradcam", "lime", "shap")):
            ax.imshow(raw01)
            ax.imshow(maps[key], cmap="jet", alpha=0.5, vmin=0, vmax=1)
            ax.set_title(key.upper())
        for ax in axes:
            ax.axis("off")
        mark = "" if pred == true else "  ✗"
        fig.suptitle(f"#{idx}  {MODEL_NAME}  pred: {names[pred]} "
                     f"({probs[pred]:.2f}){mark}   [{row.stratum}]")
        plt.tight_layout()
        tag = "ok" if pred == true else "err"
        plt.savefig(f"{FIG_DIR}/{tag}_{idx:04d}_{names[true]}-as-{names[pred]}.png", dpi=130)
        plt.close(fig)

    pd.DataFrame(faith_rows).to_csv(f"{METRICS_DIR}/faithfulness.csv", index=False)
    pd.DataFrame(agree_rows).to_csv(f"{METRICS_DIR}/agreement.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(f"{METRICS_DIR}/predictions.csv", index=False)

    f = pd.DataFrame(faith_rows)
    summary = f.groupby("method")[
        ["deletion_auc", "insertion_auc", "concentration", "p_blurred"]
    ].mean().round(4)
    print("\n" + "=" * 68)
    print(f"SUMMARY — {MODEL_NAME}   (deletion lower=better, insertion higher=better)")
    print("=" * 68)
    print(summary.to_string())
    print("\nAgreement (mean):")
    print(pd.DataFrame(agree_rows).groupby(["method_a", "method_b"])[
        ["spearman", "iou_top10"]].mean().round(4).to_string())

    # The check that decides whether any of the above means anything.
    print("\n" + "-" * 68)
    ok_checks = []
    for m in ("gradcam", "lime", "shap"):
        if m not in summary.index or "random" not in summary.index:
            continue
        beat_del = summary.loc[m, "deletion_auc"] < summary.loc["random", "deletion_auc"]
        beat_ins = summary.loc[m, "insertion_auc"] > summary.loc["random", "insertion_auc"]
        ok_checks.append(beat_del and beat_ins)
        print(f"  {m:8} beats random:  deletion {'OK' if beat_del else 'FAIL'} | "
              f"insertion {'OK' if beat_ins else 'FAIL'}")
    print("  => " + ("sanity check PASSED" if all(ok_checks) and ok_checks
                     else "SANITY CHECK FAILED — metrics are suspect"))
    print("-" * 68)

    json.dump({
        "model": MODEL_NAME, "smoke": SMOKE, "n_images": int(len(manifest)),
        "checkpoint": ckpt_path, "manifest": manifest_path,
        "lime_samples": LIME_SAMPLES, "shap_nsamples": SHAP_NSAMPLES,
        "runtime_min": round((time.time() - t_start) / 60, 2),
        "sanity_check_passed": bool(ok_checks and all(ok_checks)),
    }, open(f"{METRICS_DIR}/run_info.json", "w"), indent=2)
    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
