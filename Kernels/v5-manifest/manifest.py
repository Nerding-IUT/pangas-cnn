# v5-manifest — build the frozen, shared XAI image manifest.
#
# CPU-only, no model is loaded. This reads the already-computed test-set
# probability matrices from v2 (resnet50) and v4 (convnext_base), mounted as
# kernel_sources, and picks a fixed set of test indices that EVERY xai kernel
# will then explain.
#
# Why this exists: notebook.ipynb picks its Grad-CAM images from *that model's
# own errors*, so no two models get explained on the same pictures and a
# cross-model comparison is impossible. Choosing the images once, here, and
# freezing them is what makes direction #3 valid.
#
# Output: xai_manifest.csv  (+ manifest_stats.json for provenance)

import os
import glob
import json

import numpy as np
import pandas as pd
from datasets import load_dataset, load_dataset_builder

# How many images each stratum contributes. Sums to 39; strata that cannot be
# filled shrink rather than borrow, and the real count is reported at the end.
TARGET = {
    "class_coverage": 20,
    "paper_pair": 8,
    "fish_group": 3,
    "rare_class": 4,
    "challenging": 4,
}

# The confusion pairs the AQUA20 paper calls out (its Figure 6 / discussion).
# Both directions are tried; whichever actually occurs in our models' errors
# gets picked.
PAPER_PAIRS = [
    ("coral", "starfish"),
    ("seaSlug", "flatworm"),
    ("marine_dolphin", "shark"),
    ("fish", "eel"),
    ("coral", "seaAnemone"),
]

# Rare classes from the plan doc (<50 training images). marine_dolphin and
# shrimp are also the two v4 rescued from ~zero probability, so images where v2
# failed and v4 succeeded are preferred — that makes the capacity finding
# visible in the heatmaps.
RARE_CLASSES = ["octopus", "marine_dolphin", "crab", "shrimp"]

SOURCES = {
    "v2": "aqua20-v2-full-resnet50",
    "v4": "aqua20-v4-full-convnext-base",
}


def show_mounted_inputs():
    """Print what kernel_sources actually mounted.

    The whole pipeline depends on this working, so it is verified loudly and
    first rather than assumed.
    """
    print("=" * 68)
    print("MOUNTED INPUTS  (/kaggle/input)")
    print("=" * 68)
    if not os.path.isdir("/kaggle/input"):
        print("  !! /kaggle/input does not exist")
        return
    for entry in sorted(os.listdir("/kaggle/input")):
        print(f"  {entry}/")
        n = 0
        for root, _, files in os.walk(os.path.join("/kaggle/input", entry)):
            for f in sorted(files):
                rel = os.path.relpath(os.path.join(root, f), "/kaggle/input")
                if n < 12:
                    print(f"      {rel}")
                n += 1
        print(f"      ... {n} files total")
    print()


def find_probabilities(slug):
    """Locate probabilities.npy inside a mounted kernel output.

    Searched recursively: `kaggle kernels files` reports basenames only, so the
    directory layout under the mount point is not worth assuming.
    """
    hits = glob.glob(f"/kaggle/input/{slug}/**/probabilities.npy", recursive=True)
    if not hits:
        hits = glob.glob(f"/kaggle/input/**/{slug}/**/probabilities.npy", recursive=True)
    if not hits:
        raise FileNotFoundError(
            f"probabilities.npy not found for {slug}. "
            f"Check kernel_sources in kernel-metadata.json."
        )
    print(f"  {slug}: {hits[0]}")
    return np.load(hits[0])


def image_stats(pil_img):
    """Cheap proxies for the paper's 'challenging conditions' (Figure 4).

    AQUA20 has no turbidity/lighting labels, so these three statistics stand in
    for them: dark, flat and colour-washed images are what turbid or poorly-lit
    underwater shots look like numerically. Computed on a 64x64 thumbnail --
    these are ranking signals, not measurements.
    """
    small = pil_img.convert("RGB").resize((64, 64))
    arr = np.asarray(small, dtype=np.float32) / 255.0
    gray = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    mx, mn = arr.max(axis=2), arr.min(axis=2)
    saturation = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return {
        "luminance": float(gray.mean()),
        "rms_contrast": float(gray.std()),
        "saturation": float(saturation.mean()),
    }


def main():
    show_mounted_inputs()

    print("=" * 68)
    print("LOADING PREDICTIONS")
    print("=" * 68)
    probs = {k: find_probabilities(v) for k, v in SOURCES.items()}

    names = load_dataset_builder("taufiktrf/AQUA20").info.features["label"].names
    name2i = {n: i for i, n in enumerate(names)}
    print(f"\nClasses ({len(names)}): {names}")

    test_split = load_dataset("taufiktrf/AQUA20")["test"]
    labels = np.array(test_split["label"])
    n_test = len(labels)
    print(f"Test images: {n_test}")

    for k, p in probs.items():
        if p.shape[0] != n_test:
            raise ValueError(f"{k} probabilities has {p.shape[0]} rows, expected {n_test}")

    pred = {k: p.argmax(1) for k, p in probs.items()}
    conf = {k: p.max(1) for k, p in probs.items()}
    ok = {k: pred[k] == labels for k in probs}
    for k in ("v2", "v4"):
        print(f"  {k}: top-1 {ok[k].mean() * 100:.2f}%  ({(~ok[k]).sum()} errors)")

    # ── Image statistics, for the challenging-conditions stratum ──────────
    print("\nComputing image statistics (1612 thumbnails)...")
    stats = [image_stats(test_split[i]["image"]) for i in range(n_test)]
    stats_df = pd.DataFrame(stats)
    # Percentile rank within the test set; low = dark / flat / washed out.
    ranks = stats_df.rank(pct=True)
    difficulty = (1 - ranks).mean(axis=1).to_numpy()  # high = more challenging
    print(f"  luminance   {stats_df.luminance.min():.3f} .. {stats_df.luminance.max():.3f}")
    print(f"  contrast    {stats_df.rms_contrast.min():.3f} .. {stats_df.rms_contrast.max():.3f}")
    print(f"  saturation  {stats_df.saturation.min():.3f} .. {stats_df.saturation.max():.3f}")

    # ── Selection ─────────────────────────────────────────────────────────
    picked = {}

    def take(idx, stratum, why):
        idx = int(idx)
        if idx in picked:
            return False
        picked[idx] = {
            "test_index": idx,
            "class": names[labels[idx]],
            "class_idx": int(labels[idx]),
            "stratum": stratum,
            "why_selected": why,
            "v2_pred": names[pred["v2"][idx]],
            "v4_pred": names[pred["v4"][idx]],
            "v2_correct": bool(ok["v2"][idx]),
            "v4_correct": bool(ok["v4"][idx]),
            "v2_conf": round(float(conf["v2"][idx]), 4),
            "v4_conf": round(float(conf["v4"][idx]), 4),
            "luminance": round(stats_df.luminance[idx], 4),
            "rms_contrast": round(stats_df.rms_contrast[idx], 4),
            "saturation": round(stats_df.saturation[idx], 4),
        }
        return True

    # A. Class coverage — one clean, confident, both-models-correct image per
    #    class. These are the "what does the model look at when it is right"
    #    baseline and they guarantee all 20 classes appear.
    print("\n" + "=" * 68)
    print("STRATUM A — class coverage")
    print("=" * 68)
    for c in range(len(names)):
        both = np.flatnonzero((labels == c) & ok["v2"] & ok["v4"])
        pool, note = both, "both models correct"
        if len(pool) == 0:
            pool, note = np.flatnonzero((labels == c) & ok["v4"]), "only v4 correct"
        if len(pool) == 0:
            pool, note = np.flatnonzero(labels == c), "no model correct"
        score = (conf["v2"][pool] + conf["v4"][pool]) / 2
        best = pool[int(np.argmax(score))]
        take(best, "class_coverage", f"cleanest {names[c]} ({note}, mean conf {score.max():.3f})")
    print(f"  selected {len(picked)}")

    # B. The paper's named confusion pairs.
    print("\n" + "=" * 68)
    print("STRATUM B — the paper's confusion pairs")
    print("=" * 68)
    candidates = []
    for a, b in PAPER_PAIRS:
        for true_name, pred_name in ((a, b), (b, a)):
            ti, pi = name2i[true_name], name2i[pred_name]
            m = np.flatnonzero((labels == ti) & ((pred["v2"] == pi) | (pred["v4"] == pi)))
            if len(m) == 0:
                print(f"  {true_name} -> {pred_name}: no such error in v2 or v4")
                continue
            # Rank by how confidently *some* model made this mistake.
            c = np.maximum(
                np.where(pred["v2"][m] == pi, conf["v2"][m], 0.0),
                np.where(pred["v4"][m] == pi, conf["v4"][m], 0.0),
            )
            best = m[int(np.argmax(c))]
            who = "v4" if pred["v4"][best] == pi else "v2"
            candidates.append((float(c.max()), best, true_name, pred_name, who))
    candidates.sort(reverse=True)
    for score, idx, t, p, who in candidates[: TARGET["paper_pair"]]:
        if take(idx, "paper_pair", f"paper pair: {t} -> {p} ({who} conf {score:.3f})"):
            print(f"  #{idx}: {t} -> {p} by {who} @ {score:.3f}")

    # C. fish <-> fishInGroups — v4's dominant error mode (23% of its errors),
    #    our own finding rather than the paper's. Arguably a labelling boundary.
    print("\n" + "=" * 68)
    print("STRATUM C — fish <-> fishInGroups")
    print("=" * 68)
    fi, gi = name2i["fish"], name2i["fishInGroups"]
    got = 0
    for ti, pi in ((fi, gi), (gi, fi)):
        m = np.flatnonzero((labels == ti) & (pred["v4"] == pi))
        order = m[np.argsort(-conf["v4"][m])]
        for idx in order:
            if got >= TARGET["fish_group"]:
                break
            if take(idx, "fish_group",
                    f"v4 dominant error: {names[ti]} -> {names[pi]} (conf {conf['v4'][idx]:.3f})"):
                print(f"  #{idx}: {names[ti]} -> {names[pi]} @ {conf['v4'][idx]:.3f}")
                got += 1

    # D. Rare classes — prefer images v2 got wrong and v4 got right, which is
    #    the capacity finding made visual.
    print("\n" + "=" * 68)
    print("STRATUM D — rare classes")
    print("=" * 68)
    for cname in RARE_CLASSES:
        c = name2i[cname]
        rescued = np.flatnonzero((labels == c) & ~ok["v2"] & ok["v4"])
        pool, note = rescued, "v2 wrong, v4 right (capacity rescue)"
        if len(pool) == 0:
            pool, note = np.flatnonzero((labels == c) & ~ok["v2"]), "v2 wrong"
        if len(pool) == 0:
            pool, note = np.flatnonzero(labels == c), "any"
        for idx in pool[np.argsort(-conf["v4"][pool])]:
            if take(idx, "rare_class", f"rare class {cname}: {note}"):
                print(f"  #{idx}: {cname} — {note}")
                break

    # E. Challenging conditions — darkest / flattest / most colour-washed
    #    images not already chosen.
    print("\n" + "=" * 68)
    print("STRATUM E — challenging conditions")
    print("=" * 68)
    got = 0
    for idx in np.argsort(-difficulty):
        if got >= TARGET["challenging"]:
            break
        if take(idx, "challenging",
                f"low light/contrast/saturation (difficulty pct {difficulty[idx]:.3f})"):
            s = stats_df.iloc[idx]
            print(f"  #{idx}: {names[labels[idx]]} — lum {s.luminance:.3f} "
                  f"contrast {s.rms_contrast:.3f} sat {s.saturation:.3f}")
            got += 1

    # ── Write ─────────────────────────────────────────────────────────────
    manifest = pd.DataFrame(list(picked.values())).sort_values("test_index")
    manifest.to_csv("xai_manifest.csv", index=False)

    provenance = {
        "n_images": int(len(manifest)),
        "n_test": int(n_test),
        "classes_covered": int(manifest["class"].nunique()),
        "per_stratum": manifest["stratum"].value_counts().to_dict(),
        "n_v2_wrong": int((~manifest["v2_correct"]).sum()),
        "n_v4_wrong": int((~manifest["v4_correct"]).sum()),
        "sources": SOURCES,
        "targets": TARGET,
    }
    json.dump(provenance, open("manifest_stats.json", "w"), indent=2)

    print("\n" + "=" * 68)
    print("MANIFEST")
    print("=" * 68)
    print(json.dumps(provenance, indent=2))
    print()
    print(manifest[["test_index", "class", "stratum", "v2_pred", "v4_pred"]].to_string(index=False))

    if manifest["class"].nunique() != len(names):
        missing = set(names) - set(manifest["class"])
        print(f"\n!! WARNING: classes missing from manifest: {sorted(missing)}")
    else:
        print(f"\nAll {len(names)} classes present.")
    print(f"\nWrote xai_manifest.csv ({len(manifest)} rows) + manifest_stats.json")


if __name__ == "__main__":
    main()
