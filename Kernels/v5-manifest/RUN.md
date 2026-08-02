# v5-manifest

**Slug:** `farhantahsinkhan/aqua20-v5-manifest`

**Pushed:** 2026-08-02 · **Status:** ✅ COMPLETE · **Runtime:** ~2 min (CPU, no GPU)

URL: https://www.kaggle.com/code/farhantahsinkhan/aqua20-v5-manifest

## Purpose

Choose, **once**, the set of test images that every XAI kernel will explain.

`notebook.ipynb` picks its Grad-CAM images from *that model's own errors*, so no two models are
ever explained on the same pictures — which makes a cross-model comparison meaningless. This kernel
selects a fixed, model-agnostic set of test indices and freezes them. It is the foundation of
direction #3.

CPU-only: no model is loaded. It reads the test-set probability matrices v2 and v4 already
produced, plus the images themselves for the challenging-conditions statistics.

## Config

| | |
|---|---|
| Kernel type | script (`manifest.py`) |
| GPU | **off** |
| Internet | on (HF dataset) |
| `kernel_sources` | `aqua20-v2-full-resnet50`, `aqua20-v4-full-convnext-base` |

## Result — ✅ 39 images, all 20 classes

| Stratum | n | What it is for |
|---|---|---|
| `class_coverage` | 20 | one clean, both-models-correct image per class — the "what does it look at when right" baseline |
| `paper_pair` | 8 | the AQUA20 paper's named confusion pairs |
| `rare_class` | 4 | octopus, marine_dolphin, crab, shrimp |
| `challenging` | 4 | darkest / flattest / most colour-washed images |
| `fish_group` | 3 | fish ↔ fishInGroups, v4's dominant error mode |

14 of the 39 are v2 errors, 8 are v4 errors — a deliberate mix of successes and failures.

### Verification that the inputs are the real thing

Recomputing top-1 from the mounted probability matrices reproduced both headline numbers exactly:
**v2 85.61% (232 errors), v4 90.63% (151 errors)**. These match `Kernels/README.md`, so the
mounted arrays are genuinely the ones behind the reported results.

### All five of the paper's confusion pairs exist in our models' errors

Two of them in **both** directions, which is better than expected:

| | |
|---|---|
| #944 | fish → eel (v4, conf 1.000) |
| #1031 | flatworm → seaSlug (v2, 1.000) |
| #1425 | seaSlug → flatworm (v4, 0.988) |
| #341 | coral → starfish (v4, 1.000) |
| #3 | coral → seaAnemone (v4, 0.999) |
| #1377 | seaAnemone → coral (v4, 1.000) |
| #1067 | marine_dolphin → shark (v2, 0.997) |
| #1468 | shark → marine_dolphin (v2, 0.996) |

Every one is a **near-certain** mistake (confidence ≥ 0.988). This is the "confidently wrong"
behaviour v2's error analysis found, now confirmed on v4 and pinned to specific images.

### All four rare classes landed on capacity-rescue cases

Every rare-class slot filled with an image where **v2 was wrong and v4 was right** (#1076 octopus,
#1070 marine_dolphin, #350 crab, #1478 shrimp). That is the v4 finding — rare-class failure was
capacity, not data — available as a *visual* before/after rather than only a table of F1 numbers.

### Challenging conditions

The luminance / RMS-contrast / saturation proxy found genuinely murky frames: #784 fish
(luminance 0.204, contrast 0.097), #725 fish, #1039 jellyfish, #1075 octopus. Test-set ranges were
luminance 0.106–0.755, contrast 0.056–0.331, saturation 0.115–0.994, so these sit at the dark,
flat end rather than being merely below average.

## ⚠️ The finding that mattered most

**`kernel_sources` does not mount where the docs imply.** Outputs land at

```
/kaggle/input/notebooks/<username>/<slug>/...
```

**not** `/kaggle/input/<slug>/`. `manifest.py` only survived because it globbed recursively instead
of hard-coding the path. `xai.py` was written against the wrong assumption and was corrected before
its first push. Anything mounting a kernel output must search recursively.

Layout inside the mount is otherwise preserved: `analysis/probabilities.npy`,
`weights_resnet50.pth` at top level, 111 files total.

## Output

```
xai_manifest.csv       39 rows: test_index, class, stratum, why_selected,
                       v2_pred, v4_pred, v2/v4 correct+conf, luminance,
                       rms_contrast, saturation
manifest_stats.json    provenance and per-stratum counts
```

**This file is frozen.** Changing it invalidates every comparison made against it. If the roster or
the image set needs to change, create `v<N>-manifest2` rather than re-running this kernel.
