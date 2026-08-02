# v6-smoke_xai_resnet50

**Slug:** `farhantahsinkhan/aqua20-v6-smoke-xai-resnet50`

**Pushed:** 2026-08-02 · **Status:** see Result below

URL: https://www.kaggle.com/code/farhantahsinkhan/aqua20-v6-smoke-xai-resnet50

## Purpose

First execution of `xai.py` — the GradCAM + LIME + SHAP harness — against a real checkpoint.
3 images only. This is a **plumbing test, not science**: it proves the harness can mount a
checkpoint and the frozen manifest, run all three explainers plus the faithfulness metrics, and
write artifacts, before that pipeline is committed to 5 models × 39 images.

Deliberately run in Phase 0, against the checkpoint v2 already produced, rather than waiting for
the new models to train. LIME and SHAP are the schedule risk on this project — they are new code
against libraries it has never imported — so they get de-risked on day 1.

## Config

| | |
|---|---|
| Kernel type | script (`xai.py`) — not a notebook, per KAGGLE_RUN_GUIDE.md §2 |
| Model | resnet50, weights from v2 |
| GPU | NvidiaTeslaT4 |
| Internet | on (HF dataset, pip) |
| `kernel_sources` | `aqua20-v2-full-resnet50`, `aqua20-v5-manifest` |
| `SMOKE` / `N_SMOKE` | True / 3 |
| LIME | 1000 perturbations, quickshift segments |
| SHAP | GradientExplainer, 200 samples, 50-image background |
| Faithfulness | 50-step deletion/insertion, Gaussian-blur baseline (σ=11) |

## Version 1 — ❌ ERROR

```
TypeError: can't convert cuda:0 device type tensor to numpy.
  at shap_saliency -> int(np.asarray(idx).ravel()[0])
```

GradCAM and LIME both completed. `shap.GradientExplainer` returns its output-index tensor **on the
model's device**, and `np.asarray()` on a CUDA tensor raises rather than transferring. The local
dry-run never caught it because that ran CPU-only.

Fixed with a `_to_numpy()` helper applied to both return values. **Lesson: a CPU-only local dry-run
does not exercise device-transfer paths** — worth remembering before trusting the next one.

The version-1 log did carry one useful measurement: **LIME's 1000 perturbations took 7 s** on a T4,
against the ~10 s/image the plan budgeted. Per-model XAI cost is therefore not a concern.

## Version 2 — ⚠️ COMPLETE but the sanity check failed

Ran clean in 0.8 min. All three explainers produced maps, all metrics computed, figures written.
But **GradCAM lost to random on insertion** (0.5554 vs 0.6812) and the built-in check refused to
pass the run.

The cause was the sample, not the metric. The manifest is sorted by `test_index`, so `head(3)`
is not a sample at all — the first three rows are **#3, #319, #341, every one of them coral**.
Coral fills the entire frame, so restoring *any* random 20% of pixels recovers the prediction and
random's insertion score is inflated. Deletion, the more informative half, passed for all three
methods even here.

Two changes, one to fix it and one so the next surprise is diagnosable rather than mysterious:

- smoke now takes **one image per stratum** (`groupby("stratum").head(1)`), which also spreads the
  classes
- `p_blurred` — p(predicted class) on the fully-blurred baseline — is now logged per image and
  stored in `faithfulness.csv`. If insertion ever looks weak again, this says immediately whether
  the blur is failing to destroy the prediction or something else is wrong.

## Version 3 — ✅ COMPLETE, sanity check PASSED

5 images, one per stratum, five different situations: coral (paper pair), coral (clean),
crab (rare class), fish (fish/fishInGroups), fish (challenging conditions).

| method | deletion ↓ | insertion ↑ | concentration | p_blurred |
|---|---|---|---|---|
| shap | **0.1986** | 0.3579 | 0.5372 | 0.231 |
| lime | 0.2194 | **0.5736** | 0.3301 | 0.231 |
| gradcam | 0.2897 | 0.5107 | 0.2134 | 0.231 |
| *random* | *0.3274* | *0.3220* | *0.1898* | *0.231* |

All three beat random on **both** metrics. `p_blurred = 0.231` (down from ~1.0 unblurred) confirms
the Gaussian σ=11 baseline genuinely destroys the prediction, so insertion has real headroom — the
v2 failure really was the all-coral sample.

**Do not read the method ranking.** n=5 on one model. The point of this run is that the machinery
works, not what it says.

Agreement is low across the board (GradCAM~LIME ρ=0.18, GradCAM~SHAP ρ=0.05, LIME~SHAP ρ=0.01) with
GradCAM~random at ρ=-0.0005, which is the right shape: real methods correlate with each other a
little and with noise not at all. Whether agreement rises with model quality is a Phase 4 question.

### Runtime

**1.2 min for 5 images.** Per image: GradCAM ~0.0 s, LIME 7.5 s (1000 perturbations), SHAP ~2.0 s.
Extrapolating to the real run — 39 images — gives **~8 min per model**, against the ~1 h the plan
budgeted. XAI compute is a non-issue; all five models can run in one short wave.

### The first real observation, already

`figures/err_0581_fish-as-fishInGroups.png` is labelled **fish**, but the photograph plainly
contains three fish. ResNet50 predicted `fishInGroups` at 0.93. The model is arguably right and the
ground truth wrong. That is v4's fish↔fishInGroups labelling-boundary hypothesis showing up in an
actual image on the very first smoke run, and it is worth pulling into the report.

### Noted for later, not acted on

SHAP's maps are **sparse and speckled** — pixel-level rather than regional (concentration 0.537 vs
GradCAM's 0.213). That is expected of expected-gradients on images, but it makes SHAP the least
readable of the three panels. A smoothing pass would help the *figures*; the raw maps must stay
unsmoothed for the metrics. Cosmetic, deferred to Phase 3.

## What this run had to prove

- [x] `kernel_sources` mounts both the checkpoint and the manifest
- [x] GradCAM runs off the shared manifest rather than per-model error selection
- [x] LIME runs, and fast
- [x] SHAP GradientExplainer runs on a fine-tuned model on GPU
- [x] Deletion / insertion / concentration produce sane numbers
- [x] **The sanity check passes** — and it caught a real problem first, which is the point
- [x] The 4-panel figure is readable
