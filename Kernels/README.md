# Kernels — Kaggle run registry

One folder per Kaggle run. A folder is a **frozen snapshot of what was actually pushed** — once a
run has been pushed, do not edit its notebook. To change something, create the next `v<N>` folder.

The live working copy is `../notebook.ipynb`. Snapshots are copied *from* it into a new run folder
at push time.

## Folder layout

```
Kernels/
├── README.md                  <- this file (the run log below)
├── .gitignore                 <- keeps pulled artifacts out of git
├── v1-smoke_resnet50/
│   ├── notebook.ipynb         <- exactly what was pushed
│   ├── kernel-metadata.json
│   └── RUN.md                 <- purpose, status, results
└── v2-full_resnet50/
    └── ...
```

## Naming

| | Format | Example |
|---|---|---|
| Folder | `v<N>-<purpose>_<model>` | `v2-full_resnet50` |
| Kaggle slug | `aqua20-v<N>-<purpose>-<model>` | `aqua20-v2-full-resnet50` |

`<N>` increments globally, never reused. `<purpose>` is `smoke` (1 epoch, pipeline check) or
`full` (real training run). `<model>` matches `MODEL_NAME` in the notebook: `resnet50`,
`convnext`, `swin`.

Kaggle slugs cannot contain underscores, so the folder's `_` becomes `-` in the slug.

> **Exception:** `v1-smoke_resnet50` was pushed as `aqua20-resnet50-smoke`, before this convention
> existed. Its metadata records the real slug. Everything from v2 follows the rule above.

## Run log

| Run | Model | Sampler | Epochs | Kaggle slug | Status | Top-1 | Macro F1 | Notes |
|---|---|---|---|---|---|---|---|---|
| v1-smoke_resnet50 | resnet50 | on | 1 + 1 | `aqua20-resnet50-smoke` | ✅ COMPLETE | n/a (smoke) | n/a | pipeline check |
| v2-full_resnet50 | resnet50 | on | 25 + 75 | `aqua20-v2-full-resnet50` | ✅ COMPLETE | **85.61%** | **0.7588** | 40 Grad-CAM imgs |
| v3-smoke_convnext_base | convnext_base | on | 1 + 1 | `aqua20-v3-smoke-convnext-base` | ✅ COMPLETE | 80.89% (smoke) | 0.7679 (smoke) | smoke only, do not quote |
| **v4-full_convnext_base** | convnext_base | on | 25 + 75 | `aqua20-v4-full-convnext-base` | ✅ COMPLETE | **90.63%** | **0.8748** | current best; reproduces paper |
| v5-manifest | — (CPU) | — | — | `aqua20-v5-manifest` | ✅ COMPLETE | — | — | frozen 39-image XAI manifest |
| v6-smoke_xai_resnet50 | resnet50 | — | — | `aqua20-v6-smoke-xai-resnet50` | ✅ COMPLETE | — | — | GradCAM+LIME+SHAP harness proven |
| v7-smoke_newmodels | swin+inc+vgg | — | smoke | `aqua20-v7-smoke-newmodels` | ✅ COMPLETE | — | — | all 3 archs smoke passed |
| **v8-full_swin** | swin | on | 25 + 75 | `aqua20-v8-full-swin` | ✅ COMPLETE | **89.83%** | **0.8631** | beats paper's 88.65% by +1.18 pp |
| **v9-full_inceptionv3** | inceptionv3 | on | 25 + 75 | `aqua20-v9-full-inceptionv3` | ✅ COMPLETE | **83.68%** | **0.7863** | beats paper's 76.36% by +7.32 pp |
| **v10-full_vgg19** | vgg19 | on | 25 + 75 | `aqua20-v10-full-vgg19` | ✅ COMPLETE | **66.07%** | **0.5352** | paper: 77.98%; 140M params overfits |
| v11-xai_resnet50 | resnet50 | — | XAI | `aqua20-v11-xai-resnet50` | ✅ COMPLETE | — | — | 39 images × 3 XAI methods |
| v12-xai_convnext_base | convnext_base | — | XAI | `aqua20-v12-xai-convnext-base` | ✅ COMPLETE | — | — | 39 images × 3 XAI methods |
| v13-xai_swin | swin | — | XAI | `aqua20-v13-xai-swin` | ✅ COMPLETE | — | — | 39 images × 3 XAI methods; SHAP OOM → zero fallback |
| v14-xai_inceptionv3 | inceptionv3 | — | XAI | `aqua20-v14-xai-inceptionv3` | ✅ COMPLETE | — | — | 39 images × 3 XAI methods |
| v15-xai_vgg19 | vgg19 | — | XAI | `aqua20-v15-xai-vgg19` | ✅ COMPLETE | — | — | 39 images × 3 XAI methods |
| v16-resnet50_label_smooth | resnet50 | on | 25 + 75 | `aqua20-v16-resnet50-label-smooth` | ✅ COMPLETE | 85.36% | 0.7631 | label_smoothing=0.1; neutral vs baseline |
| v17-tta_resnet50 | resnet50 | — | eval | `aqua20-v17-tta-resnet50` | ✅ COMPLETE | **86.72%** | **0.7722** | 8-view TTA on v2 weights; +1.11 pp Top-1 |
| v18-resnet50_cosine_annealing | resnet50 | on | 25 + 75 | `aqua20-v18-resnet50-cosine-anneal` | ✅ COMPLETE | 85.48% | 0.7676 | CosineAnnealingLR; +0.88 pp F1 |
| **v19-tta_cosine_resnet50** | resnet50 | — | eval | `aqua20-v19-tta-cosine-resnet50` | ✅ COMPLETE | **87.03%** | **0.7908** | 8-view TTA on v18 weights; **best ResNet-50 result** |



### The XAI track (directions #2 and #3)

From v5 the registry carries a second kind of run. **Training kernels and explainability kernels are
decoupled**: training produces a checkpoint, and a separate `xai.py` kernel mounts that checkpoint
via `kernel_sources` and explains it. This exists because `notebook.ipynb` picks its Grad-CAM images
from *that model's own errors*, so no two models were ever explained on the same pictures — which
makes a cross-model comparison meaningless.

`v5-manifest` fixes that by choosing 39 test images **once**, model-agnostically, and freezing them.
Every XAI kernel explains that same set. Plan and phasing: `../../XAI_EXECUTION_PLAN.md`.

⚠️ **`kernel_sources` does not mount at `/kaggle/input/<slug>/`.** It mounts at
`/kaggle/input/notebooks/<username>/<slug>/`. Glob recursively; do not hard-code either path.

**v4 is the best run and reproduces the paper.** AQUA20 reports 90.69% for ConvNeXt; v4 gets
90.63%. Against v2's ResNet50 it is +5.02 pp top-1 and +0.116 macro-F1, with 151 errors instead of
232. Details in `v4-full_convnext_base/RUN.md`.

Two results there change how future runs should be set up:

- **The rare classes v2 "ignored" were a capacity problem, not a data problem.** Changing only the
  architecture moved shrimp, shark, seaCucumber and marine_dolphin from ~0 probability-when-wrong
  into the 0.15–0.21 band. marine_dolphin went F1 0.1667 → 0.7500. No data was touched. The fix
  `CLAUDE.md` recommended for these classes would have been wasted effort.
- **`ReduceLROnPlateau` has no `min_lr`, so it drove the backbone LR to 1e-8 by stage-2 epoch 43.**
  Epochs 45–75 changed nothing at all — 2.26 h of a 5.54 h run. Last improvement was epoch 31 of 75.
  v2's "do not add early stopping" advice was ResNet50-specific and does not transfer.

v3's numbers are from **2 epochs** and are not a result — they exist because the smoke test runs the
full evaluation path. Do not quote them against v2.

### Measured epoch cost (from v3)

| Model | Stage-1 epoch | Stage-2 epoch | 25 + 75 projects to |
|---|---|---|---|
| resnet50 (v2) | — | — | ~1.7 h (actual) |
| convnext_base | 85 s | 262 s | **6.06 h** |

ConvNeXt-Base is **3.6× ResNet50** — matching its FLOPs ratio (15.4 vs 4.1 GFLOPs). Budget any
future convnext_base run at ~6.5 h wall clock, and note that its checkpoint is ~340 MB against
ResNet50's 94 MB.

Runs from v3 carry a `TIME_BUDGET_SEC` guard: if Stage 2 overruns, training stops and evaluation +
error analysis + Grad-CAM still run off the best checkpoint, instead of the session being killed
with nothing saved. **Size it below the session limit** — `KAGGLE_RUN_GUIDE.md` treats that as 9 h.

### Checkpoint selection changed at v3

v1/v2 selected the best checkpoint on val **accuracy**; v3 onward selects on val **macro-F1**
(`SELECTION_METRIC` in the notebook). Macro-F1 is the consistent choice — training is class-balanced
by the sampler, and accuracy is dominated by fish + coral (~54% of val). The cost is that
v3/v4 vs v2 is no longer a pure architecture comparison. Both metrics are logged every epoch in
`analysis/epoch_log.csv`, so an accuracy-selected view can still be reconstructed; a fully clean
comparison would need ResNet50 re-run under the new rule.

Runs from v3 are also **seeded** (`SEED = 42`) — reproducible up to cuDNN autotuning, not bit-exact.

### `convnext` vs `convnext_base`

`convnext` is ConvNeXt-**Tiny** (28M). The AQUA20 paper's best model at 90.69% is **87.6M params**
— that is ConvNeXt-**Base**, registered separately as `convnext_base`. They are different models;
do not report Tiny's number as the paper's ConvNeXt.

### Prior runs (not in this registry)

These predate the registry and live elsewhere; listed for traceability only.

| Run | Where | Status | Result |
|---|---|---|---|
| Arian V1.0 (no sampler) | `../../Kaggle Files/V1.0/` | ✅ COMPLETE | Top-1 83.81%, macro F1 0.7218 |
| Arian, sampler added | `mubtasimsajid/pangas-cnn` (his account) | ❌ ERROR at Grad-CAM cell | Top-1 83.62%, macro F1 0.7416 — training finished, only the last cell crashed |

## How to push a run

```bash
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export KAGGLE_API_TOKEN="<token>"        # see KAGGLE_RUN_GUIDE.md
cd "Kernels/v2-full_resnet50"
timeout 120 kaggle kernels push -p . > /tmp/push.txt 2>&1 ; grep -v outdated /tmp/push.txt
```

Then poll with `kaggle kernels status farhantahsinkhan/<slug>`. Full procedure, including the
background-poller loop and the JSON log decoder, is in `../../KAGGLE_RUN_GUIDE.md`.

### ⚠️ Pull results INTO the run folder — never into a temp directory

```bash
kaggle kernels output farhantahsinkhan/<slug> -p "Kernels/v<N>-<purpose>_<model>/outputs"
```

Pulling to a scratch/temp path has already nearly lost a set of figures once: `%TEMP%` gets cleared
routinely, and Kaggle output is the *only* copy of anything a kernel produced. `outputs/` inside the
run folder keeps each artifact next to the run that made it, and `.gitignore` already excludes it so
nothing is committed to the repo.

**Report-bound figures get promoted out.** `Kernels/.gitignore` hides `*/outputs/` and every `*.png`
under `Kernels/`, which is right for 340 MB checkpoints and wrong for a figure going into the
report. Copy those to `../../report_assets/` with a self-describing name — that folder is outside
any git repo and is treated as permanent. See `report_assets/README.md`.

After a run reaches a terminal state, record the outcome in that run's `RUN.md` **and** in the
run-log table above.

### Pulling results is slow and truncates

`kaggle kernels output` walks files alphabetically and frequently gets cut off part-way — v1
needed two attempts and still did not deliver every file. If something looks missing, **re-pull
before assuming the run failed to produce it.**

Payload size was the driver: v1's notebook copied everything into `output/` inside
`/kaggle/working`, which is *already* the output directory, shipping ~190 MB and 80 files instead
of ~95 MB and 40. **Fixed from v2 on** — see `v1-smoke_resnet50/RUN.md` for the diagnosis.

v2 also names its artifacts so the alphabetical download order works *for* you:

```
analysis/  <  confusion_matrix_*  <  gradcam/  <  training_state_*  <  weights_*
```

The ~150 KB of CSV/JSON in `analysis/` arrives first; the 94 MB checkpoint last. A truncated pull
now costs you the weights, never the numbers.
