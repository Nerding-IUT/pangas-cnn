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

| Run | Model | Sampler | Epochs | Kaggle slug | Status | Top-1 | Macro F1 | Grad-CAM |
|---|---|---|---|---|---|---|---|---|
| v1-smoke_resnet50 | resnet50 | on | 1 + 1 | `aqua20-resnet50-smoke` | ✅ COMPLETE | n/a (smoke) | n/a | ✅ works |
| v2-full_resnet50 | resnet50 | on | 25 + 75 | `aqua20-v2-full-resnet50` | ✅ COMPLETE | **85.61%** | **0.7588** | ✅ 40 imgs |

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

Then poll with `kaggle kernels status farhantahsinkhan/<slug>` and pull results with
`kaggle kernels output farhantahsinkhan/<slug> -p <dir>`. Full procedure, including the
background-poller loop and the JSON log decoder, is in `../../KAGGLE_RUN_GUIDE.md`.

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
