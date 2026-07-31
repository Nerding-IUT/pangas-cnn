# v3-smoke_convnext_base

**Slug:** `farhantahsinkhan/aqua20-v3-smoke-convnext-base`

**Pushed:** 2026-07-30 · **Status:** ⏳ pending · **Expected runtime:** ~15 min

URL: https://www.kaggle.com/code/farhantahsinkhan/aqua20-v3-smoke-convnext-base

## Purpose

Smoke test for the first non-ResNet architecture. 1 + 1 epochs, so it proves the *pipeline*,
never convergence. It has to answer five questions before v4 commits to a multi-hour run:

1. Does `convnext_base` build, load ImageNet weights, and train at all?
2. Does batch 32 at 224px fit in a single T4's 15 GB? ConvNeXt-Base is 3.5× ResNet50's FLOPs and
   has never been run here.
3. Does Grad-CAM work through `model.features[-1]`? That target layer has never actually executed
   — the plan's whole contribution depends on it.
4. Do the four new mechanics behave: macro-F1 checkpoint selection, seeding, `analysis/epoch_log.csv`,
   and the `err_` / `zz_ok_` Grad-CAM rename?
5. **How long is one epoch?** This is the number v4 is really being launched for — see below.

## The measurement v4 depends on

ConvNeXt-Base's per-epoch time is unknown and 100 epochs is a hard commitment. Kaggle kills a
session at ~12 h and a killed session saves **nothing**. So:

- read the stage-1 and stage-2 epoch times out of `analysis/epoch_log.csv` (`elapsed_sec`)
- stage 2 is the expensive one — it backprops through the whole backbone
- extrapolate `25 × t_stage1 + 75 × t_stage2` before pushing v4

Rough prior from FLOPs: ResNet50 is 4.1 GFLOPs and v2 took ~100 min for 25 + 75. ConvNeXt-Base is
15.4 GFLOPs, so **4–6 h** is the expectation. If the extrapolation lands past ~9 h, v4 needs a
smaller epoch budget rather than a hope.

A `TIME_BUDGET_SEC` guard (10.5 h) now backs this up: if Stage 2 overruns, training stops and the
run still produces evaluation, error analysis and Grad-CAM from the best checkpoint so far. It is
insurance, not a plan — a run that trips it has lost epochs.

## What changed since v2

### The architecture

| | v2 | v3 |
|---|---|---|
| Model | resnet50 (25.6M) | **convnext_base (87.6M)** |
| `MODEL_NAME` | `resnet50` | `convnext_base` |

`MODEL_NAME` is `convnext_base`, not `convnext`, on purpose. The repo's `convnext` is ConvNeXt-**Tiny**
(28M). The AQUA20 paper's best model — the 90.69% one — is **87.6M params**, which is ConvNeXt-Base.
Tiny and Base are different models and conflating them would misattribute the paper's number.
Both builders are now registered, so Tiny stays available under `convnext`.

### The four deferred fixes, all taken

| Fix | What it does | Was |
|---|---|---|
| **Macro-F1 selection** | `SELECTION_METRIC = "macro_f1"` picks the best checkpoint | val accuracy |
| **Seeding** | `SEED = 42` across `random` / `numpy` / `torch` / CUDA | unseeded |
| **Per-epoch logging** | appends to `analysis/epoch_log.csv` every epoch | nothing |
| **Grad-CAM rename** | `err_*` / `zz_ok_*` so errors sort first | `ok_*` / `wrong_*` |

**Macro-F1 selection is the one with a cost.** It is the right choice — training is class-balanced
by the sampler and macro-F1 is the headline number, while accuracy is dominated by fish + coral
(~54% of val). But it means v3/v4 and v2 no longer differ only by architecture. Both metrics are
recorded every epoch, so the accuracy-selected comparison can still be reconstructed from
`epoch_log.csv`; a fully clean read would need ResNet50 re-run under the same rule.

**Seeding is not bit-exact.** Weight init, data order, sampler draws and augmentation are fixed,
but `cudnn.benchmark = True` and nondeterministic GPU kernels remain, so expect small drift rather
than identical numbers. The trade was deliberate: autotuning is worth real minutes on a multi-hour
run, and the point of seeding here is comparable runs, not bit-identical ones.

`epoch_log.csv` appends per epoch rather than accumulating in memory, so the learning curves — which
the plan's week 7 wants and which nothing so far has produced — survive even if the session dies.

## Deliberately NOT changed

The training recipe is otherwise identical to v2: same two-stage schedule, same LRs, same batch
size, same sampler, same augmentation, same `ReduceLROnPlateau`. The point of these runs is to read
off what the *architecture* changes.

The head init is the known wart carried over on purpose. Every builder uses
`kaiming_normal_(mode="fan_out", nonlinearity="relu")` on the final Linear, which for a 20-way head
gives weight std ≈ 0.32 — far larger than the `trunc_normal_(std=0.02)` ConvNeXt ships with, and
large enough to saturate the softmax at the start of stage 1. ResNet50 survived it (v2 reached
85.61%), and changing it only for ConvNeXt would confound the comparison. **Per-epoch logging now
makes it observable**: if stage 1 starts with a huge loss and climbs slowly, that is this init, and
it becomes a cheap, well-motivated follow-up run rather than a guess.

## Config

| | |
|---|---|
| Model | convnext_base (ImageNet pretrained, 87.6M) |
| Weighted sampler | on (inverse-frequency) |
| Epochs | **1 + 1 (smoke)** |
| Batch size | 32 |
| Selection metric | macro_f1 |
| Seed | 42 |
| Time budget | 10.5 h (will not trip here) |
| Machine | NvidiaTeslaT4 (2×T4, one used) |
| Internet | on |

## Expected output layout

```
analysis/                        <- downloads FIRST
  epoch_log.csv                  NEW — per-epoch curves
  summary.json                   now records selection metric, best epoch, seed, n_params
  per_class.csv
  confusion_matrix.csv
  confusion_pairs.csv
  confident_mistakes.csv
  gradcam_manifest.csv
  probabilities.npy
confusion_matrix_convnext_base.png
gradcam/convnext_base/           err_*.png first, then zz_ok_*.png
training_state_convnext_base.json
weights_convnext_base.pth        ~350 MB    <- downloads LAST
```

The checkpoint is ~3.5× v2's 94 MB because the model is 3.4× larger. The naming rule matters more
than ever: a truncated pull should cost the weights, never the numbers.

## Result — ✅ COMPLETE (~8 min)

Every question the smoke test was pushed to answer came back clean.

| Question | Answer |
|---|---|
| convnext_base builds and trains? | ✅ **87,586,964 params** — matches the paper's 87.6M exactly |
| Batch 32 @ 224 on one T4? | ✅ no OOM |
| Grad-CAM through `features[-1]`? | ✅ 40 figures, all 20 classes, both directions |
| Macro-F1 selection / seeding / epoch log / `err_` rename? | ✅ all four |
| Pull behaviour? | ✅ 51 files, no truncation; `err_*` now arrives before `zz_ok_*` |

### The measurement v4 was waiting for

From `analysis/epoch_log.csv` (`elapsed_sec` is cumulative from training start):

| Stage | Epoch cost | × epochs | Projected |
|---|---|---|---|
| 1 (head only) | 85 s | 25 | 0.59 h |
| 2 (full fine-tune) | 262 s | 75 | 5.46 h |
| | | | **6.06 h** |

**3.6× v2's ResNet50** (~1.7 h), which lands almost exactly on the FLOPs prior of 3.75×
(15.4 vs 4.1 GFLOPs). So the 4–6 h expectation was right, slightly optimistic.

This changed a decision. `TIME_BUDGET_SEC` was 10.5 h here, sized against a 12 h session limit —
but `KAGGLE_RUN_GUIDE.md` treats the limit as **9 h**, and at 9 h a 10.5 h guard never fires and the
session dies with nothing saved. v4 uses **7.5 h**: 24% headroom over the projection, safely inside
either limit. The guard was insurance that would not have paid out.

### Smoke-test metrics — promising, not a result

1 + 1 epochs is nowhere near convergence, so these are a signal and nothing more:

| | v3 (2 epochs, convnext_base) | v2 (100 epochs, resnet50) |
|---|---|---|
| Top-1 | 80.89% | **85.61%** |
| Top-3 | 95.16% | 97.58% |
| Macro F1 | **0.7679** | 0.7588 |
| Macro precision | 0.7173 | 0.7810 |
| Macro recall | **0.8538** | 0.7667 |

After two epochs ConvNeXt-Base's macro-F1 is **already above fully-trained ResNet50's**. Top-1 is
still 4.7 pp behind, and the shape says why: recall 0.854 against precision 0.717 — it is casting a
wide net, over-predicting rare classes, which is what the weighted sampler asks for and what
precision then pays for. Expect precision to climb as training continues.

The rare classes v2 flat-out **ignored** are already being considered. v2's
`mean_true_prob_when_wrong` for shark was 0.002 — the model never entertained it. Here, after two
epochs, **shark recall is 0.947 and seaCucumber 0.900**. If that survives full training it is the
single most report-worthy difference between the two architectures.

Errors are also *less* confident than v2's (0.641 vs 0.828 mean confidence when wrong), and 54.9%
of them still rank the true class 2nd. Undertrained models are naturally less confident, so this
number is only worth comparing after v4.

### Log verification

Clean. `Device: cuda | Model: convnext_base`, `GPU: Tesla T4 (15.6 GB), 2 visible`, 4 workers,
`Target layer: Sequential (channels_last=False)`, Grad-CAM selection spanning all 20 classes,
383.4 MB total output. No OOM, no traceback. The only warnings are cosmetic — unauthenticated HF
Hub requests, and `SyntaxWarning`s from nbconvert's own dependencies.

**One idle resource worth noting: `2 visible` GPUs, one used.** `machine_shape: NvidiaTeslaT4`
gives 2×T4 and the notebook has never used `DataParallel`, so half the machine sits idle through a
6 h run. ConvNeXt uses LayerNorm rather than BatchNorm, so splitting the batch would be
mathematically clean here — unlike ResNet50, where it would change BatchNorm statistics. Plausibly
~1.7× faster.

**Deliberately not done for v4.** It would change the effective training setup mid-comparison, and
it touches Grad-CAM hooks and `state_dict` key prefixes (`module.`) — new failure modes in code
that has run exactly once. Worth a smoke test of its own if convnext_base runs become routine.

### The head-init worry: real but mild

Stage-1 epoch-1 mean train loss was **3.24** against ln(20) = 3.00 — elevated, consistent with the
oversized `kaiming_normal_(fan_out)` head init, but it recovered inside a single epoch to 62.6% val
accuracy. Not catastrophic, so carrying it unchanged for comparability remains the right call.
Per-epoch logging made this observable instead of speculative, exactly as intended.
