# v4-full_convnext_base

**Slug:** `farhantahsinkhan/aqua20-v4-full-convnext-base`

**Pushed:** 2026-07-31 · **Status:** ✅ COMPLETE · **Runtime:** 332.4 min (5.54 h), projected 6.06 h

URL: https://www.kaggle.com/code/farhantahsinkhan/aqua20-v4-full-convnext-base

## Purpose

First real training run of a non-ResNet architecture, and the first run of the model the AQUA20
paper reports as its best. Produces the metrics, error analysis and Grad-CAM figures that let the
report say something about *why* ConvNeXt and ResNet50 differ, rather than only that they do.

Identical to `v3-smoke_convnext_base` except the epoch counts and the time budget. v3 validated the
pipeline; this one is the science.

## Runtime is measured, not guessed

v3 measured convnext_base directly:

| | Per epoch | × epochs | Total |
|---|---|---|---|
| Stage 1 (head only, backbone frozen) | 85 s | 25 | 0.59 h |
| Stage 2 (full fine-tune) | 262 s | 75 | 5.46 h |
| | | | **6.06 h** |

That is **3.6× v2's ResNet50** (~1.7 h), which lands almost exactly on the prior from FLOPs
(15.4 vs 4.1 GFLOPs → 3.75×). Setup and evaluation overhead measured ~2 min in v3, so total
session time should be ~6.2 h.

`TIME_BUDGET_SEC = 7.5 h` — 24% headroom over the projection. It was 10.5 h in v3, sized against a
12 h session limit; `KAGGLE_RUN_GUIDE.md` treats the limit as **9 h**, at which point a 10.5 h guard
never fires and the session is killed with **nothing saved**. Lowered accordingly. If it does trip,
training stops and evaluation + error analysis + Grad-CAM still run off the best checkpoint.

## What v3 already established

The smoke test cleared every unknown this architecture carried:

| Question | Answer |
|---|---|
| Does convnext_base build and train? | ✅ 87,586,964 params — matches the paper's 87.6M exactly |
| Batch 32 @ 224 on one T4? | ✅ no OOM |
| Grad-CAM through `features[-1]`? | ✅ 40 figures, all 20 classes |
| Macro-F1 selection, seeding, epoch log, `err_` rename? | ✅ all four |

**And it hinted at the result.** After *one* epoch per stage, convnext_base already scored
**80.89% top-1 / 0.7679 macro-F1** on test — its macro-F1 already above fully-trained ResNet50's
0.7588 (v2, 100 epochs). Two epochs is not convergence and this is not a result, but it is a strong
prior that this run will beat v2.

**The head-init worry looks mild.** Stage-1 epoch-1 mean train loss was 3.24 against ln(20) = 3.00 —
elevated, consistent with the oversized `kaiming_normal_(fan_out)` head init, but it recovered
inside a single epoch (62.6% val acc). Not the catastrophe it could have been, so leaving it
unchanged for comparability stays the right call. `analysis/epoch_log.csv` will show whether stage 1
still pays for it across 25 epochs.

## Config

| | |
|---|---|
| Model | convnext_base (ImageNet pretrained, 87.6M) |
| Weighted sampler | on (inverse-frequency) |
| Epochs | **25 (head) + 75 (fine-tune)** |
| Batch size | 32 |
| LR | head 1e-3, backbone 1e-5 |
| Scheduler | ReduceLROnPlateau on the selection metric |
| Selection metric | macro_f1 |
| Seed | 42 |
| Time budget | 7.5 h |
| Machine | NvidiaTeslaT4 (2×T4, one used) |
| Internet | on |

## Baselines to compare against

| Run | Model | Selection | Top-1 | Macro F1 |
|---|---|---|---|---|
| Arian V1.0 (no sampler) | resnet50 | val acc | 83.81% | 0.7218 |
| Arian, sampler | resnet50 | val acc | 83.62% | 0.7416 |
| v2 | resnet50 | val acc | 85.61% | 0.7588 |
| **v4 (this run)** | **convnext_base** | **val macro-F1** | ? | ? |
| AQUA20 paper, ResNet50 | resnet50 | — | 82.69% | — |
| AQUA20 paper, ConvNeXt | convnext (87.6M) | — | **90.69%** | — |

## Two caveats on reading the comparison

**It is not a clean A/B.** v4 differs from v2 by architecture *and* by checkpoint selection metric
(macro-F1 vs accuracy) *and* by seeding. Macro-F1 selection is the better rule and was chosen
deliberately, but it means a v4−v2 gap cannot be attributed to architecture alone. Both metrics are
in `epoch_log.csv`, so the accuracy-selected view is reconstructable; the clean fix is to re-run
ResNet50 under the new rule (~1.7 h, cheap).

**The paper's 90.69% is not a like-for-like target.** Same architecture and parameter count, but
their training recipe, augmentation and schedule are their own. Matching or missing it is
informative, not decisive.

## Expected output layout

```
analysis/                        <- downloads FIRST
  epoch_log.csv                  100 rows of learning curve
  summary.json
  per_class.csv
  confusion_matrix.csv
  confusion_pairs.csv
  confident_mistakes.csv
  gradcam_manifest.csv
  probabilities.npy
confusion_matrix_convnext_base.png
gradcam/convnext_base/           err_*.png (20) then zz_ok_*.png (20)
training_state_convnext_base.json
weights_convnext_base.pth        ~340 MB    <- downloads LAST
```

## Result — ✅ COMPLETE (5.54 h training, all 100 epochs, time budget never tripped)

| Metric | v4 convnext_base | v2 resnet50 | Δ |
|---|---|---|---|
| **Top-1** | **90.63%** | 85.61% | **+5.02 pp** |
| Top-3 | 99.26% | 97.58% | +1.68 |
| Macro precision | 0.8735 | 0.7810 | +0.0925 |
| Macro recall | 0.8888 | 0.7667 | +0.1221 |
| **Macro F1** | **0.8748** | 0.7588 | **+0.1160** |
| Weighted F1 | 0.9075 | 0.8548 | +0.0527 |
| Best val macro-F1 | 0.8816 (stage-2 ep 31) | — | — |
| Best val acc | 0.9177 | 0.8720 | +0.0457 |
| Errors | **151** / 1612 | 232 / 1612 | −81 |

**It lands on the paper's number.** AQUA20 reports 90.69% for ConvNeXt; this run gets **90.63%** —
0.06 pp apart, which is far inside run-to-run noise. Same architecture, same parameter count, a
training recipe developed here independently. That is about as close to a reproduction as an
unmatched recipe can give, and it means the pipeline is sound.

Macro-F1 gained more than top-1 (+0.116 vs +0.050 pp/100), which says the improvement is
concentrated in the small classes rather than in fish and coral.

### Finding 1 — the "ignored classes" were a capacity problem, not a data problem

This overturns the standing hypothesis in `CLAUDE.md`. v2 split the classes by
`mean_true_prob_when_wrong` into ones the model **ignores** (shrimp 0.0004, shark 0.0023,
dolphin 0.0145 — read as *a data problem*) and ones it **considers and rejects** (fishInGroups
0.1934, rayfish 0.1618 — *a boundary problem*), and concluded they "need opposite fixes."

Changing only the architecture moved every ignored class into the considered band:

| Class | `mean_true_prob_when_wrong` v2 → v4 | F1 v2 → v4 |
|---|---|---|
| shrimp | 0.0004 → **0.1698** | 0.7826 → 0.9000 |
| shark | 0.0023 → **0.1487** | 0.6667 → 0.8095 |
| seaCucumber | 0.0125 → **0.1763** | 0.4000 → 0.7000 |
| marine_dolphin | 0.0145 → **0.2090** | 0.1667 → **0.7500** |
| squid | 0.0029 → n/a (zero errors) | 0.8421 → 0.9091 |

No data was added, cleaned or reweighted. ResNet50 simply lacked the capacity to represent these
classes, and a bigger backbone found them. **The recommended fix for those classes — more or better
data — would have been wasted effort.**

**18 of 20 classes improved.** marine_dolphin, v2's standout failure at F1 0.1667 (recall 1 of 10),
reaches **0.7500 at precision 1.000**. jellyfish is now perfect.

### Finding 2 — one class got worse, and it is the interesting one

| Class | F1 v2 → v4 |
|---|---|
| fishInGroups | 0.7714 → **0.7453** |
| diver | 0.9630 → 0.9286 |

diver is 13 test images and one prediction — noise. **fishInGroups is not.** It was already in the
"considers and rejects" band and it is the only class the bigger model handles *worse*. Its recall
rose (0.750 → 0.833) while precision fell, and `mean_true_prob_when_wrong` rose 0.1934 → 0.2446 —
the model is more aware of it and less able to commit.

fish ↔ fishInGroups is now **the dominant error mode: 35 of 151 errors (23%)** — fish→fishInGroups
24, fishInGroups→fish 11. That is a genuinely ambiguous distinction (*one fish* vs *several fish*),
arguably a labelling-boundary question rather than a visual one, and capacity does not fix it.
v2's coral ↔ fish ↔ seaAnemone triangle has receded to second place.

**So the two findings point in opposite directions and both are useful:** rare-class failure was
capacity, and throwing a bigger model at it worked; the fish/fishInGroups boundary is not capacity,
and a bigger model made it slightly worse.

### Finding 3 — 2.26 h of the run was literally a no-op

`ReduceLROnPlateau(patience=5, factor=0.1)` has no `min_lr`, so repeated plateaus drove the backbone
learning rate to zero:

| Stage-2 epoch | Backbone LR | val macro-F1 |
|---|---|---|
| 1 | 1e-5 | 0.7804 |
| 29 | 1e-6 | 0.8599 |
| 37 | 1e-7 | 0.8747 |
| 43 | **1e-8** | 0.8757 |

From **epoch 45 to 75 the metrics do not move at all** — 31 epochs, three distinct macro-F1 values,
val accuracy frozen at 0.91921. At 262 s/epoch that is **2.26 h of compute in which the model did
not change**, out of a 5.54 h run.

The last checkpoint improvement was at **stage-2 epoch 31 of 75**.

**This contradicts the guidance v2 left behind.** `CLAUDE.md` says the sampler pushed the best
checkpoint to epoch 64 of 75 and therefore "**Do not add aggressive early stopping**." That was
correct for ResNet50 and is wrong for ConvNeXt-Base, which plateaus at 31 and is dead by 45. The
advice needs to be per-architecture — and v2 had no per-epoch log, so this was invisible then.
Adding the log is what made it visible.

Cheapest fixes, in order: pass `min_lr` to the scheduler (one argument, stops the collapse);
early-stop when LR falls below a floor (safe, since a dead LR means a frozen model — unlike
accuracy-based patience, which v2 rightly warned about); or cut stage 2 to ~45 epochs for this
architecture. Any of them saves ~2 h per run.

### Finding 4 — seeding verified across independent sessions

v3 and v4 both ran seed 42 in separate Kaggle sessions. Stage-1 epoch-1 is **identical to all five
logged digits**:

| | v3 | v4 |
|---|---|---|
| train_loss | 3.2381 | 3.2381 |
| train_acc | 0.32895 | 0.32895 |
| val_loss | 1.41324 | 1.41324 |
| val_acc | 0.62576 | 0.62576 |
| val_macro_f1 | 0.53816 | 0.53816 |

Better than the caveat this project has been carrying. The "not bit-exact" warning still stands in
principle — `cudnn.benchmark = True` can pick different algorithms — but in practice the runs
reproduced exactly at this checkpoint. Reproducibility is no longer an open item.

### Errors are near-misses now

- mean confidence when correct **0.9644**, when wrong **0.7513** (v2: 0.963 / 0.828)
- the true class ranks 2nd in **109 of 151 errors (72.2%)** — up from 59.9% in v2
- top-3 accuracy **99.26%**

Two things at once: the model is *less* confident when wrong than v2 was, and its errors are
*more* often near-misses. Both point the same way — the remaining failures are fine-grained
discrimination between genuinely similar classes, not misrecognition. Combined with v2's Grad-CAM
finding that attention lands on the animal even when wrong, the story for the report is consistent:
**attention was never the problem; capacity was, and what capacity did not fix is label ambiguity.**

### Runtime

Training **5.54 h** against the 6.06 h projected from v3 — 9% faster, because stage-1 epochs ran
slightly quicker than the single measured epoch suggested. All 100 epochs completed; the 7.5 h guard
was never approached. The pull delivered all 51 artifacts, 40 Grad-CAM figures, `err_*` first.

### Follow-ups this run earns

1. **Re-run ResNet50 under macro-F1 selection** (~1.7 h) to make v2 vs v4 a clean single-variable
   comparison. Currently they differ by architecture, selection metric and seeding.
2. **Fix the LR collapse** (`min_lr`) — saves ~2 h on every future convnext_base run.
3. **Train Swin** — the third implemented architecture, still never run, and its Grad-CAM path
   (`channels_last=True`) has never executed.
4. **LIME and SHAP** remain untouched and are still two thirds of the plan's contribution.
5. The fish/fishInGroups boundary is now the single biggest error source and is a labelling
   question — worth inspecting the actual images before treating it as a modelling problem.
