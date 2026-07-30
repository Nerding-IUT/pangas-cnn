# v2-full_resnet50

**Slug:** `farhantahsinkhan/aqua20-v2-full-resnet50`

**Pushed:** 2026-07-30 (version 1) · **Status:** ✅ COMPLETE · **Expected runtime:** ~90 min

URL: https://www.kaggle.com/code/farhantahsinkhan/aqua20-v2-full-resnet50

## Purpose

First real training run on the fixed pipeline. Produces the metrics, a full error analysis, and
the Grad-CAM figures for the report — all from one consistent set of weights.

Arian's sampler run produced comparable metrics but crashed before Grad-CAM, so its heatmaps do
not exist and its numbers survive only as text in a log.

## Training is deliberately unchanged

No changes to the training recipe, so results stay **directly comparable** to Arian's sampler run.
Any difference is run-to-run noise, not a changed method. The only training-adjacent change since
his run is the removal of a duplicate forward pass per batch (speed and BatchNorm correctness,
not accuracy).

Consciously **not** done here, to keep the comparison clean — revisit in a later run:

- checkpoint selection is still on val **accuracy**, not macro-F1 (accuracy is dominated by
  fish+coral, ~54% of val, while macro-F1 is the headline number)
- no seeding, so runs are not reproducible
- no early stopping; stage 2 plateaued around epoch 7 in V1.0 and burned ~68 epochs

## Config

| | |
|---|---|
| Model | resnet50 (ImageNet pretrained) |
| Weighted sampler | on (inverse-frequency) |
| Epochs | 25 (head) + 75 (fine-tune) |
| Batch size | 32 |
| Machine | NvidiaTeslaT4 (2×T4) |
| Internet | on |

## Output layout

`/kaggle/working` **is** the kernel output, so nothing is copied into a subfolder of it — v1
shipped everything twice and both pulls truncated. Names are ordered so that a truncated download
costs you the weights, never the numbers:

```
analysis/                       ~150 KB   <- downloads FIRST
  summary.json                            headline metrics + confidence stats
  per_class.csv                           P/R/F1, top-3 recall, confidence split per class
  confusion_matrix.csv                    the matrix as data, not just a PNG
  confusion_pairs.csv                     every off-diagonal pair, ranked by count
  confident_mistakes.csv                  errors sorted by confidence, with true-class rank
  gradcam_manifest.csv                    which figure is which
  probabilities.npy                       full 1612x20 softmax, for any later re-analysis
confusion_matrix_resnet50.png
gradcam/resnet50/                40 imgs
training_state_resnet50.json
weights_resnet50.pth             94 MB    <- downloads LAST
```

**Worked as intended:** all 7 analysis files arrived within 25 s of starting the pull, long
before the checkpoint.

**One ordering flaw to fix in v3:** inside `gradcam/`, `ok_` sorts *before* `wrong_`, so the
correct-prediction figures download first and the interesting misclassified ones last — backwards.
Rename the prefixes (e.g. `err_` / `zz_ok_`) so errors come first.

## Error analysis included

Beyond the metrics that were already printed:

- **Confidence split** — mean confidence when right vs wrong. Tells you whether failures are
  confident (a real problem) or hesitant (a threshold problem).
- **Near-miss rate** — share of errors where the true class was still ranked 2nd.
- **Per-class top-3 recall** — was the right answer at least close, for this class?
- **`mean_true_prob_when_wrong`** — when the model misses this class, does it still give the true
  class meaningful probability, or does it not consider it at all?
- **Most confident mistakes** — ranked. These are the report-worthy failures, and the Grad-CAM
  selection now prioritises exactly these within each class.

## Baselines to compare against

| Run | Top-1 | Macro F1 | marine_dolphin F1 |
|---|---|---|---|
| Arian V1.0, no sampler | 83.81% | 0.7218 | 0.1538 |
| Arian, sampler | 83.62% | 0.7416 | 0.5333 |
| **v2 (this run)** | **85.61%** | **0.7588** | 0.1667 |

## Result — ✅ COMPLETE (~100 min)

| Metric | v2 | vs V1.0 | vs Arian sampler |
|---|---|---|---|
| Top-1 | **85.61%** | +1.80 | +1.99 |
| Top-3 | 97.58% | −0.06 | — |
| Macro precision | 0.7810 | +0.0196 | +0.0196 |
| Macro recall | 0.7667 | +0.0519 | +0.0132 |
| **Macro F1** | **0.7588** | +0.0370 | +0.0172 |
| Weighted F1 | 0.8548 | +0.0200 | +0.0185 |
| Best val acc | 0.8720 | +0.0031 | +0.0069 |
| Errors | 232 / 1612 | | |

Best on every aggregate metric. **Caveat:** the only training-relevant change since Arian's run
was removing the duplicate forward pass (which was corrupting BatchNorm running statistics). The
+2 pp is ~2.2 standard errors, so it is *consistent with* that fix helping but a single unseeded
run cannot prove it. Seeding + a repeat would settle it.

### Error analysis

**The model fails confidently but narrowly.**

- mean confidence when correct: **0.963**; when wrong: **0.828** — errors are not hesitant, so
  no confidence threshold will filter them out
- but the true class was ranked **2nd in 139 of 232 errors (59.9%)**, and top-3 accuracy is
  97.58% — the right answer is nearly always near the top

So this is a *discrimination* problem between visually similar classes, not a recognition failure.

**Confusions are dominated by the three biggest classes** (coral 348, fish 538, seaAnemone 221):

| Pair (both directions) | Count |
|---|---|
| coral ↔ fish | 33 |
| coral ↔ seaAnemone | 24 |
| fish ↔ seaAnemone | 22 |

Coral↔SeaAnemone is one of the pairs the AQUA20 paper flagged. By *rate* rather than count the
worst are eel→fish (12.2% of all eels) and fishInGroups→coral (11.1%).

**Rare classes are ignored, not confused.** `mean_true_prob_when_wrong` — the probability the
true class still receives when the model gets it wrong — splits the classes cleanly:

| Ignored (≈0 probability) | | Considered but rejected | |
|---|---|---|---|
| shrimp | 0.0004 | fishInGroups | 0.1934 |
| shark | 0.0023 | rayfish | 0.1618 |
| squid | 0.0029 | fish | 0.1350 |
| seaCucumber | 0.0125 | crab | 0.1332 |
| marine_dolphin | 0.0145 | coral | 0.1092 |

These need different fixes: the left column is a representation/data problem, the right column a
decision-boundary problem.

**marine_dolphin is the standout failure** — recall 0.10 (1 of 10), and it is *more* confident
when wrong (0.887) than when right (0.866). It regressed vs Arian's run (F1 0.53 → 0.17), but on
10 test images that is 4 correct → 1 correct: real, and extremely noisy.

**flatworm has top-3 recall of 1.000** with F1 only 0.571 — the right answer is *always* in the
top 3. That is a pure ranking problem and probably the cheapest class to fix.

**diver was perfect** — recall 1.000, zero errors.

### Caveat on rare classes

marine_dolphin, octopus, seaCucumber and squid have only 10 test images each. One sample moves
F1 by 0.05–0.10, so per-class swings between runs are mostly noise. Aggregate metrics are the
reliable comparison; treat rare-class deltas as indicative only.

### Epoch budget — earlier advice was wrong

| | V1.0 (no sampler) | v2 (sampler) |
|---|---|---|
| Stage 2 plateau | ~epoch 7 | ~epoch 33 |
| Best epoch | 44 | **64** |
| Stage 1 end / Stage 2 end val acc | 0.7934 / 0.8544 | 0.7287 / 0.8712 |

The sampler changed the training dynamics: v2 kept improving far longer, and its **best
checkpoint came at epoch 64 of 75**. Early stopping with a patience of ~10 would have halted
around epoch 43 and *lost* accuracy.

So the "~68 wasted epochs" criticism from V1.0 does not carry over. 42 of 75 stage-2 epochs
(56%) bought ≤0.5 pp — still not free, but the long tail is now earning something. **Do not add
aggressive early stopping.** If trimming, cut stage 1 instead: it ends at 0.7287, below where
stage 2 starts, so its last epochs contribute little.

## Pull result

Downloaded cleanly: **51 files, no truncation** — 7/7 analysis, 40/40 Grad-CAM, weights, and log.
Removing the duplicate `output/` copy halved the payload and fixed the problem that broke both
v1 pulls.

## Grad-CAM highlight

`wrong_007_marine_dolphin-as-turtle.png` — predicted turtle at 1.00 confidence. The heatmap sits
squarely on the animal's broad rounded body and spread flippers, not on the background. Read
that way, the model attended to the right object and the *pose* is the problem: head-on with
flippers extended, the body plan reads as shell-plus-flippers.

This is precisely the plan's Week-6 question — does the model look in the wrong place, or the
right place and still fail? Here it is the latter, which argues the failure is discrimination,
not attention.
