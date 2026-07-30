# v1-smoke_resnet50

**Slug:** `farhantahsinkhan/aqua20-resnet50-smoke`
(pushed before the naming convention existed — see `../README.md`)

**Pushed:** 2026-07-30 · **Status:** ✅ COMPLETE · **Runtime:** ~4 min

## Purpose

Validate the full pipeline end-to-end after the bug fixes — **not** a real training run.
Epochs cut to 1 + 1 so a failure costs minutes instead of 90.

The Grad-CAM path had never executed successfully before this: Arian's run on the same notebook
died at the last cell with
`RuntimeError: a Tensor with 32 elements cannot be converted to Scalar`.

## Config

| | |
|---|---|
| Model | resnet50 (ImageNet pretrained) |
| Weighted sampler | on |
| Epochs | 1 (head) + 1 (fine-tune) |
| Machine | NvidiaTeslaT4 (2×T4) |
| Internet | on (downloads AQUA20 from HuggingFace) |

## Result

Reached the final cell. Metrics are meaningless at 1 epoch — the point was that nothing crashed.

Fixes confirmed working in production:

| Fix | Evidence |
|---|---|
| Grad-CAM `@torch.no_grad()` removed | Real heatmaps produced |
| Grad-CAM batch-size crash | Loop completed all 40 iterations |
| Per-model file naming | `best_resnet50.pth`, `confusion_matrix_resnet50.png` |
| Per-architecture target layer | ResNet50 → `layer4[-1]` |
| Stratified selection | 20 correct picks spanning **20 distinct classes**, one each |
| Misclassification targeting | `wrong_*.png` captured real errors (coral→jellyfish, crab→shrimp, diver→octopus) |

Sample output — the model focused on the branching, tentacle-like structure of a coral and
predicted jellyfish, i.e. the heatmap explains the error rather than just reporting it:
`output/gradcam/resnet50/wrong_000_coral-as-jellyfish.png`

## Notes

Outputs are not stored in this folder (see `.gitignore`). Re-pull with
`kaggle kernels output farhantahsinkhan/aqua20-resnet50-smoke -p <dir>`.

### ⚠️ Pulling results is unreliable — and the notebook is making it worse

Pull attempt 1 returned 23 of 40 Grad-CAM images and no `.log`. Attempt 2 timed out at 500 s
having fetched only 8 files.

The download walks files alphabetically and gets cut off mid-way — the evidence is that all 20
`ok_*.png` arrived and `wrong_*.png` stopped dead at `wrong_002`. The images do exist on Kaggle;
the client just never reaches them. **Always re-pull before concluding something is missing.**

The root cause is payload size, and **cell 24 doubles it for no benefit**:

| Path | What it is |
|---|---|
| `best_resnet50.pth` | written by training |
| `output/best_resnet50.pth` | identical 94 MB copy made by cell 24 |
| `outputs/gradcam/resnet50/*.png` | 40 images written by the Grad-CAM cell |
| `output/gradcam/resnet50/*.png` | identical 40 images copied by cell 24 |

`/kaggle/working` **is** the kernel's output directory — it is what `kernels output` downloads and
what the Output tab shows. Copying into `output/` inside it just ships everything twice: ~190 MB
and 80 images instead of ~95 MB and 40.

**Recommended before v2:** delete the copy step in cell 24 (keep the summary print). Halves the
transfer and removes the confusing `output/` vs `outputs/` near-duplicate naming.
