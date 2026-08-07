# Model Dossier: ConvNeXt-Base

This dossier analyzes the explainability and saliency map performance for **ConvNeXt-Base** (our top-performing CNN architecture).

---

## 1. Performance Overview
- **Ours Top-1 Accuracy:** **90.63%** (vs. paper's **90.69%**, reproducing it within 0.06 pp)
- **Macro F1 Score:** **0.8748**
- **Sanity Check:** **PASSED** (all saliency methods beat random baseline)

---

## 2. Explainability Summary

| Method | Mean Deletion AUC↓ | Mean Insertion AUC↑ | Mean Concentration↑ |
|:---|:---|:---|:---|
| **Grad-CAM** | 0.4905 | 0.8345 | 0.4912 |
| **LIME** | 0.4640 | 0.8467 | 0.3705 |
| **SHAP** | 0.4056 | 0.7519 | 0.5591 |

- **High-Capacity Robustness:** ConvNeXt-Base exhibits relatively high Deletion AUC (~0.46 to 0.49). As discussed in the main findings, this is a sign of highly distributed feature representations; deleting the most salient local features does not cause the model to crash because it can immediately leverage redundant contextual cues.
- **Saliency Focus:** ConvNeXt has the highest Grad-CAM Concentration (0.4912) among CNNs, showing its attention maps are sharp and align well with the objects of interest.

---

## 3. Notable Strengths & Rescues
- **Rare Class Performance (4/4 correct):** Successfully classified all rare class images in the manifest (crab, dolphin, octopus, shrimp). Saliency maps show that ConvNeXt-Base isolates the actual body shapes of these rare animals instead of focusing on background features.
- **Paper Confusion Pairs (3/8 correct):** Outperformed the baseline models by correctly identifying difficult confusion pairs like flatworm (#1031), dolphin (#1067), and shark (#1468). Saliency maps show tight attention on the distinct diagnostic regions (e.g., dolphin's dorsal fin shape, flatworm's skin patterns).

---

## 4. Instructive Misclassifications
1. **Test Index #3 (Coral labeled, predicted SeaAnemone with 0.998 conf):** Focuses heavily on the soft, branching tentacles of the organism. The visual features are extremely similar to sea anemones, causing a high-confidence misclassification.
2. **Test Index #341 (Coral labeled, predicted Starfish with 1.000 conf):** The model focuses on the rigid, star-shaped geometry of the coral species. The structure mimics the symmetry of a starfish.
3. **Test Index #944 (Fish labeled, predicted Eel with 1.000 conf):** Saliency maps focus on the long, slender body shape of the fish species, which matches the morphometric features of an eel.
