import multiprocessing as _mp
try:
    _mp.set_start_method("fork")
except RuntimeError:
    pass

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from datasets import load_dataset_builder

from config import DEVICE, CHECKPOINT_PATH, MODEL_NAME
from data.dataset import get_dataloaders
from models import MODEL_REGISTRY


def get_class_names():
    builder = load_dataset_builder("taufiktrf/AQUA20")
    return builder.info.features["label"].names


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_top3 = [], [], []

    for batch in loader:
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        outputs = model(images)

        _, preds = outputs.max(1)
        _, top3 = outputs.topk(3, dim=1)

        all_preds.append(preds.cpu())
        all_top3.append(top3.cpu())
        all_labels.append(labels.cpu())

    return (
        torch.cat(all_labels).numpy(),
        torch.cat(all_preds).numpy(),
        torch.cat(all_top3).numpy(),
    )


def compute_topk_accuracy(labels, topk_preds, k):
    return np.mean([labels[i] in topk_preds[i, :k] for i in range(len(labels))])


def main():
    class_names = get_class_names()
    print(f"Classes ({len(class_names)}): {class_names}\n")

    _, _, test_loader = get_dataloaders()

    model = MODEL_REGISTRY[MODEL_NAME]["build"]().to(DEVICE)
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)

    labels, preds, top3_preds = evaluate(model, test_loader)

    top1_acc = accuracy_score(labels, preds)
    top3_acc = compute_topk_accuracy(labels, top3_preds, 3)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )

    print("=" * 60)
    print(f"Top-1 Accuracy:  {top1_acc:.4f} ({top1_acc * 100:.2f}%)")
    print(f"Top-3 Accuracy:  {top3_acc:.4f} ({top3_acc * 100:.2f}%)")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall:    {recall:.4f}")
    print(f"Macro F1-Score:  {f1:.4f}")
    print("=" * 60)

    print("\nPer-class Classification Report:")
    print(classification_report(labels, preds, target_names=class_names, digits=4))

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Confusion matrix saved to confusion_matrix.png")


if __name__ == "__main__":
    main()
