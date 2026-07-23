import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import (
    DEVICE,
    EPOCHS_STAGE1,
    EPOCHS_STAGE2,
    LR_HEAD,
    LR_BACKBONE,
    CHECKPOINT_PATH,
)
from data.dataset import get_dataloaders
from models.resnet50 import build_model, freeze_backbone, unfreeze_all


def train_one_epoch(model, loader, optimizer, criterion, desc):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=desc)
    for batch in pbar:
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(
            {"loss": f"{total_loss/(pbar.n+1):.4f}", "acc": f"{correct/total:.4f}"}
        )
    return total_loss / len(loader), correct / total


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total


def main():
    train_loader, val_loader, test_loader = get_dataloaders()
    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    # === Stage 1: Train head only ===
    freeze_backbone(model)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR_HEAD)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.1)
    for epoch in range(1, EPOCHS_STAGE1 + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            desc=f"Stage1 Epoch {epoch}/{EPOCHS_STAGE1}",
        )
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step(val_acc)
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> saved (best val acc: {best_acc:.4f})")
    # === Stage 2: Unfreeze & fine-tune ===
    unfreeze_all(model)
    optimizer = torch.optim.Adam(
        [
            {
                "params": [p for n, p in model.named_parameters() if "fc" not in n],
                "lr": LR_BACKBONE,
            },
            {"params": model.fc.parameters(), "lr": LR_HEAD},
        ]
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.1)
    for epoch in range(1, EPOCHS_STAGE2 + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            desc=f"Stage2 Epoch {epoch}/{EPOCHS_STAGE2}",
        )
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step(val_acc)
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> saved updated (best val acc: {best_acc:.4f})")
    print(f"\nTraining complete. Best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
