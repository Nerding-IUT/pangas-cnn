from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from config import BATCH_SIZE, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, NUM_WORKERS


def get_transform(train=True):
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(IMG_SIZE + 32),
                transforms.CenterCrop(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )


def transform_images(examples, transform_fn):
    examples["image"] = [transform_fn(img.convert("RGB")) for img in examples["image"]]
    return examples


def get_dataloaders():
    dataset = load_dataset("AQUA20")
    split = dataset["train"].train_test_split(test_size=0.2, seed=42)
    train_split = split["train"]
    val_split = split["test"]
    test_split = dataset["test"]
    train_split.set_transform(
        lambda examples: transform_images(examples, get_transform(train=True))
    )
    val_split.set_transform(
        lambda examples: transform_images(examples, get_transform(train=False))
    )
    test_split.set_transform(
        lambda examples: transform_images(examples, get_transform(train=False))
    )
    train_loader = DataLoader(
        train_split, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_split, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_split, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    return train_loader, val_loader, test_loader
