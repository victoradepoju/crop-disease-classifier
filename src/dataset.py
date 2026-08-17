from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def find_data_root():
    root = Path(__file__).resolve().parent.parent / "data" / "combined"
    if not root.is_dir() or not any(root.iterdir()):
        raise FileNotFoundError(
            f"No combined dataset found at {root}. Run `python3 src/prepare_data.py` first."
        )
    return root


def build_transforms(img_size):
    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def build_dataloaders(
    data_root,
    img_size=224,
    batch_size=32,
    val_split=0.15,
    test_split=0.15,
    num_workers=4,
    seed=42,
):
    train_tf, eval_tf = build_transforms(img_size)

    reference = datasets.ImageFolder(data_root)
    targets = [label for _, label in reference.samples]
    indices = list(range(len(reference)))

    train_idx, holdout_idx = train_test_split(
        indices, test_size=val_split + test_split, stratify=targets, random_state=seed
    )
    holdout_targets = [targets[i] for i in holdout_idx]
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=test_split / (val_split + test_split),
        stratify=holdout_targets,
        random_state=seed,
    )

    train_ds = Subset(datasets.ImageFolder(data_root, transform=train_tf), train_idx)
    val_ds = Subset(datasets.ImageFolder(data_root, transform=eval_tf), val_idx)
    test_ds = Subset(datasets.ImageFolder(data_root, transform=eval_tf), test_idx)

    # Class sizes vary by ~40x across merged sources (e.g. Cassava ~4k/class vs ~100-300/class
    # elsewhere), so sample per-class inversely to frequency rather than plain shuffling --
    # otherwise large classes would dominate gradient updates.
    train_targets = [targets[i] for i in train_idx]
    class_counts = np.bincount(train_targets, minlength=len(reference.classes))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[label] for label in train_targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, reference.classes
