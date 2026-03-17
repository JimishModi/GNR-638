"""
AID Dataset Loader.

Loads the Aerial Images Dataset (AID) in ImageFolder format,
performs a stratified 80/20 train-val split, and provides
a subsampling utility for few-shot experiments.
"""

import os
from collections import defaultdict

import numpy as np
import torch
import platform
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Detect OS and set workers accordingly (Windows needs 0 to avoid WinError 1455)
NUM_WORKERS = 0 if platform.system() == "Windows" else 4
PIN_MEMORY = False if platform.system() == "Windows" else True


# ----- ImageNet normalisation stats -----
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(image_size: int = 224):
    """
    Return standard training and validation transforms.

    Training: random resized crop + horizontal flip + normalisation.
    Validation: resize + centre crop + normalisation.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.143)),   # ≈ 256 for 224
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_transform, val_transform


def get_aid_datasets(data_dir: str,
                     image_size: int = 224,
                     val_ratio: float = 0.20,
                     seed: int = 42):
    """
    Load the AID dataset and split into train/val sets with stratification.

    Args:
        data_dir: Path to the AID folder (one sub-folder per class).
        image_size: Input image resolution.
        val_ratio: Fraction of data used for validation.
        seed: Random seed for the split.

    Returns:
        train_dataset, val_dataset, class_names
    """
    train_transform, val_transform = get_transforms(image_size)

    # Load the full dataset (transforms applied on-the-fly via wrapper)
    full_dataset = datasets.ImageFolder(root=data_dir)
    class_names = full_dataset.classes
    targets = np.array(full_dataset.targets)

    # --- Stratified split ---
    rng = np.random.RandomState(seed)
    train_indices, val_indices = [], []

    for cls_idx in range(len(class_names)):
        cls_mask = np.where(targets == cls_idx)[0]
        rng.shuffle(cls_mask)
        n_val = int(len(cls_mask) * val_ratio)
        val_indices.extend(cls_mask[:n_val].tolist())
        train_indices.extend(cls_mask[n_val:].tolist())

    # Wrap subsets with their own transforms
    train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
    val_dataset = TransformSubset(full_dataset, val_indices, val_transform)

    return train_dataset, val_dataset, class_names


class TransformSubset(torch.utils.data.Dataset):
    """
    A dataset subset that applies a specific transform.

    This is needed because ImageFolder shares one transform object;
    we need separate train/val transforms on non-overlapping index sets.
    """

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset.samples[self.indices[idx]]
        from PIL import Image
        img = Image.open(img).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    @property
    def targets(self):
        """Expose targets so downstream code can do stratified subsampling."""
        return [self.dataset.targets[i] for i in self.indices]


def get_subset_dataset(train_dataset, fraction: float, seed: int = 42):
    """
    Sub-sample the training set per class (for few-shot experiments).

    Args:
        train_dataset: A TransformSubset or dataset with .targets attribute.
        fraction: Fraction of training samples to keep per class.
        seed: Random seed for reproducibility.

    Returns:
        A Subset of the training dataset.
    """
    if fraction >= 1.0:
        return train_dataset

    rng = np.random.RandomState(seed)
    targets = np.array(train_dataset.targets)
    unique_classes = np.unique(targets)

    subset_indices = []
    for cls_idx in unique_classes:
        cls_mask = np.where(targets == cls_idx)[0]
        n_keep = max(1, int(len(cls_mask) * fraction))     # At least 1 sample
        rng.shuffle(cls_mask)
        subset_indices.extend(cls_mask[:n_keep].tolist())

    return Subset(train_dataset, subset_indices)


def get_dataloaders(data_dir: str,
                    image_size: int = 224,
                    val_ratio: float = 0.20,
                    batch_size: int = 32,
                    num_workers: int = 4,
                    seed: int = 42,
                    train_fraction: float = 1.0):
    """
    Convenience function that returns ready-to-use DataLoaders.

    Args:
        data_dir: Path to AID root directory.
        image_size: Input resolution.
        val_ratio: Validation split ratio.
        batch_size: Batch size.
        num_workers: Number of DataLoader workers.
        seed: Random seed.
        train_fraction: Fraction of training set to use (for few-shot).

    Returns:
        train_loader, val_loader, class_names
    """
    train_dataset, val_dataset, class_names = get_aid_datasets(
        data_dir, image_size, val_ratio, seed
    )

    if train_fraction < 1.0:
        train_dataset = get_subset_dataset(train_dataset, train_fraction, seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=False if NUM_WORKERS == 0 else True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=False if NUM_WORKERS == 0 else True,
        drop_last=False,
    )
    return train_loader, val_loader, class_names
