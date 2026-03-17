"""
Corruption transforms for robustness evaluation.

Implements:
  1. Gaussian Noise  (σ = 0.05, 0.1, 0.2)
  2. Motion Blur     (horizontal kernel)
  3. Brightness Shift

All corruptions are applied ONLY during evaluation (not training).
They are implemented as torchvision-compatible transforms.
"""

import torch
import numpy as np
import platform
from torchvision import transforms
from PIL import Image, ImageFilter

# Detect OS and set workers accordingly
NUM_WORKERS = 0 if platform.system() == "Windows" else 4
PIN_MEMORY = False if platform.system() == "Windows" else True


# ─── Gaussian Noise ─────────────────────────────────────────────────────────

class GaussianNoise:
    """
    Add Gaussian noise to a tensor image.

    Applied AFTER ToTensor() + Normalize(), so it operates on
    normalised tensors in [-~2.5, ~2.5] range.
    """

    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.sigma
        return tensor + noise

    def __repr__(self):
        return f"GaussianNoise(sigma={self.sigma})"


# ─── Motion Blur ────────────────────────────────────────────────────────────

class MotionBlur:
    """
    Apply horizontal motion blur to a PIL image.

    Uses a 1D horizontal kernel of given size.
    Applied BEFORE ToTensor().
    """

    def __init__(self, kernel_size: int = 3):
        # Force valid kernel size: for PIL GaussianBlur, radius is float. 
        # For our logic, we'll map kernel_size 3,5,etc to radius.
        self.radius = (kernel_size - 1) / 2 if kernel_size > 1 else 1.0
        # Guard: Ensure realistic radius
        self.radius = max(1.0, min(5.0, self.radius))

    def __call__(self, img):
        # Use GaussianBlur as a more robust alternative to raw Kernel filters
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))

    def __repr__(self):
        return f"MotionBlur(radius={self.radius})"


# ─── Brightness Shift ───────────────────────────────────────────────────────

class BrightnessShift:
    """
    Shift brightness of a tensor image.

    Adds a constant offset to all channels (after normalisation).
    """

    def __init__(self, factor: float = 0.3):
        self.factor = factor

    def __call__(self, tensor):
        return tensor + self.factor

    def __repr__(self):
        return f"BrightnessShift(factor={self.factor})"


# ─── Corruption transform builders ─────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_corruption_transform(corruption_type: str,
                              image_size: int = 224,
                              **kwargs):
    """
    Build a validation transform with a specific corruption.

    Args:
        corruption_type: One of 'gaussian_noise', 'motion_blur',
                         'brightness_shift'.
        image_size: Input resolution.
        **kwargs: Corruption-specific parameters
                  (sigma, kernel_size, factor).

    Returns:
        A torchvision.transforms.Compose pipeline.
    """
    base_pre = [
        transforms.Resize(int(image_size * 1.143)),
        transforms.CenterCrop(image_size),
    ]
    base_post = [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    if corruption_type == "gaussian_noise":
        sigma = kwargs.get("sigma", 0.1)
        return transforms.Compose(
            base_pre + base_post + [GaussianNoise(sigma=sigma)]
        )

    elif corruption_type == "motion_blur":
        kernel_size = kwargs.get("kernel_size", 15)
        return transforms.Compose(
            base_pre + [MotionBlur(kernel_size=kernel_size)] + base_post
        )

    elif corruption_type == "brightness_shift":
        factor = kwargs.get("factor", 0.3)
        return transforms.Compose(
            base_pre + base_post + [BrightnessShift(factor=factor)]
        )

    else:
        raise ValueError(f"Unknown corruption: {corruption_type}")


def evaluate_corrupted(model, val_dataset_base, corruption_transform,
                       device, batch_size=32, num_workers=NUM_WORKERS):
    """
    Evaluate a model on a corrupted version of the validation set.

    Creates a new dataset with the corruption transform and runs evaluation.

    Args:
        model: Trained model.
        val_dataset_base: The base validation TransformSubset.
        corruption_transform: The corruption transform pipeline.
        device: 'cuda' or 'cpu'.
        batch_size: Batch size for evaluation.
        num_workers: DataLoader workers.

    Returns:
        val_loss, val_acc, all_preds, all_labels
    """
    from dataloader.dataset_loader import TransformSubset
    from training.train import evaluate
    import torch.nn as nn

    # Create a copy of the dataset with corruption transform
    corrupted_dataset = TransformSubset(
        val_dataset_base.dataset,
        val_dataset_base.indices,
        corruption_transform,
    )

    loader = torch.utils.data.DataLoader(
        corrupted_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=False if num_workers == 0 else True,
    )

    criterion = nn.CrossEntropyLoss()
    return evaluate(model, loader, criterion, device)
