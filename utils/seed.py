"""
Deterministic seeding for full reproducibility.

Sets seeds for Python, NumPy, PyTorch (CPU + CUDA) and enables
deterministic CuDNN behaviour so that every run is identical.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic CuDNN (slightly slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Python hash seed for reproducible hashing
    os.environ["PYTHONHASHSEED"] = str(seed)
