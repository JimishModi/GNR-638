"""
Logging utilities for the training pipeline.

Provides a pre-configured logger and an AverageMeter for tracking
running statistics (loss, accuracy, etc.) during training.
"""

import logging
import os
import sys


def get_logger(name: str, log_dir: str = "outputs/logs") -> logging.Logger:
    """
    Create a logger that writes to both console and a log file.

    Args:
        name: Logger name (usually the module or experiment name).
        log_dir: Directory where log files are saved.

    Returns:
        Configured logging.Logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter("[%(asctime)s %(name)s] %(message)s",
                                    datefmt="%H:%M:%S")
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"{name}.log"), mode="a"
    )
    file_handler.setLevel(logging.INFO)
    file_fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    return logger


class AverageMeter:
    """
    Computes and stores the running average and current value.

    Usage:
        meter = AverageMeter("loss")
        for batch_loss in losses:
            meter.update(batch_loss, n=batch_size)
        print(meter)
    """

    def __init__(self, name: str = "meter"):
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"
