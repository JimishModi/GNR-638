"""
Optimizer and scheduler factory.

Provides standard Adam optimizer and cosine annealing scheduler.
"""

import torch.optim as optim


def get_optimizer(model, lr: float = 0.001,
                  weight_decay: float = 1e-4):
    """
    Create an Adam optimizer over only the trainable parameters.

    Args:
        model: PyTorch model.
        lr: Learning rate.
        weight_decay: L2 regularization weight.

    Returns:
        torch.optim.Adam instance.
    """
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    return optimizer


def get_scheduler(optimizer, epochs: int):
    """
    Create a cosine annealing learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule.
        epochs: Total number of training epochs (T_max).

    Returns:
        torch.optim.lr_scheduler.CosineAnnealingLR instance.
    """
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    return scheduler
