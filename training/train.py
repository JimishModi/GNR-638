"""
Training loop with gradient-norm tracking.

Provides:
  - train_one_epoch(): Single-epoch training with gradient norms.
  - evaluate(): Full validation pass.
  - train_model(): Complete training loop with logging and checkpointing.
"""

import os
import sys
import io
import time
import gc
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from utils.logger_utils import AverageMeter, get_logger


def cleanup_gpu_memory():
    """Force GPU memory cleanup between runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


logger = get_logger("train")


def train_one_epoch(model, loader, criterion, optimizer, device, dry_run=False):
    """
    Train for one epoch.

    Returns:
        epoch_loss, epoch_acc, gradient_norms (dict: layer_name -> list of norms)
    """
    model.train()
    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("acc")
    grad_norms = defaultdict(list)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Record gradient norms per named parameter
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms[name].append(param.grad.data.norm(2).item())

        optimizer.step()

        # Accuracy
        _, preds = outputs.max(1)
        correct = preds.eq(labels).sum().item()
        batch_size = labels.size(0)

        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(correct / batch_size, batch_size)

    # Average gradient norms across batches
    avg_grad_norms = {k: np.mean(v) for k, v in grad_norms.items()}

    return loss_meter.avg, acc_meter.avg, avg_grad_norms


@torch.no_grad()
def evaluate(model, loader, criterion, device, dry_run=False, return_misclassified=False):
    """Full validation pass with GPU memory safety."""
    cleanup_gpu_memory()
    model.eval()
    loss_meter = AverageMeter("val_loss")
    acc_meter = AverageMeter("val_acc")
    all_preds = []
    all_labels = []
    misclassified = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = outputs.max(1)
            correct = preds.eq(labels)
            batch_size = labels.size(0)

            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(correct.sum().item() / batch_size, batch_size)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            if return_misclassified:
                mis_mask = ~correct
                if mis_mask.any():
                    if len(misclassified) < 500:
                        for i in range(batch_size):
                            if mis_mask[i]:
                                misclassified.append((
                                    images[i].cpu(), 
                                    labels[i].item(), 
                                    preds[i].item()
                                ))

        if dry_run:
            break

    if return_misclassified:
        return loss_meter.avg, acc_meter.avg, all_preds, all_labels, misclassified
    return loss_meter.avg, acc_meter.avg, all_preds, all_labels


def train_model(model, train_loader, val_loader,
                epochs, lr, device,
                save_dir="outputs/checkpoints",
                model_name="model",
                weight_decay=1e-4,
                dry_run=False):
    """Training loop with GPU memory harvesting."""
    from training.optimizer import get_optimizer, get_scheduler
    cleanup_gpu_memory()

    os.makedirs(save_dir, exist_ok=True)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = get_scheduler(optimizer, epochs)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "grad_norms": [],
        "best_val_acc": 0.0,
    }

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        start = time.time()

        # Training phase with cache clearing
        model.train()
        loss_meter = AverageMeter("loss")
        acc_meter = AverageMeter("acc")
        grad_norms = defaultdict(list)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norms[name].append(param.grad.data.norm(2).item())

            optimizer.step()

            _, preds = outputs.max(1)
            correct = preds.eq(labels).sum().item()
            batch_size = labels.size(0)

            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(correct / batch_size, batch_size)

            # Clear cache periodically to prevent fragmentation (Task 12-D)
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

        train_loss, train_acc = loss_meter.avg, acc_meter.avg
        avg_grad_norms = {k: np.mean(v) for k, v in grad_norms.items()}

        # Validation phase
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device, dry_run=dry_run
        )
        scheduler.step()

        elapsed = time.time() - start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["grad_norms"].append(avg_grad_norms)

        logger.info(
            f"[{model_name}] Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history["best_val_acc"] = best_val_acc
            ckpt_path = os.path.join(save_dir, f"{model_name}_best.pth")
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"  [OK] Saved best checkpoint (val_acc={best_val_acc:.4f})")

    cleanup_gpu_memory()
    return history
