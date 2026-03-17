"""
Evaluation metrics: accuracy and FLOPs/MACs estimation.

FLOPs/MACs are estimated via forward hooks on Conv2d and Linear layers.
This avoids external dependencies (fvcore, thop) per assignment rules.
"""

import torch
import torch.nn as nn
import numpy as np


def compute_accuracy(preds, labels):
    """Compute top-1 accuracy from prediction and label lists."""
    preds = np.array(preds)
    labels = np.array(labels)
    return (preds == labels).mean()


# --- MACs for a Conv2d: kernel_ops x output_elements. -------------------------

class _FLOPCounter:
    """Accumulates MACs from Conv2d and Linear layers."""

    def __init__(self):
        self.total_macs = 0
        self.hooks = []

    def _conv_hook(self, module, input, output):
        """MACs for a Conv2d: kernel_ops × output_elements."""
        batch_size = output.shape[0]
        out_channels, out_h, out_w = output.shape[1], output.shape[2], output.shape[3]
        in_channels = module.in_channels // module.groups
        kernel_ops = in_channels * module.kernel_size[0] * module.kernel_size[1]
        # Each output element needs kernel_ops multiply-adds
        macs = kernel_ops * out_channels * out_h * out_w
        self.total_macs += macs

    def _linear_hook(self, module, input, output):
        """MACs for a Linear layer: in_features × out_features."""
        macs = module.in_features * module.out_features
        self.total_macs += macs

    def register(self, model):
        """Register hooks on all Conv2d and Linear layers."""
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                h = module.register_forward_hook(self._conv_hook)
                self.hooks.append(h)
            elif isinstance(module, nn.Linear):
                h = module.register_forward_hook(self._linear_hook)
                self.hooks.append(h)

    def remove(self):
        """Remove all hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def estimate_flops_macs(model, input_size=(3, 224, 224), device="cpu"):
    """
    Estimate FLOPs and MACs for a model.

    FLOPs ≈ 2 × MACs (each multiply-add = 2 floating point operations).

    Args:
        model: PyTorch model.
        input_size: (C, H, W) input tensor size.
        device: Device for the dummy forward pass.

    Returns:
        macs (int), flops (int)
    """
    counter = _FLOPCounter()
    model_device = next(model.parameters()).device
    model.eval()
    counter.register(model)

    dummy = torch.randn(1, *input_size).to(model_device)
    with torch.no_grad():
        model(dummy)

    counter.remove()
    macs = counter.total_macs
    flops = 2 * macs
    return macs, flops


def print_model_stats(model, model_name: str,
                      input_size=(3, 224, 224)):
    """
    Print model statistics: parameters, MACs, FLOPs.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                          if p.requires_grad)
    macs, flops = estimate_flops_macs(model, input_size)

    print(f"\n{'-'*60}")
    print(f"  Model: {model_name}")
    print(f"  Total Parameters:     {total_params:>15,}")
    print(f"  Trainable Parameters: {trainable_params:>15,}")
    print(f"  MACs:                 {macs:>15,}")
    print(f"  FLOPs:                {flops:>15,}")
    print(f"{'-'*60}\n")

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "macs": macs,
        "flops": flops,
    }
