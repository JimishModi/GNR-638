"""
Freeze / Unfreeze utilities for transfer learning strategies.

Provides functions for:
  1. Freezing the entire backbone (linear probe)
  2. Unfreezing the last block only
  3. Full unfreezing (fine-tuning)
  4. Selective unfreezing (~20% of backbone parameters)
"""

import torch.nn as nn


def count_parameters(model):
    """
    Count total and trainable parameters.

    Returns:
        total_params, trainable_params
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def freeze_backbone(model, model_name: str):
    """
    Freeze ALL parameters, then unfreeze only the classification head.
    This implements the linear probe strategy.
    """
    # Step 1: Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: Unfreeze classifier head
    head = _get_head(model, model_name)
    for param in head.parameters():
        param.requires_grad = True

    total, trainable = count_parameters(model)
    print(f"[freeze_backbone] Total: {total:,} | Trainable: {trainable:,} "
          f"({100 * trainable / total:.2f}%)")
    return model


def unfreeze_last_block(model, model_name: str):
    """
    Freeze everything except the last architectural block + head.

    Last block definitions:
        - ResNet50:        model.layer4
        - DenseNet121:     model.features.denseblock4 + model.features.norm5
        - EfficientNet-B0: model.blocks[-1] (last block group)
    """
    # Start by freezing everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last block
    if model_name == "resnet50":
        for param in model.layer4.parameters():
            param.requires_grad = True
    elif model_name == "densenet121":
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True
        if hasattr(model.features, "norm5"):
            for param in model.features.norm5.parameters():
                param.requires_grad = True
    elif model_name == "efficientnet_b0":
        # timm EfficientNet stores blocks in model.blocks
        for param in model.blocks[-1].parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Always unfreeze the head
    head = _get_head(model, model_name)
    for param in head.parameters():
        param.requires_grad = True

    total, trainable = count_parameters(model)
    print(f"[unfreeze_last_block] Total: {total:,} | Trainable: {trainable:,} "
          f"({100 * trainable / total:.2f}%)")
    return model


def unfreeze_full(model, model_name: str):
    """
    Unfreeze ALL parameters (full fine-tuning).
    """
    for param in model.parameters():
        param.requires_grad = True

    total, trainable = count_parameters(model)
    print(f"[unfreeze_full] Total: {total:,} | Trainable: {trainable:,} "
          f"({100 * trainable / total:.2f}%)")
    return model


def selective_unfreeze(model, model_name: str, target_pct: float = 0.20):
    """
    Selectively unfreeze layers from the END of the backbone until
    approximately `target_pct` (20%) of backbone parameters are trainable.

    The head is always unfrozen independently.

    Prints which layers are unfrozen and the actual percentage.
    """
    # Step 1: Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: Unfreeze the head first
    head = _get_head(model, model_name)
    head_param_names = set()
    for name, param in model.named_parameters():
        for hname, _ in head.named_parameters():
            if name.endswith(hname):
                param.requires_grad = True
                head_param_names.add(name)

    # Step 3: Collect backbone parameters (everything except head)
    backbone_params = []
    for name, param in model.named_parameters():
        if name not in head_param_names:
            backbone_params.append((name, param))

    total_backbone = sum(p.numel() for _, p in backbone_params)
    target_trainable = int(total_backbone * target_pct)

    # Step 4: Unfreeze from the END until we reach target
    unfrozen_count = 0
    unfrozen_layers = []

    for name, param in reversed(backbone_params):
        if unfrozen_count >= target_trainable:
            break
        param.requires_grad = True
        unfrozen_count += param.numel()
        unfrozen_layers.append(name)

    # Report
    actual_pct = 100 * unfrozen_count / total_backbone if total_backbone > 0 else 0
    print(f"\n[selective_unfreeze] Target: {target_pct*100:.0f}% of backbone")
    print(f"  Backbone params: {total_backbone:,}")
    print(f"  Unfrozen backbone params: {unfrozen_count:,} ({actual_pct:.1f}%)")
    print(f"  Unfrozen layers ({len(unfrozen_layers)}):")
    for layer_name in unfrozen_layers:
        print(f"    - {layer_name}")

    total, trainable = count_parameters(model)
    print(f"  Total model: {total:,} | Trainable: {trainable:,} "
          f"({100 * trainable / total:.2f}%)\n")
    return model


def _get_head(model, model_name: str):
    """Internal helper to retrieve the classifier head."""
    if model_name == "resnet50":
        return model.fc
    elif model_name == "densenet121":
        return model.classifier
    elif model_name == "efficientnet_b0":
        return model.classifier
    else:
        if hasattr(model, "classifier"):
            return model.classifier
        elif hasattr(model, "fc"):
            return model.fc
        raise ValueError(f"Cannot find head for {model_name}")
