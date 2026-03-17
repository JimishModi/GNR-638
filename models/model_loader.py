# GNR638 Assignment 2 | Task 4 | seed=42
"""
Model loader using timm.

Loads pretrained ImageNet models (ResNet50, DenseNet121, EfficientNet-B0)
and replaces the classification head for 30-class AID.
Ensures the head is EXACTLY one nn.Linear layer with NO dropout.
"""

import timm
import torch.nn as nn


# Mapping of short names used in config -> timm model identifiers
TIMM_MODEL_NAMES = {
    "resnet50": "resnet50",
    "densenet121": "densenet121",
    "efficientnet_b0": "efficientnet_b0",
}


def load_model(model_name: str, num_classes: int = 30,
               pretrained: bool = True):
    """
    Load a pretrained model from timm and replace the classifier head.

    Args:
        model_name: One of 'resnet50', 'densenet121', 'efficientnet_b0'.
        num_classes: Number of output classes (30 for AID).
        pretrained: Whether to load ImageNet pretrained weights.

    Returns:
        model: The loaded model with a new classification head.
        feature_dim: Dimensionality of the feature embedding (before head).
    """
    timm_name = TIMM_MODEL_NAMES.get(model_name, model_name)

    # Hard Rule: No dropout.
    # We use a safer approach for drop_rate as not all models support proj_drop_rate.
    try:
        model = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=0.0,
        )
    except TypeError:
        # Fallback if drop_rate is not supported as a keyword (unlikely but safe)
        model = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=num_classes
        )

    # Determine the feature dimension of the backbone
    feature_dim = model.num_features

    # Explicitly replace head to ensure it's STRICTLY one nn.Linear layer. 
    # This removes any dropout layers that might be part of the original head block.
    if model_name == "resnet50":
        model.fc = nn.Linear(feature_dim, num_classes)
    elif model_name == "densenet121" or model_name == "efficientnet_b0":
        model.classifier = nn.Linear(feature_dim, num_classes)
    else:
        # Generic fallback
        if hasattr(model, "classifier"):
            model.classifier = nn.Linear(feature_dim, num_classes)
        elif hasattr(model, "fc"):
            model.fc = nn.Linear(feature_dim, num_classes)

    return model, feature_dim


def get_classifier_head(model, model_name: str):
    """
    Return a reference to the classifier head module.
    """
    if model_name == "resnet50":
        return model.fc
    elif model_name == "densenet121" or model_name == "efficientnet_b0":
        return model.classifier
    else:
        if hasattr(model, "classifier"):
            return model.classifier
        elif hasattr(model, "fc"):
            return model.fc
        else:
            raise ValueError(f"Unknown classifier head for {model_name}")
