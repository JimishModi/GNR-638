"""
Confusion matrix plotting utility.

Uses sklearn.metrics.confusion_matrix and matplotlib to generate
a colour-coded confusion matrix heatmap.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(preds, labels, class_names,
                          save_path="outputs/plots/confusion_matrix.png",
                          title="Confusion Matrix",
                          figsize=(14, 12)):
    """
    Plot and save a confusion matrix.

    Args:
        preds: List/array of predicted class indices.
        labels: List/array of true class indices.
        class_names: List of class name strings.
        save_path: File path for the saved figure.
        title: Plot title.
        figsize: Figure size tuple.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cm = confusion_matrix(labels, preds)
    # Normalise rows to get percentages
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n_classes = len(class_names)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)

    # Annotate cells with raw counts
    thresh = cm_norm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, f"{cm[i, j]}",
                    ha="center", va="center", fontsize=5,
                    color="white" if cm_norm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved → {save_path}")
