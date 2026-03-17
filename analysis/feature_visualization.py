# GNR638 Assignment 2 | Task 1 | seed=42 | Python 3.11
"""
Feature embedding visualisation using PCA, t-SNE, and UMAP.

Extracts features from the penultimate layer of a model and projects
them to 2D for qualitative analysis of transfer quality.
Also provides failure case analysis by identifying top confused pairs.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Requirements: pip install umap-learn
try:
    import umap
except ImportError:
    # We will ensure this is in requirements.txt
    umap = None


def extract_features(model, loader, device, dry_run=False):
    """
    Extract feature embeddings from the penultimate layer (global pool output).

    Uses a forward hook on the global pool layer to extract features 
    AFTER global average pooling but BEFORE the classification head.

    Args:
        model: A timm model.
        loader: DataLoader providing (images, labels).
        device: 'cuda' or 'cpu'.
        dry_run: If True, only process one batch.

    Returns:
        features: np.ndarray of shape (N, D)
        labels:   np.ndarray of shape (N,)
    """
    model.eval()
    all_features = []
    all_labels = []

    features_buffer = {}

    def hook_fn(module, input, output):
        # Result of global pool is typically (B, D) or (B, D, 1, 1)
        # We ensure it is flattened to (B, D)
        if output.dim() > 2:
            output = output.view(output.size(0), -1)
        features_buffer["feat"] = output.detach().cpu()

    # Register hook on the global pool layer
    # Most timm CNNs have a 'global_pool' attribute
    if hasattr(model, "global_pool"):
        handle = model.global_pool.register_forward_hook(hook_fn)
    else:
        # Fallback for models with different naming conventions
        found = False
        for name, mod in model.named_modules():
            if isinstance(mod, nn.AdaptiveAvgPool2d):
                handle = mod.register_forward_hook(hook_fn)
                found = True
                break
        if not found:
            raise RuntimeError("Could not find global pool layer for feature extraction.")

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            _ = model(images)
            
            feat = features_buffer["feat"]
            all_features.append(feat.numpy())
            all_labels.append(labels.numpy())

            if dry_run:
                break

    handle.remove()

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return features, labels


def plot_pca(features, labels, class_names,
             save_path="outputs/pca.png",
             title="PCA -- Feature Embeddings"):
    """
    Project features to 2D via PCA and plot a scatter diagram.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    pca = PCA(n_components=2)
    proj = pca.fit_transform(features)

    fig, ax = plt.subplots(figsize=(12, 10))
    n_classes = len(class_names)
    cmap = plt.cm.get_cmap("tab20", n_classes)

    for cls_idx in range(n_classes):
        mask = labels == cls_idx
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=[cmap(cls_idx)], label=class_names[cls_idx],
                   s=10, alpha=0.7)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(fontsize=6, ncol=3, loc="best", markerscale=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  PCA plot saved >> {save_path}")


def plot_tsne(features, labels, class_names,
              save_path="outputs/tsne.png",
              title="t-SNE -- Feature Embeddings",
              perplexity=30):
    """
    Project features to 2D via t-SNE and plot a scatter diagram.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n_samples = len(features)
    dynamic_perplexity = min(perplexity, max(1, n_samples - 1))
    
    tsne = TSNE(n_components=2, perplexity=dynamic_perplexity,
                random_state=42, n_iter=1000, init='pca', learning_rate='auto')
    proj = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(12, 10))
    n_classes = len(class_names)
    cmap = plt.cm.get_cmap("tab20", n_classes)

    for cls_idx in range(n_classes):
        mask = labels == cls_idx
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=[cmap(cls_idx)], label=class_names[cls_idx],
                   s=10, alpha=0.7)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend(fontsize=6, ncol=3, loc="best", markerscale=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  t-SNE plot saved >> {save_path}")


def plot_umap(features, labels, class_names,
              save_path="outputs/umap.png",
              title="UMAP -- Feature Embeddings"):
    """
    Project features to 2D via UMAP and plot a scatter diagram.
    
    Requirements: n_components=2, n_neighbors=15, min_dist=0.1, random_state=42.
    """
    if umap is None:
        print("  [WARNING] umap-learn not installed. Skipping UMAP plot.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n_samples = len(features)
    n_neighbors = min(15, max(2, n_samples - 1))

    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
    proj = reducer.fit_transform(features)

    fig, ax = plt.subplots(figsize=(12, 10))
    n_classes = len(class_names)
    cmap = plt.cm.get_cmap("tab20", n_classes)

    for cls_idx in range(n_classes):
        mask = labels == cls_idx
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=[cmap(cls_idx)], label=class_names[cls_idx],
                   s=10, alpha=0.7)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP dimension 1")
    ax.set_ylabel("UMAP dimension 2")
    ax.legend(fontsize=6, ncol=3, loc="best", markerscale=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  UMAP plot saved >> {save_path}")


def failure_case_analysis(cm, misclassified_samples, class_names, 
                          save_path="outputs/failure_cases.png",
                          top_n=5, imgs_per_pair=5):
    """
    Analyze and visualize top confused class pairs.

    Args:
        cm: 30x30 confusion matrix (numpy).
        misclassified_samples: List of (image_tensor, true_idx, pred_idx).
        class_names: List of class name strings.
        save_path: Output file path.
        top_n: Number of most confused pairs to show.
        imgs_per_pair: Number of images to show per confused pair.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 1. Find top N confused class pairs (off-diagonal maxima)
    cm_off_diag = cm.copy()
    np.fill_diagonal(cm_off_diag, 0)
    
    # Get indices of top N values in flattened array
    flat_indices = np.argsort(cm_off_diag.ravel())[::-1][:top_n]
    pairs = [np.unravel_index(idx, cm.shape) for idx in flat_indices]
    
    # Filter 0 confusion pairs (if any)
    pairs = [p for p in pairs if cm[p[0], p[1]] > 0]
    num_rows = len(pairs)
    
    if num_rows == 0:
        print("  [INFO] No misclassifications found for confusion analysis.")
        return

    fig, axes = plt.subplots(num_rows, imgs_per_pair, figsize=(imgs_per_pair * 2.5, num_rows * 3))
    if num_rows == 1:
        axes = axes[np.newaxis, :]

    for row_idx, (true_idx, pred_idx) in enumerate(pairs):
        # 2. Extract up to 5 misclassified images for this pair
        relevant_imgs = [s for s in misclassified_samples if s[1] == true_idx and s[2] == pred_idx]
        relevant_imgs = relevant_imgs[:imgs_per_pair]
        
        true_name = class_names[true_idx]
        pred_name = class_names[pred_idx]
        
        for col_idx in range(imgs_per_pair):
            ax = axes[row_idx, col_idx]
            if col_idx < len(relevant_imgs):
                img_tensor, _, _ = relevant_imgs[col_idx]
                
                # Undo ImageNet normalization for visualization
                img = img_tensor.permute(1, 2, 0).cpu().numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                
                ax.imshow(img)
                if col_idx == 0:
                    ax.set_ylabel(f"True: {true_name}\nPred: {pred_name}", 
                                  fontsize=10, fontweight="bold")
            else:
                ax.axis('off')
            
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0 and col_idx == 0:
                ax.set_title("Top confused examples", loc='left', pad=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Failure case analysis saved >> {save_path}")
