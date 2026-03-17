# GNR638 Assignment 2 | Task 2 | seed=42
"""
Layer-wise feature probing.

Extracts feature representations from early, middle, and final layers
of each model architecture, then trains linear classifiers (logistic
regression) on each to assess layer-wise feature quality.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# --- Layer name definitions for each architecture ---------------------------
# Rationale for layer selection:
# - Early: Captures low-level primitives (edges, textures, colors). 
#          Usually after the first major block group.
# - Middle: Captures mid-level patterns (parts, motifs, complex textures). 
#           Usually at the 50% depth mark of the feature extractor.
# - Final: Captures high-level semantic features (objects, scene layouts).
#          The last layer before the global average pooling.

LAYER_CONFIGS = {
    "resnet50": {
        # ResNet50 has 4 stages. stage 1 (layer1) is early, stage 3 (layer3) is middle, stage 4 (layer4) is final.
        "early":  "layer1",          # 256 channels, local patterns
        "middle": "layer2",          # 512 channels, intermediate textures
        "final":  "layer4",          # 2048 channels, high-level scene features
    },
    "densenet121": {
        # DenseNet121 has 4 dense blocks. We pick 1, 2, and 4.
        "early":  "features.denseblock1",
        "middle": "features.denseblock2",
        "final":  "features.denseblock4",
    },
    "efficientnet_b0": {
        # EfficientNet-B0 has 7 stages of MBConv blocks.
        "early":  "blocks.0",       # First stage (MBConv 1x1, 3x3)
        "middle": "blocks.4",       # Middle stage (approx 50% depth)
        "final":  "blocks.6",       # Final stage before head (blocks[-1])
    },
}


def get_layer_hooks(model, model_name: str):
    """
    Register forward hooks on early/mid/final layers.

    Returns:
        hooks_dict: {depth_name: {"handle": hook_handle,
                                   "features": []}}
    """
    layer_names = LAYER_CONFIGS.get(model_name)
    if layer_names is None:
        raise ValueError(f"No layer config for {model_name}")

    hooks_dict = {}

    for depth_name, layer_name in layer_names.items():
        # Navigate to the target module safely
        try:
            module = model
            for part in layer_name.split("."):
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
        except (AttributeError, IndexError) as e:
            # Safety check fallback: if layer name is invalid, warn and skip
            print(f"  [WARNING] Target layer {layer_name} not found in {model_name}. Error: {e}")
            continue

        # Create a storage container
        storage = {"features": []}

        def make_hook(store):
            def hook_fn(mod, inp, out):
                # Ensure we apply Global Average Pooling if the output is spatial
                # This ensures linear probe sees a fixed-length vector (rank 1)
                if isinstance(out, (tuple, list)):
                    out = out[0] # Handle models returning multiple tensors
                
                if out.dim() == 4:
                    # Spatial features (B, C, H, W) -> (B, C)
                    pooled = nn.functional.adaptive_avg_pool2d(out, 1)
                    pooled = pooled.view(pooled.size(0), -1)
                elif out.dim() == 3:
                    # Sequence features (B, N, C) -> (B, C)
                    pooled = out.mean(dim=1)
                else:
                    # Already (B, C) or (B, N)
                    pooled = out
                store["features"].append(pooled.detach().cpu())
            return hook_fn

        handle = module.register_forward_hook(make_hook(storage))
        hooks_dict[depth_name] = {
            "handle": handle,
            "features": storage["features"],
            "layer_name": layer_name,
        }

    return hooks_dict


def extract_layerwise_features(model, loader, hooks_dict, device, dry_run=False):
    """
    Run a forward pass over the loader and collect features from hooks.

    Returns:
        features_dict: {depth_name: np.ndarray (N, D)}
        labels: np.ndarray (N,)
    """
    model.eval()
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            _ = model(images)
            all_labels.append(labels.numpy())

            if dry_run:
                break

    labels = np.concatenate(all_labels, axis=0)

    features_dict = {}
    for depth_name, info in hooks_dict.items():
        if not info["features"]:
            print(f"  [WARNING] No features collected for {depth_name}!")
            continue
        feats = torch.cat(info["features"], dim=0).numpy()
        features_dict[depth_name] = feats
        # Clear stored features to free memory
        info["features"].clear()

    return features_dict, labels


def remove_hooks(hooks_dict):
    """Remove all registered forward hooks."""
    for info in hooks_dict.values():
        info["handle"].remove()


def train_linear_probe(features, labels, num_classes=30):
    """
    Train a logistic regression classifier on extracted features.
    
    Training accuracy serves as a proxy for the 'linear separability' 
    of the representation at a given depth.
    """
    # Validate we have multiple classes
    unique_classes = np.unique(labels)
    if len(unique_classes) < 2:
        print(f"  [WARNING] Skipping probe: only {len(unique_classes)} class(es) in data. Need at least 2.")
        return 0.0, None, None

    # Standardise features for better convergence
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    clf = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(features_scaled, labels)
    accuracy = clf.score(features_scaled, labels)
    return accuracy, clf, scaler


def evaluate_linear_probe(clf, scaler, features, labels):
    """Evaluate a trained linear probe on new features."""
    if clf is None or scaler is None:
        return 0.0, None
    features_scaled = scaler.transform(features)
    accuracy = clf.score(features_scaled, labels)
    preds = clf.predict(features_scaled)
    return accuracy, preds


def plot_accuracy_vs_depth(results, save_path="outputs/layerwise_acc.png"):
    """
    Plot validation accuracy vs layer depth for each model.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    depths = ["early", "middle", "final"]
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, accs in results.items():
        # Ensure we have all depths
        available_depths = [d for d in depths if d in accs]
        values = [accs[d] for d in available_depths]
        ax.plot(available_depths, values, marker="o", linewidth=2, label=model_name)

    ax.set_xlabel("Layer Depth", fontsize=12)
    ax.set_ylabel("Linear Probe Accuracy", fontsize=12)
    ax.set_title("Feature Quality vs. Network Depth", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Layer-wise accuracy plot saved >> {save_path}")


def plot_feature_norms(norms_dict, save_path="outputs/feature_norms.png"):
    """
    Plot feature L2 norm statistics across layers.
    Helpful to see how feature variance changes with depth.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    depths = ["early", "middle", "final"]
    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(depths))
    width = 0.25
    for i, (model_name, norms) in enumerate(norms_dict.items()):
        means = [norms[d][0] for d in depths if d in norms]
        stds = [norms[d][1] for d in depths if d in norms]
        current_x = x[:len(means)]
        ax.bar(current_x + i * width, means, width, yerr=stds,
               label=model_name, alpha=0.8, capsize=3)

    ax.set_xticks(x + width)
    ax.set_xticklabels(depths)
    ax.set_xlabel("Layer Depth", fontsize=12)
    ax.set_ylabel("Feature L2 Norm (Mean +/- Std)", fontsize=12)
    ax.set_title("Feature Norm Evolution Across Layers", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Feature norms plot saved >> {save_path}")


def plot_layerwise_pca(features_dict, labels, class_names,
                       save_dir="outputs/layerwise_pca",
                       model_name="model"):
    """
    PCA 10x10 or 30x30 subset visualizations for each layer depth.
    """
    os.makedirs(save_dir, exist_ok=True)

    n_classes = len(class_names)
    cmap = plt.cm.get_cmap("tab20", n_classes)

    for depth_name, features in features_dict.items():
        pca = PCA(n_components=2)
        proj = pca.fit_transform(features)

        fig, ax = plt.subplots(figsize=(10, 8))
        for cls_idx in range(n_classes):
            mask = labels == cls_idx
            if not mask.any(): continue
            ax.scatter(proj[mask, 0], proj[mask, 1],
                       c=[cmap(cls_idx)], label=class_names[cls_idx],
                       s=15, alpha=0.7)

        var1 = pca.explained_variance_ratio_[0] * 100
        var2 = pca.explained_variance_ratio_[1] * 100
        ax.set_title(f"{model_name} | {depth_name} layer | PCA",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel(f"PC1 ({var1:.1f}%)")
        ax.set_ylabel(f"PC2 ({var2:.1f}%)")
        ax.legend(fontsize=6, ncol=3, loc="best", markerscale=1.5)
        plt.tight_layout()
        path = os.path.join(save_dir, f"{model_name}_{depth_name}_pca.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  PCA plot saved >> {path}")
