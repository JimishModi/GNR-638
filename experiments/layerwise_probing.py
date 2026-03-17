"""
Experiment 5: Layer-wise Feature Probing.

Extracts feature representations from early, middle, and final layers
of each model and trains separate linear classifiers (logistic regression)
on each representation.

Generates:
  - Validation accuracy vs layer depth plot
  - Feature norm statistics
  - PCA 2D plots using a fixed subset (30 classes × 30 samples/class)
"""

import os
import sys
import yaml
import torch
import numpy as np

import platform

# Detect OS and set workers accordingly
NUM_WORKERS = 0 if platform.system() == "Windows" else 4
PIN_MEMORY = False if platform.system() == "Windows" else True

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seed import set_seed
from utils.logger_utils import get_logger
from dataloader.dataset_loader import get_aid_datasets, get_subset_dataset
from models.model_loader import load_model
from models.freeze_utils import freeze_backbone
from evaluation.metrics import print_model_stats
from analysis.probing import (
    get_layer_hooks,
    extract_layerwise_features,
    remove_hooks,
    train_linear_probe,
    evaluate_linear_probe,
    plot_accuracy_vs_depth,
    plot_feature_norms,
    plot_layerwise_pca,
)


logger = get_logger("layerwise_probing")


def _get_probing_subset(dataset, class_names, samples_per_class=30, seed=42):
    """
    Get a fixed subset of the dataset for PCA visualization.

    Selects exactly `samples_per_class` samples per class.
    30 classes × 30 samples = 900 samples total.
    """
    rng = np.random.RandomState(seed)
    targets = np.array(dataset.targets)
    indices = []

    for cls_idx in range(len(class_names)):
        cls_mask = np.where(targets == cls_idx)[0]
        n_select = min(samples_per_class, len(cls_mask))
        selected = rng.choice(cls_mask, size=n_select, replace=False)
        indices.extend(selected.tolist())

    from torch.utils.data import Subset
    return Subset(dataset, indices)


def run_layerwise_probing(config):
    """
    Run layer-wise feature probing for all three models.

    For each model:
      1. Load pretrained model (backbone frozen)
      2. Extract features at early/mid/final layers
      3. Train linear probes on training features
      4. Evaluate on validation features
      5. Generate accuracy-vs-depth plots, norm stats, PCA
    """
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    save_dir = os.path.join(config["output_dir"], "layerwise_probing")
    os.makedirs(save_dir, exist_ok=True)

    # Load full datasets
    train_dataset, val_dataset, class_names = get_aid_datasets(
        data_dir=config["data_dir"],
        image_size=config["image_size"],
        val_ratio=config["val_ratio"],
        seed=config["seed"],
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"],
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=False if NUM_WORKERS == 0 else True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["batch_size"],
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=False if NUM_WORKERS == 0 else True,
    )

    # PCA subset: 30 classes × 30 samples/class
    # PCA subset: 30 classes × 30 samples/class (minimum 5 for dry run)
    samples_per_class = max(5, config.get("probing", {}).get("samples_per_class", 30))
    pca_subset = _get_probing_subset(
        val_dataset, class_names,
        samples_per_class=samples_per_class, seed=config["seed"],
    )
    pca_loader = torch.utils.data.DataLoader(
        pca_subset, batch_size=config["batch_size"],
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=False if NUM_WORKERS == 0 else True,
    )

    accuracy_results = {}   # model_name → {depth → val_acc}
    norms_results = {}      # model_name → {depth → (mean, std)}

    for model_name in config["models"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"  LAYER-WISE PROBING -- {model_name}")
        logger.info(f"{'='*60}")

        set_seed(config["seed"])

        # Load pretrained model (frozen backbone)
        model, feat_dim = load_model(model_name,
                                     num_classes=len(class_names))
        model = freeze_backbone(model, model_name)
        model = model.to(device)
        print_model_stats(model, model_name)

        # ── Extract training features ──
        logger.info("  Extracting training features...")
        hooks = get_layer_hooks(model, model_name)
        dry_run = config.get("dry_run", False)
        train_features, train_labels = extract_layerwise_features(
            model, train_loader, hooks, device, dry_run=dry_run
        )
        remove_hooks(hooks)

        # ── Extract validation features ──
        logger.info("  Extracting validation features...")
        hooks = get_layer_hooks(model, model_name)
        val_features, val_labels = extract_layerwise_features(
            model, val_loader, hooks, device, dry_run=dry_run
        )
        remove_hooks(hooks)

        # ── Extract PCA subset features ──
        logger.info("  Extracting PCA subset features...")
        hooks = get_layer_hooks(model, model_name)
        pca_features, pca_labels = extract_layerwise_features(
            model, pca_loader, hooks, device, dry_run=dry_run
        )
        remove_hooks(hooks)

        # ── Train and evaluate linear probes ──
        model_accs = {}
        model_norms = {}

        for depth_name in ["early", "middle", "final"]:
            logger.info(f"\n  --- {depth_name} layer ---")

            # Feature norms
            feats = train_features[depth_name]
            norms = np.linalg.norm(feats, axis=1)
            mean_norm = norms.mean()
            std_norm = norms.std()
            model_norms[depth_name] = (mean_norm, std_norm)
            logger.info(f"    Feature dim: {feats.shape[1]}")
            logger.info(f"    Feature norm: {mean_norm:.2f} +/- {std_norm:.2f}")

            # Train linear probe
            train_acc, clf, scaler = train_linear_probe(
                train_features[depth_name], train_labels,
                num_classes=len(class_names),
            )
            logger.info(f"    Train accuracy: {train_acc:.4f}")

            # Evaluate on validation set
            if clf is not None and scaler is not None:
                val_acc, _ = evaluate_linear_probe(
                    clf, scaler, val_features[depth_name], val_labels
                )
            else:
                val_acc = 0.0
                logger.warning(f"    Skipping val eval for {depth_name}: probe not fitted.")

            logger.info(f"    Val accuracy:   {val_acc:.4f}")

            model_accs[depth_name] = val_acc

        accuracy_results[model_name] = model_accs
        norms_results[model_name] = model_norms

        # PCA plots for this model
        plot_layerwise_pca(
            pca_features, pca_labels, class_names,
            save_dir=os.path.join(save_dir, "plots", "pca"),
            model_name=model_name,
        )

    # ── Cross-model plots ──
    plots_dir = os.path.join(save_dir, "plots")

    # Accuracy vs depth
    plot_accuracy_vs_depth(accuracy_results,
                           save_path=os.path.join(plots_dir, "accuracy_vs_depth.png"))

    # Feature norms
    plot_feature_norms(norms_results,
                       save_path=os.path.join(plots_dir, "feature_norms.png"))

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("  LAYER-WISE PROBING SUMMARY")
    logger.info(f"{'='*60}")
    for model_name in config["models"]:
        logger.info(f"\n  {model_name}:")
        for depth in ["early", "middle", "final"]:
            acc = accuracy_results[model_name][depth]
            mn, sd = norms_results[model_name][depth]
            logger.info(f"    {depth:8s}: val_acc={acc:.4f}  "
                         f"norm={mn:.2f}+/-{sd:.2f}")

    return accuracy_results, norms_results


if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_layerwise_probing(config)
