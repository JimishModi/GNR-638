# GNR638 Assignment 2 | Task 1 | seed=42
"""
Experiment 1: Linear Probe Transfer.

Freeze ALL backbone parameters, train only the classification head.
Features are visualized using PCA, t-SNE, and UMAP.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seed import set_seed
from utils.logger_utils import get_logger
from dataloader.dataset_loader import get_dataloaders
from models.model_loader import load_model
from models.freeze_utils import freeze_backbone, count_parameters
from training.train import train_model, evaluate
from evaluation.metrics import print_model_stats
from evaluation.confusion_matrix import plot_confusion_matrix
from analysis.feature_visualization import (
    extract_features, 
    plot_pca, 
    plot_tsne, 
    plot_umap, 
    failure_case_analysis
)


logger = get_logger("linear_probe")


def plot_accuracy_curves(history, model_name, save_dir):
    """Plot training and validation accuracy curves."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_acc"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(epochs, history["train_acc"], label="Train Acc", linewidth=2)
    axes[0].plot(epochs, history["val_acc"], label="Val Acc", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"{model_name} -- Linear Probe Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    axes[1].plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title(f"{model_name} -- Linear Probe Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_linear_probe_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Accuracy curves saved >> {path}")


def run_linear_probe(config):
    """
    Run linear probe experiment for all model architectures.
    """
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    save_dir = os.path.join(config["output_dir"], "linear_probe")
    os.makedirs(save_dir, exist_ok=True)

    # Dataloaders - ensure we use the full dataset for probing
    train_loader, val_loader, class_names = get_dataloaders(
        data_dir=config["data_dir"],
        image_size=config["image_size"],
        val_ratio=config["val_ratio"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        seed=config["seed"],
    )

    results = {}

    for model_name in config["models"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"  LINEAR PROBE -- {model_name}")
        logger.info(f"{'='*60}")

        set_seed(config["seed"])

        # Load model and freeze the backbone
        model, feat_dim = load_model(model_name, num_classes=len(class_names))
        model = freeze_backbone(model, model_name)
        
        # Report efficiency metrics
        print_model_stats(model, model_name)

        # Train linear classifier only
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config["epochs_full"],
            lr=config["learning_rate"],
            device=device,
            save_dir=os.path.join(save_dir, "checkpoints"),
            model_name=f"{model_name}_linear_probe",
            dry_run=config.get("dry_run", False)
        )

        # ── Final Evaluation & Failure Analysis ──
        # Load best weights
        ckpt_path = os.path.join(save_dir, "checkpoints", f"{model_name}_linear_probe_best.pth")
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        
        criterion = nn.CrossEntropyLoss()
        # Collect misclassified samples for analysis
        _, best_val_acc, preds, labels, misclassified = evaluate(
            model, val_loader, criterion, device, return_misclassified=True
        )
        logger.info(f"  Best Validation Accuracy: {best_val_acc:.4f}")

        # ── Visualisations ──
        plots_dir = os.path.join(save_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # 1. Accuracy curves
        plot_accuracy_curves(history, model_name, plots_dir)

        # 2. Confusion Matrix
        plot_confusion_matrix(
            preds, labels, class_names,
            save_path=os.path.join(plots_dir, f"{model_name}_confusion_matrix.png"),
            title=f"{model_name} -- Linear Probe CM"
        )

        # 3. Failure Case Analysis (Top 5 confused pairs)
        cm = confusion_matrix(labels, preds)
        failure_case_analysis(
            cm, misclassified, class_names,
            save_path=os.path.join(plots_dir, f"{model_name}_failure_cases.png")
        )

        # 4. Feature Visualisations (PCA, t-SNE, UMAP)
        logger.info(f"  Extracting features for visualization...")
        features, feature_labels = extract_features(
            model, val_loader, device, dry_run=config.get("dry_run", False)
        )

        plot_pca(
            features, feature_labels, class_names,
            save_path=os.path.join(plots_dir, f"{model_name}_pca.png")
        )
        plot_tsne(
            features, feature_labels, class_names,
            save_path=os.path.join(plots_dir, f"{model_name}_tsne.png")
        )
        plot_umap(
            features, feature_labels, class_names,
            save_path=os.path.join(plots_dir, f"{model_name}_umap.png")
        )

        results[model_name] = {"val_acc": best_val_acc}

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("  LINEAR PROBE RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    for m, r in results.items():
        logger.info(f"  {m:15s}: {r['val_acc']:.4f}")

    return results


if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    run_linear_probe(cfg)
