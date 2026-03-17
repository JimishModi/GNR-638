"""
Experiment 2: Fine-Tuning Strategies.

Compares four transfer learning strategies:
  1. Linear probe         -- only classifier head trained
  2. Last block fine-tune -- last architectural block + head
  3. Full fine-tuning     -- all parameters trainable
  4. Selective unfreezing  -- ~20% of backbone parameters

Generates:
  - Accuracy vs percentage of unfrozen parameters
  - Gradient norm statistics across layers
  - Training loss vs epoch (all strategies overlaid)
"""

import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seed import set_seed
from utils.logger_utils import get_logger
from dataloader.dataset_loader import get_dataloaders
from models.model_loader import load_model
from models.freeze_utils import (
    freeze_backbone,
    unfreeze_last_block,
    unfreeze_full,
    selective_unfreeze,
    count_parameters,
)
from training.train import train_model
from evaluation.metrics import print_model_stats


logger = get_logger("fine_tuning")

# Strategy registry: name → freeze function
STRATEGIES = {
    # "linear_probe":   freeze_backbone,
    # "last_block":     unfreeze_last_block,
    "full_finetune":  unfreeze_full,
    "selective_20pct": selective_unfreeze,
}


def run_fine_tuning(config):
    """
    Run the four fine-tuning strategies for all three models.

    For each (model, strategy) pair we:
      1. Apply the freeze/unfreeze pattern
      2. Train for config["epochs_full"] epochs
      3. Record training loss, validation accuracy, gradient norms
    """
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    save_dir = os.path.join(config["output_dir"], "fine_tuning")
    os.makedirs(save_dir, exist_ok=True)

    train_loader, val_loader, class_names = get_dataloaders(
        data_dir=config["data_dir"],
        image_size=config["image_size"],
        val_ratio=config["val_ratio"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        seed=config["seed"],
    )

    all_results = {}  # model_name → {strategy_name → result}

    for model_name in config["models"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"  FINE-TUNING -- {model_name}")
        logger.info(f"{'='*60}")

        model_results = {}

        for strategy_name, freeze_fn in STRATEGIES.items():
            logger.info(f"\n--- Strategy: {strategy_name} ---")
            set_seed(config["seed"])

            model, feat_dim = load_model(model_name,
                                         num_classes=len(class_names))

            # Apply the freeze/unfreeze strategy
            if strategy_name == "selective_20pct":
                model = freeze_fn(model, model_name, target_pct=0.20)
            else:
                model = freeze_fn(model, model_name)

            total, trainable = count_parameters(model)
            unfrozen_pct = 100.0 * trainable / total

            stats = print_model_stats(model, model_name)

            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=config["epochs_full"],
                lr=config["learning_rate"],
                device=device,
                save_dir=os.path.join(save_dir, "checkpoints"),
                model_name=f"{model_name}_{strategy_name}",
            )

            model_results[strategy_name] = {
                "history": history,
                "total_params": total,
                "trainable_params": trainable,
                "unfrozen_pct": unfrozen_pct,
                "best_val_acc": history["best_val_acc"],
            }

            logger.info(f"  {strategy_name}: unfrozen={unfrozen_pct:.1f}%, "
                         f"best_val_acc={history['best_val_acc']:.4f}")

        all_results[model_name] = model_results

    # ── Generate plots ──

    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Accuracy vs percentage of unfrozen parameters
    _plot_acc_vs_unfrozen(all_results, plots_dir)

    # 2. Gradient norm statistics
    _plot_gradient_norms(all_results, plots_dir)

    # 3. Training loss vs epoch
    _plot_training_loss(all_results, plots_dir)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("  FINE-TUNING SUMMARY")
    logger.info(f"{'='*60}")
    for model_name, strats in all_results.items():
        logger.info(f"\n  {model_name}:")
        for sname, res in strats.items():
            logger.info(f"    {sname:20s}: unfrozen={res['unfrozen_pct']:.1f}%  "
                         f"val_acc={res['best_val_acc']:.4f}")

    return all_results


def _plot_acc_vs_unfrozen(all_results, save_dir):
    """Plot accuracy vs percentage of unfrozen parameters."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, strats in all_results.items():
        pcts = [strats[s]["unfrozen_pct"] for s in STRATEGIES]
        accs = [strats[s]["best_val_acc"] for s in STRATEGIES]
        ax.plot(pcts, accs, marker="o", linewidth=2, label=model_name)

        # Annotate strategy names on first model
        if model_name == list(all_results.keys())[0]:
            for pct, acc, sname in zip(pcts, accs, STRATEGIES.keys()):
                ax.annotate(sname, (pct, acc), fontsize=7,
                           textcoords="offset points", xytext=(5, 5))

    ax.set_xlabel("Unfrozen Parameters (%)", fontsize=12)
    ax.set_ylabel("Best Validation Accuracy", fontsize=12)
    ax.set_title("Accuracy vs Percentage of Unfrozen Parameters",
                 fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "acc_vs_unfrozen.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Accuracy vs unfrozen plot saved >> {path}")


def _plot_gradient_norms(all_results, save_dir):
    """Plot gradient norm statistics across layers for each model/strategy."""
    for model_name, strats in all_results.items():
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, (strategy_name, res) in enumerate(strats.items()):
            ax = axes[idx]
            # Use last epoch's gradient norms as representative
            if res["history"]["grad_norms"]:
                last_grad = res["history"]["grad_norms"][-1]
                if last_grad:
                    # Get top-20 layers by norm for readability
                    sorted_layers = sorted(last_grad.items(),
                                           key=lambda x: x[1], reverse=True)[:20]
                    names = [n.replace(".", "\n", 1) for n, _ in sorted_layers]
                    values = [v for _, v in sorted_layers]
                    ax.barh(range(len(names)), values, alpha=0.8)
                    ax.set_yticks(range(len(names)))
                    ax.set_yticklabels(names, fontsize=5)
                    ax.invert_yaxis()
            ax.set_title(f"{strategy_name}", fontsize=10)
            ax.set_xlabel("Gradient Norm", fontsize=8)

        fig.suptitle(f"{model_name} -- Gradient Norms (last epoch)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(save_dir, f"{model_name}_gradient_norms.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Gradient norms plot saved >> {path}")


def _plot_training_loss(all_results, save_dir):
    """Plot training loss vs epoch for all strategies (one plot per model)."""
    for model_name, strats in all_results.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        for strategy_name, res in strats.items():
            epochs = range(1, len(res["history"]["train_loss"]) + 1)
            ax.plot(epochs, res["history"]["train_loss"],
                    linewidth=2, label=strategy_name)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Training Loss", fontsize=12)
        ax.set_title(f"{model_name} -- Training Loss by Strategy",
                     fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(save_dir, f"{model_name}_training_loss.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Training loss plot saved >> {path}")


if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_fine_tuning(config)
