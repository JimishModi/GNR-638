"""
Experiment 4: Corruption Robustness Evaluation.

Evaluates trained models under image corruptions applied ONLY at
inference time:
  - Gaussian noise (sigma = 0.05, 0.1, 0.2)
  - Motion blur (kernel_size = 15)
  - Brightness shift (factor = 0.3)

Computes:
  - Validation accuracy under each corruption
  - Corruption error = 1 − Acc_corrupted
  - Relative robustness = Acc_corrupted / Acc_clean
"""

import os
import sys
import io
import yaml
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import platform

# Detect OS and set workers accordingly
NUM_WORKERS = 0 if platform.system() == "Windows" else 4
PIN_MEMORY = False if platform.system() == "Windows" else True

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seed import set_seed
from utils.logging import get_logger
from dataloader.dataset_loader import get_aid_datasets
from models.model_loader import load_model
from models.freeze_utils import unfreeze_full
from training.train import train_model, evaluate
from evaluation.corruption_tests import (
    get_corruption_transform,
    evaluate_corrupted,
)
from evaluation.metrics import print_model_stats


logger = get_logger("corruption")


def run_corruption_robustness(config):
    """
    Evaluate corruption robustness for all three models.

    Pipeline:
      1. Train each model (full fine-tuning) on clean data
      2. Evaluate on clean validation set (baseline)
      3. Evaluate on each corruption variant
      4. Compute corruption error and relative robustness
    """
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    save_dir = os.path.join(config["output_dir"], "corruption")
    os.makedirs(save_dir, exist_ok=True)

    # Load data — we need access to the base val dataset for re-wrapping
    train_dataset, val_dataset, class_names = get_aid_datasets(
        data_dir=config["data_dir"],
        image_size=config["image_size"],
        val_ratio=config["val_ratio"],
        seed=config["seed"],
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"],
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=False if NUM_WORKERS == 0 else True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["batch_size"],
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=False if NUM_WORKERS == 0 else True,
    )

    # Define corruption configurations with safe defaults
    corruption_cfg = config.get("corruption", {})
    brightness_factors = corruption_cfg.get("brightness_factors", [0.5, 1.5])
    gaussian_sigmas = corruption_cfg.get("gaussian_noise_sigmas", [0.05, 0.1, 0.2])
    mb_kernel_size = corruption_cfg.get("motion_blur_kernel_size", 15)

    corruption_configs = []
    for sigma in gaussian_sigmas:
        corruption_configs.append({
            "name": f"gaussian_sigma={sigma}",
            "type": "gaussian_noise",
            "params": {"sigma": sigma},
        })
    
    corruption_configs.append({
        "name": f"motion_blur_k={mb_kernel_size}",
        "type": "motion_blur",
        "params": {"kernel_size": mb_kernel_size},
    })
    
    for factor in brightness_factors:
        corruption_configs.append({
            "name": f"brightness_shift={factor}",
            "type": "brightness_shift",
            "params": {"factor": factor},
        })

    all_results = {}

    for model_name in config["models"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"  CORRUPTION ROBUSTNESS -- {model_name}")
        logger.info(f"{'='*60}")

        set_seed(config["seed"])

        # Train model (full fine-tuning)
        model, feat_dim = load_model(model_name,
                                     num_classes=len(class_names))
        model = unfreeze_full(model, model_name)
        print_model_stats(model, model_name)

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config["epochs_full"],
            lr=config["learning_rate"],
            device=device,
            save_dir=os.path.join(save_dir, "checkpoints"),
            model_name=f"{model_name}_corruption",
        )

        # Load best checkpoint
        ckpt = os.path.join(save_dir, "checkpoints",
                            f"{model_name}_corruption_best.pth")
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device))
        model = model.to(device)

        # Clean evaluation (baseline)
        criterion = nn.CrossEntropyLoss()
        _, clean_acc, _, _ = evaluate(model, val_loader, criterion, device)
        logger.info(f"  Clean accuracy: {clean_acc:.4f}")

        # Corrupted evaluations
        model_results = {"clean_acc": clean_acc, "corruptions": {}}

        for corr in corruption_configs:
            transform = get_corruption_transform(
                corr["type"],
                image_size=config["image_size"],
                **corr["params"],
            )
            _, corr_acc, _, _ = evaluate_corrupted(
                model, val_dataset, transform, device,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
            )

            corruption_error = 1.0 - corr_acc
            relative_robustness = corr_acc / clean_acc if clean_acc > 0 else 0

            model_results["corruptions"][corr["name"]] = {
                "acc": corr_acc,
                "corruption_error": corruption_error,
                "relative_robustness": relative_robustness,
            }

            logger.info(
                f"  {corr['name']:30s}: acc={corr_acc:.4f}  "
                f"CE={corruption_error:.4f}  "
                f"RR={relative_robustness:.4f}"
            )

        all_results[model_name] = model_results

    # ── Plots ──
    plots_dir = os.path.join(save_dir, "plots")
    _plot_corruption_results(all_results, config, plots_dir)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("  CORRUPTION ROBUSTNESS SUMMARY")
    logger.info(f"{'='*60}")
    for model_name, res in all_results.items():
        logger.info(f"\n  {model_name} (clean={res['clean_acc']:.4f}):")
        for cname, cres in res["corruptions"].items():
            logger.info(
                f"    {cname:30s}: acc={cres['acc']:.4f}  "
                f"CE={cres['corruption_error']:.4f}  "
                f"RR={cres['relative_robustness']:.4f}"
            )

    return all_results


def _plot_corruption_results(all_results, config, save_dir):
    """Generate corruption robustness plots."""
    os.makedirs(save_dir, exist_ok=True)

    corr_names = list(list(all_results.values())[0]["corruptions"].keys())

    # 1. Accuracy under corruptions (grouped bar)
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(corr_names) + 1)  # +1 for clean
    width = 0.25

    for i, model_name in enumerate(config["models"]):
        res = all_results[model_name]
        accs = [res["clean_acc"]] + [res["corruptions"][c]["acc"]
                                      for c in corr_names]
        ax.bar(x + i * width, accs, width, label=model_name, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(["Clean"] + corr_names, rotation=45, ha="right",
                       fontsize=8)
    ax.set_ylabel("Validation Accuracy", fontsize=12)
    ax.set_title("Corruption Robustness: Accuracy", fontsize=14,
                 fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "corruption_accuracy.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Relative robustness
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, model_name in enumerate(config["models"]):
        res = all_results[model_name]
        rr = [res["corruptions"][c]["relative_robustness"] for c in corr_names]
        ax.bar(np.arange(len(corr_names)) + i * width, rr, width,
               label=model_name, alpha=0.85)

    ax.set_xticks(np.arange(len(corr_names)) + width)
    ax.set_xticklabels(corr_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Relative Robustness (Acc_corr / Acc_clean)", fontsize=12)
    ax.set_title("Relative Robustness Under Corruptions", fontsize=14,
                 fontweight="bold")
    ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="Perfect")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "relative_robustness.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Corruption plots saved >> {save_dir}")


if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_corruption_robustness(config)
