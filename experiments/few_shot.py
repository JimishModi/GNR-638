# GNR638 Assignment 2 | Task 3 | seed=42
"""
Experiment 3: Few-Shot Learning Analysis.

Analyzes model performance across different data regimes:
- 100% Training Data (30 epochs)
- 20% Training Data (20 epochs)
- 5% Training Data (20 epochs)

Compares two training modes:
- Mode A: Linear Probe (Backbone frozen, head trained)
- Mode B: Full Fine-tuning (All layers trainable)

Calculates Performance Drop (Δ) and Train-Val Gap.
"""

import os
import sys
import csv
import yaml
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seed import set_seed
from utils.logging import get_logger
from dataloader.dataset_loader import get_dataloaders
from models.model_loader import load_model
from models.freeze_utils import freeze_backbone, unfreeze_full
from training.train import train_model


logger = get_logger("few_shot")


def run_few_shot(config):
    """
    Run the few-shot analysis: 3 models x 3 regimes x 2 modes = 18 runs.
    """
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    save_dir = os.path.join(config["output_dir"], "few_shot")
    os.makedirs(save_dir, exist_ok=True)

    fractions = config["few_shot_fractions"]   # [1.0, 0.2, 0.05]
    modes = ["Mode_A", "Mode_B"]               # A: Frozen, B: Full
    
    results_records = []
    
    # Stratified split once for the whole experiment
    # get_dataloaders handles the internal splitting and fractioning
    
    for model_name in config["models"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"  FEW-SHOT ANALYSIS — {model_name}")
        logger.info(f"{'='*60}")

        model_results = {}

        for mode in modes:
            logger.info(f"\n--- Strategy: {mode} ---")
            mode_accs = {} # frac -> best_val_acc

            for frac in fractions:
                frac_pct = int(frac * 100)
                logger.info(f"  Regime: {frac_pct}% data")
                
                # Deterministic sampling for every run
                set_seed(config["seed"])
                
                # Fetch data
                train_loader, val_loader, class_names = get_dataloaders(
                    data_dir=config["data_dir"],
                    image_size=config["image_size"],
                    val_ratio=config["val_ratio"],
                    batch_size=config["batch_size"],
                    num_workers=config["num_workers"],
                    seed=config["seed"],
                    train_fraction=frac,
                )

                # Initialize model
                model, _ = load_model(model_name, num_classes=len(class_names))
                
                # Apply Mode
                if mode == "Mode_A":
                    model = freeze_backbone(model, model_name)
                else:
                    model = unfreeze_full(model, model_name)

                # Set epochs per regime
                epochs = config["epochs_full"] if frac >= 1.0 else config["epochs_few_shot"]

                # Train
                history = train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=epochs,
                    lr=config["learning_rate"],
                    device=device,
                    save_dir=os.path.join(save_dir, "checkpoints"),
                    model_name=f"{model_name}_{mode}_{frac_pct}pct",
                    dry_run=config.get("dry_run", False)
                )

                best_val_acc = history["best_val_acc"]
                final_train_acc = history["train_acc"][-1]
                gap = final_train_acc - history["val_acc"][-1]
                
                mode_accs[frac] = best_val_acc

                # Record results
                results_records.append({
                    "model": model_name,
                    "mode": mode,
                    "regime": f"{frac_pct}%",
                    "epochs": epochs,
                    "train_acc": final_train_acc,
                    "val_acc": best_val_acc,
                    "train_val_gap": gap
                })

            # Calculate Delta = (Acc_100 - Acc_5) / Acc_100 only if all required keys exist
            if 1.0 in mode_accs and 0.05 in mode_accs:
                acc_100 = mode_accs[1.0]
                acc_5 = mode_accs[0.05]
                delta = (acc_100 - acc_5) / acc_100 if acc_100 > 0 else 0
            else:
                delta = None
                logger.info(f"  [INFO] Skipping Delta calculation for {mode} (missing regimes)")
            
            # Update records with Delta
            for record in results_records:
                if record["model"] == model_name and record["mode"] == mode:
                    record["delta"] = delta

        # Generate plot for this model
        _plot_model_few_shot(model_name, results_records, save_dir)

    # Save CSV
    csv_path = os.path.join(config["output_dir"], "fewshot_results.csv")
    df = pd.DataFrame(results_records)
    df.to_csv(csv_path, index=False)
    logger.info(f"\n  Few-shot results exported to >> {csv_path}")

    return results_records


def _plot_model_few_shot(model_name: str, records: list, save_dir: str):
    """
    Generate grouped bar chart for a single model comparing Mode A and Mode B.
    """
    model_records = [r for r in records if r["model"] == model_name]
    if not model_records:
        return

    # Prepare data for plotting
    df = pd.DataFrame(model_records)
    regimes = ["5%", "20%", "100%"]
    
    # Sort or filter to ensure regime order
    available_regimes = df["regime"].unique().tolist()
    regimes_to_plot = [r for r in regimes if r in available_regimes]

    if len(regimes_to_plot) < 2:
        logger.info(f"  [INFO] Skipping few-shot plot for {model_name}: insufficient regimes completed.")
        return

    mode_a_vals = [df[(df["mode"] == "Mode_A") & (df["regime"] == r)]["val_acc"].values[0] for r in regimes_to_plot]
    mode_b_vals = [df[(df["mode"] == "Mode_B") & (df["regime"] == r)]["val_acc"].values[0] for r in regimes_to_plot]

    x = np.arange(len(regimes_to_plot))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, mode_a_vals, width, label='Mode A (Head Only)', color='skyblue', alpha=0.8)
    rects2 = ax.bar(x + width/2, mode_b_vals, width, label='Mode B (Full FT)', color='salmon', alpha=0.8)

    ax.set_xlabel('Data Regime', fontsize=12)
    ax.set_ylabel('Best Validation Accuracy', fontsize=12)
    ax.set_title(f'Few-Shot: {model_name} (Mode A vs Mode B)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    # Add accuracy labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"fewshot_plot_{model_name}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved >> {plot_path}")


if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    run_few_shot(cfg)
