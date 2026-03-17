"""
Regenerate fine-tuning plots from logs and checkpoints.
"""

import os
import re
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import yaml
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader.dataset_loader import get_dataloaders
from models.model_loader import load_model
from models.freeze_utils import (
    freeze_backbone,
    unfreeze_last_block,
    unfreeze_full,
    selective_unfreeze,
    count_parameters,
)
from training.train import evaluate, cleanup_gpu_memory

STRATEGIES = {
    "linear_probe":   freeze_backbone,
    "last_block":     unfreeze_last_block,
    "full_finetune":  unfreeze_full,
    "selective_20pct": selective_unfreeze,
}

# Known accuracies provided by user (overriding logs if necessary)
KNOWN_ACCURACIES = {
    "resnet50": {
        "linear_probe": 0.8831,
        "last_block": 0.9646,
        "full_finetune": 0.9632,
        "selective_20pct": 0.9632,
    },
    "densenet121": {
        "linear_probe": 0.9141,
        "last_block": 0.9531,
        "full_finetune": 0.9589,
        "selective_20pct": 0.9582,
    }
}

def parse_logs(log_path):
    """Parse train.log to get training loss history and accuracies."""
    history = defaultdict(lambda: defaultdict(list))
    # Pattern: 2026-03-16 11:11:17,097 | train | INFO | [resnet50_linear_probe] Epoch 15/30 | Train Loss: 0.5987  Acc: 0.8466 | Val Loss: 0.6032  Acc: 0.8615 | LR: 0.000501 | Time: 115.1s
    pattern = re.compile(r"\[(?P<model>\w+)_(?P<strategy>\w+)\] Epoch \d+/\d+ \| Train Loss: (?P<train_loss>[\d\.]+) .* Acc: (?P<val_acc>[\d\.]+) \|")
    
    if not os.path.exists(log_path):
        print(f"Warning: Log file not found at {log_path}")
        return history

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                model = match.group("model")
                strategy = match.group("strategy")
                if strategy in STRATEGIES:
                    history[model][strategy].append(float(match.group("train_loss")))
    
    return history

def get_grad_norms(model, loader, device):
    """Calculate gradient norms on a single batch."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    
    model.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.data.norm(2).item()
    
    return grad_norms

def main():
    parser = argparse.ArgumentParser(description="Regenerate Fine-Tuning Plots")
    parser.add_argument("--data_path", type=str, default="./dataloader/AID")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    save_dir = os.path.join(args.output_dir, "fine_tuning")
    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Parse Logs
    log_path = os.path.join(args.output_dir, "logs", "train.log")
    log_history = parse_logs(log_path)

    # 2. Setup Data
    train_loader, val_loader, class_names = get_dataloaders(
        data_dir=args.data_path,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    all_results = {}
    models = ["resnet50", "densenet121", "efficientnet_b0"]

    for model_name in models:
        model_results = {}
        for strategy_name, freeze_fn in STRATEGIES.items():
            ckpt_path = os.path.join(save_dir, "checkpoints", f"{model_name}_{strategy_name}_best.pth")
            if not os.path.exists(ckpt_path):
                # EfficientNet only has 2 strategies
                if model_name == "efficientnet_b0" and strategy_name not in ["linear_probe", "last_block"]:
                    continue
                print(f"Skipping {model_name} {strategy_name}: Checkpoint not found at {ckpt_path}")
                continue

            print(f"Processing {model_name} - {strategy_name}...")
            model, _ = load_model(model_name, num_classes=len(class_names))
            
            # Match freeze pattern
            if strategy_name == "selective_20pct":
                model = freeze_fn(model, model_name, target_pct=0.20)
            else:
                model = freeze_fn(model, model_name)
            
            total, trainable = count_parameters(model)
            unfrozen_pct = 100.0 * trainable / total
            
            # Load weights
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.to(device)

            # Evaluate or use known accuracy
            if model_name in KNOWN_ACCURACIES and strategy_name in KNOWN_ACCURACIES[model_name]:
                best_val_acc = KNOWN_ACCURACIES[model_name][strategy_name]
            else:
                _, best_val_acc, _, _ = evaluate(model, val_loader, nn.CrossEntropyLoss(), device)
            
            # Calculate gradient norms (representative)
            grad_norms = get_grad_norms(model, train_loader, device)

            model_results[strategy_name] = {
                "unfrozen_pct": unfrozen_pct,
                "best_val_acc": best_val_acc,
                "history": {"train_loss": log_history.get(model_name, {}).get(strategy_name, [])},
                "grad_norms": [grad_norms] # Wrap in list to match expected format in plot functions if needed
            }
            cleanup_gpu_memory()

        all_results[model_name] = model_results

    # 3. Plotting
    _plot_acc_vs_unfrozen(all_results, plots_dir)
    _plot_gradient_norms(all_results, plots_dir)
    _plot_training_loss(all_results, plots_dir)

    print(f"\nAll plots regenerated in {plots_dir}")

def _plot_acc_vs_unfrozen(all_results, save_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, strats in all_results.items():
        # Filter strategies that were actually run
        available_strats = [s for s in STRATEGIES if s in strats]
        pcts = [strats[s]["unfrozen_pct"] for s in available_strats]
        accs = [strats[s]["best_val_acc"] for s in available_strats]
        
        # Sort by pcts for a clean line
        sorted_indices = np.argsort(pcts)
        pcts = np.array(pcts)[sorted_indices]
        accs = np.array(accs)[sorted_indices]
        
        ax.plot(pcts, accs, marker="o", linewidth=2, label=model_name)

        # Annotate strategy names
        for pct, acc, sname in zip(pcts, accs, np.array(available_strats)[sorted_indices]):
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

def _plot_gradient_norms(all_results, save_dir):
    for model_name, strats in all_results.items():
        n_strats = len(strats)
        cols = 2
        rows = (n_strats + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(16, 6 * rows))
        axes = axes.flatten()

        for idx, (strategy_name, res) in enumerate(strats.items()):
            ax = axes[idx]
            last_grad = res["grad_norms"][-1]
            if last_grad:
                sorted_layers = sorted(last_grad.items(),
                                       key=lambda x: x[1], reverse=True)[:20]
                names = [n.replace(".", "\n", 1) for n, _ in sorted_layers]
                values = [v for _, v in sorted_layers]
                ax.barh(range(len(names)), values, alpha=0.8)
                ax.set_yticks(range(len(names)))
                # Adjust fontsize for readability
                ax.set_yticklabels(names, fontsize=6)
                ax.invert_yaxis()
            ax.set_title(f"{strategy_name}", fontsize=10)
            ax.set_xlabel("Gradient Norm", fontsize=8)

        # Remove empty subplot if odd number of strategies
        if n_strats % 2 != 0:
            fig.delaxes(axes[-1])

        fig.suptitle(f"{model_name} -- Gradient Norms (checkpoint state)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        path = os.path.join(save_dir, f"{model_name}_gradient_norms.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

def _plot_training_loss(all_results, save_dir):
    for model_name, strats in all_results.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for strategy_name, res in strats.items():
            losses = res["history"]["train_loss"]
            if losses:
                epochs = range(1, len(losses) + 1)
                ax.plot(epochs, losses, linewidth=2, label=strategy_name)

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

if __name__ == "__main__":
    main()
