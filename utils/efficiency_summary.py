# GNR638 Assignment 2 | Task 5 | seed=42
"""
Model Efficiency Summary Utility.

Computes Total Parameters, Trainable Parameters, MACs, and FLOPs 
for ResNet50, DenseNet121, and EfficientNet-B0 using the thop library.
Generates a console table and saves results to CSV and PNG.
"""

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_loader import load_model

# Try to import thop
try:
    from thop import profile
except ImportError:
    print("  [ERROR] 'thop' library not found. Falling back to internal estimator.")
    from evaluation.metrics import estimate_flops_macs as internal_profile
    profile = None


def generate_efficiency_summary():
    """
    Load models, compute complexity metrics, and save summaries.
    """
    models_to_test = ["resnet50", "densenet121", "efficientnet_b0"]
    input_size = (1, 3, 224, 224)
    dummy_input = torch.randn(input_size)
    
    results = []
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print(f"{'Model':<20} | {'Params (M)':<12} | {'Trainable (M)':<15} | {'MACs (G)':<10} | {'FLOPs (G)':<10}")
    print("-" * 80)

    for name in models_to_test:
        # Load model for 30-class task
        model, _ = load_model(name, num_classes=30, pretrained=True)
        model.eval()

        # Count parameters
        total_p = sum(p.numel() for p in model.parameters()) / 1e6
        train_p = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

        # Compute MACs/FLOPs
        if profile:
            # thop.profile returns MACs (labeled as 'flops') and params
            macs, _ = profile(model, inputs=(dummy_input, ), verbose=False)
            macs_g = macs / 1e9
            flops_g = 2 * macs_g  # Standard approximation: 1 MAC = 2 FLOPs
        else:
            # Fallback to internal counter
            macs, flops = internal_profile(model, input_size=(3, 224, 224))
            macs_g = macs / 1e9
            flops_g = flops / 1e9

        print(f"{name:<20} | {total_p:<12.2f} | {train_p:<15.2f} | {macs_g:<10.3f} | {flops_g:<10.3f}")
        
        results.append({
            "Model": name,
            "Total Params (M)": round(total_p, 2),
            "Trainable (M)": round(train_p, 2),
            "MACs (G)": round(macs_g, 3),
            "FLOPs (G)": round(flops_g, 3)
        })

    print("="*80 + "\n")

    # ── Save CSV ──
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "efficiency_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Efficiency summary saved to → {csv_path}")

    # ── Save PNG Table ──
    _save_table_as_png(df, os.path.join(output_dir, "efficiency_summary.png"))


def _save_table_as_png(df, save_path):
    """Render the dataframe as a matplotlib table and save to PNG."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')
    
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colColours=["#f2f2f2"] * len(df.columns)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title("Model Efficiency Metrics Summary", fontsize=12, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Efficiency table PNG saved to → {save_path}")


if __name__ == "__main__":
    generate_efficiency_summary()
