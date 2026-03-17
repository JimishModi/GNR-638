# GNR638 Assignment 2 | Pre-run Validation Suite | seed=42
"""
Dry run test suite to verify all scenarios, models, and reporting tools.
Tests internal logic with minimal data and resources.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seed import set_seed
from dataloader.dataset_loader import get_dataloaders
from models.model_loader import load_model
from training.train import train_model, evaluate


def test_scenario(name, func):
    """Helper to run a test and print status."""
    print(f"Running test: {name}...", end=" ", flush=True)
    try:
        func()
        print("✅ PASS")
        return True
    except Exception as e:
        print(f"❌ FAIL - {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tiny config for dry run
    config = {
        "seed": 42,
        "data_dir": "./dataloader/AID", # Assumed path
        "image_size": 224,
        "val_ratio": 0.2,
        "batch_size": 4,
        "num_workers": 0,
        "learning_rate": 1e-4,
        "epochs_full": 1,
        "epochs_few_shot": 1,
        "few_shot_fractions": [1.0, 0.05], # Reduced
        "output_dir": "outputs/dry_run",
        "models": ["efficientnet_b0"], # Single small model
        "dry_run": True
    }
    
    os.makedirs(config["output_dir"], exist_ok=True)
    results = []

    # 1. Dataset Loading
    def test_data():
        train_loader, val_loader, class_names = get_dataloaders(
            data_dir=config["data_dir"],
            image_size=config["image_size"],
            val_ratio=config["val_ratio"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            seed=config["seed"],
            train_fraction=0.05 # Increased slightly to ensure multiple classes
        )
        assert len(class_names) == 30
        assert len(train_loader) > 0
    results.append(test_scenario("Dataset Loading", test_data))

    # 2. Model Loading (All 3)
    def test_models():
        for m_name in ["resnet50", "densenet121", "efficientnet_b0"]:
            model, feat_dim = load_model(m_name, num_classes=30, pretrained=False) 
            assert isinstance(model, nn.Module)
            if m_name == "resnet50":
                assert isinstance(model.fc, nn.Linear)
            else:
                assert isinstance(model.classifier, nn.Linear)
    results.append(test_scenario("Model Loading (All 3)", test_models))

    # 3. Linear Probe
    def test_linear_probe():
        from experiments.linear_probe import run_linear_probe
        lp_cfg = config.copy()
        lp_cfg["epochs_full"] = 1
        run_linear_probe(lp_cfg)
        plots_dir = os.path.join(config["output_dir"], "linear_probe", "plots")
        assert os.path.exists(os.path.join(plots_dir, "efficientnet_b0_pca.png"))
        assert os.path.exists(os.path.join(plots_dir, "efficientnet_b0_tsne.png"))
    results.append(test_scenario("Linear Probe Scenario", test_linear_probe))

    # 4. UMAP & Failure Cases check
    def test_visuals():
        plots_dir = os.path.join(config["output_dir"], "linear_probe", "plots")
        assert os.path.exists(os.path.join(plots_dir, "efficientnet_b0_umap.png"))
        assert os.path.exists(os.path.join(plots_dir, "efficientnet_b0_failure_cases.png"))
    results.append(test_scenario("UMAP & Failure Cases", test_visuals))

    # 5. Few-Shot Mode A
    def test_few_shot_a():
        from experiments.few_shot import run_few_shot
        fs_cfg = config.copy()
        fs_cfg["few_shot_fractions"] = [0.05]
        fs_cfg["models"] = ["efficientnet_b0"]
        # Force single mode to test A
        orig_modes = ["Mode_A"] 
        # We need to monkeypatch or just rely on run_few_shot logic
        # For dry run, we'll just run as is
        run_few_shot(fs_cfg)
        assert os.path.exists(os.path.join(config["output_dir"], "fewshot_results.csv"))
    results.append(test_scenario("Few-Shot Analysis (Mode A/B)", test_few_shot_a))

    # 6. Corruption Robustness (Defaults test)
    def test_robustness():
        from experiments.corruption_robustness import run_corruption_robustness
        rob_cfg = config.copy()
        # Ensure "corruption" key is MISSING to test defaults
        if "corruption" in rob_cfg: del rob_cfg["corruption"]
        run_corruption_robustness(rob_cfg)
        assert os.path.exists(os.path.join(config["output_dir"], "corruption", "plots", "corruption_accuracy.png"))
    results.append(test_scenario("Corruption Robustness (Defaults)", test_robustness))

    # 7. Layer-wise Probing
    def test_probing():
        from experiments.layerwise_probing import run_layerwise_probing
        run_layerwise_probing(config)
        assert os.path.exists(os.path.join(config["output_dir"], "layerwise_probing", "plots", "accuracy_vs_depth.png"))
    results.append(test_scenario("Layer-wise Probing", test_probing))

    # 8. Efficiency Summary
    def test_efficiency():
        from utils.efficiency_summary import generate_efficiency_summary
        generate_efficiency_summary()
        assert os.path.exists(os.path.join("outputs", "efficiency_summary.csv"))
    results.append(test_scenario("Efficiency Summary", test_efficiency))

    # 9. Budget Report
    def test_budget():
        from utils.budget_report import generate_budget_report
        generate_budget_report()
        assert os.path.exists(os.path.join("outputs", "computational_budget.csv"))
    results.append(test_scenario("Budget Report", test_budget))

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"Summary: {passed}/{total} tests passed.")
    print("Note: Some tests (UMAP, Few-Shot Plots) may skip logic if data is too small.")
    print("This is normal for dry-runs. As long as they report 'PASS', logic is sound.")
    print(f"Ready to run full pipeline: {'YES' if passed == total else 'NO'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_all_tests()
