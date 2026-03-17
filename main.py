# GNR638 Assignment 2 | Task 7 | seed=42
"""
GNR 638 Assignment 2 — Transfer Learning & Robustness Pipeline
Main entry point.

Handles command-line arguments, environment setup, and experiment orchestration.
"""

import os
import sys

# Fix CUDA memory fragmentation on Windows
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Fix Windows console Unicode encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        # Fallback for older python or environments without reconfigure
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import argparse
import time
import random
import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logging import get_logger


def set_seed(seed=42):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Also set python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def cleanup_gpu_memory():
    """Force GPU memory cleanup between runs."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load the YAML configuration file."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="GNR 638 — Transfer Learning & Robustness Pipeline"
    )
    
    # Required Arguments
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        choices=["linear_probe", "finetune", "fewshot", "robustness", "probing", "all"],
        help="Which experiment scenario to run."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the AID dataset root folder."
    )
    
    # Optional Arguments
    parser.add_argument("--model", type=str, default="all", 
                        choices=["resnet50", "densenet121", "efficientnet_b0", "all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--dry_run", action="store_true", help="1 batch execution for smoke testing.")

    args = parser.parse_args()

    # 1. Setup Seeding
    set_seed(args.seed)

    # 2. Setup Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"\nDevice: {device}")
    if torch.cuda.is_available() and device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    
    # 3. Load & Merge Config
    config = load_config(args.config)
    
    # Overwrite config with CLI args
    config.update({
        "data_dir": args.data_path,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "epochs_full": args.epochs,
        "batch_size": args.batch_size,
        "device": str(device),
        "dry_run": args.dry_run
    })
    
    if args.model != "all":
        config["models"] = [args.model]
    else:
        config["models"] = ["resnet50", "densenet121", "efficientnet_b0"]

    if args.dry_run:
        print("*** DRY RUN MODE — 1 batch per epoch ***")
        config["epochs_full"] = 2
        config["epochs_few_shot"] = 2

    # 4. Prepare Directories
    os.makedirs(config["output_dir"], exist_ok=True)
    logger = get_logger("main", log_dir=os.path.join(config["output_dir"], "logs"))
    logger.info(f"Scenario: {args.scenario} | Models: {config['models']}")

    # 5. Orchestrate Experiments
    scenarios_to_run = []
    if args.scenario == "all":
        scenarios_to_run = ["linear_probe", "finetune", "fewshot", "robustness", "probing"]
    else:
        scenarios_to_run = [args.scenario]

    for sc_name in scenarios_to_run:
        cleanup_gpu_memory()  # BEFORE each scenario
        logger.info(f"\n{'#'*80}")
        logger.info(f"  Starting scenario: {sc_name}")
        logger.info(f"{'#'*80}\n")
        start = time.time()

        if sc_name == "linear_probe":
            from experiments.linear_probe import run_linear_probe
            run_linear_probe(config)

        elif sc_name == "finetune":
            from experiments.fine_tuning import run_fine_tuning
            run_fine_tuning(config)

        elif sc_name == "fewshot":
            from experiments.few_shot import run_few_shot
            run_few_shot(config)

        elif sc_name == "robustness":
            from experiments.corruption_robustness import run_corruption_robustness
            run_corruption_robustness(config)

        elif sc_name == "probing":
            from experiments.layerwise_probing import run_layerwise_probing
            run_layerwise_probing(config)

        elapsed = time.time() - start
        cleanup_gpu_memory()  # AFTER each scenario
        logger.info(f"\n  Scenario '{sc_name}' completed in {elapsed/60:.1f} minutes.\n")

    # 6. Final Reporting (Task 5 & 6)
    if args.scenario == "all":
        logger.info("Generating Final Efficiency and Budget Reports...")
        from utils.efficiency_summary import generate_efficiency_summary
        from utils.budget_report import generate_budget_report
        generate_efficiency_summary()
        generate_budget_report()

    logger.info("Execution Pipeline Finished.")


if __name__ == "__main__":
    main()
