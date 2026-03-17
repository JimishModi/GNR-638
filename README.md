# GNR 638 Assignment 2: CNN Transfer Learning & Robustness Analysis
**Course:** GNR638 Graduate Deep Learning  
**Dataset:** AID (Aerial Images Dataset)  
**Models:** ResNet50, DenseNet121, EfficientNet-B0  
**Seed:** 42 (Fixed for reproducibility)

## 1. Project Overview
This repository implements a comprehensive pipeline for analyzing transfer learning strategies and robustness of pretrained Convolutional Neural Networks on the 30-class AID dataset. The project covers:
- **Linear Probe Transfer**: Evaluating frozen features via PCA, t-SNE, and UMAP.
- **Fine-Tuning Strategies**: Comparing 4 strategies (Linear, Last Block, Full, Selective 20%).
- **Few-Shot Analysis**: Comparing Head-only (Mode A) vs. Full Fine-Tuning (Mode B) at 100%, 20%, and 5% data.
- **Corruption Robustness**: Analyzing performance under noise, blur, and brightness shifts.
- **Layer-wise Probing**: Probing feature quality and norms across network depth.

## 2. Requirements & Setup
- **Python Version**: 3.11 (Explicitly required)
- **OS**: Windows 11 / Linux
- **GPU**: NVIDIA CUDA (12.1+ recommended)

### Installation
```bash
# Create virtual environment
python -m venv .venv311
source .venv311/bin/activate  # Linux/Mac
.venv311\Scripts\activate     # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Dataset Setup
Expected folder structure for the AID dataset:
```text
dataloader/AID/
├── Airport/
├── BareLand/
├── BaseballField/
...
└── Viaduct/
```
The pipeline automatically performs an **80/20 stratified train-validation split**.

## 4. Usage Instructions
Run the `main.py` entry point with the desired scenario.

### Run All Scenarios (Models: all, Epochs: 30)
```bash
python main.py --scenario all --data_path ./dataloader/AID
```

### Run Specific Scenarios
- **Linear Probe**: `python main.py --scenario linear_probe --data_path ./dataloader/AID`
- **Fine-Tuning**: `python main.py --scenario finetune --data_path ./dataloader/AID`
- **Few-Shot**: `python main.py --scenario fewshot --data_path ./dataloader/AID`
- **Robustness**: `python main.py --scenario robustness --data_path ./dataloader/AID`
- **Probing**: `python main.py --scenario probing --data_path ./dataloader/AID`

### Additional Flags
- `--dry_run`: Execute only 1 batch per epoch (for smoke testing).
- `--model`: `resnet50`, `densenet121`, `efficientnet_b0`, or `all`.
- `--seed`: Default is 42.

## 5. Outputs
All results are saved into the `outputs/` directory:
- `outputs/linear_probe/`: PCA, t-SNE, UMAP plots, Confusion Matrices, Failure Case Analysis grids.
- `outputs/few_shot/`: Grouped bar charts (Mode A vs B) and `fewshot_results.csv`.
- `outputs/efficiency_summary.csv/png`: Complexity table (Params, MACs, FLOPs).
- `outputs/computational_budget.csv/png`: Resource usage report.

## 6. Project Structure
```text
.
├── analysis/           # Visualization and Probing logic
├── configs/            # Configuration files
├── dataloader/         # Data loading and splitting
├── evaluation/         # Metrics and confusion matrix utilities
├── experiments/        # Scenario-specific execution scripts
├── models/             # Model loading and freezing utilities
├── training/           # Training and evaluation loops
├── main.py             # Main entry point
├── README.md           # Documentation
└── requirements.txt    # Dependency list
```

## 7. Deterministic Results
The seed is fixed at **42**. GPU behavior is set to deterministic via `torch.backends.cudnn.deterministic`. GPU/VRAM are auto-detected and logged.
