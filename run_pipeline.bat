@echo off
chcp 65001 > nul
echo GNR638 Assignment 2 - Full Pipeline Run
echo ==========================================

call .venv311\Scripts\activate

set PYTORCH_ALLOC_CONF=expandable_segments:True
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

echo Starting pipeline at %date% %time%
echo Results will be saved to outputs/
echo.

echo [1/5] Running Linear Probe...
python main.py --scenario linear_probe --model all ^
  --data_path ./dataloader/AID ^
  --output_dir outputs/ ^
  --seed 42 --epochs 30 --batch_size 16
if %errorlevel% neq 0 (
    echo ERROR: Linear probe failed! Check logs.
    pause
    exit /b 1
)
echo [1/5] Linear Probe COMPLETE
echo.

echo [2/5] Running Fine-Tuning...
python main.py --scenario finetune --model all ^
  --data_path ./dataloader/AID ^
  --output_dir outputs/ ^
  --seed 42 --epochs 30 --batch_size 16
if %errorlevel% neq 0 (
    echo ERROR: Fine-tuning failed! Check logs.
    pause
    exit /b 1
)
echo [2/5] Fine-Tuning COMPLETE
echo.

echo [3/5] Running Few-Shot Analysis...
python main.py --scenario fewshot --model all ^
  --data_path ./dataloader/AID ^
  --output_dir outputs/ ^
  --seed 42 --epochs 20 --batch_size 16
if %errorlevel% neq 0 (
    echo ERROR: Few-shot failed! Check logs.
    pause
    exit /b 1
)
echo [3/5] Few-Shot COMPLETE
echo.

echo [4/5] Running Corruption Robustness...
python main.py --scenario robustness --model all ^
  --data_path ./dataloader/AID ^
  --output_dir outputs/ ^
  --seed 42 --batch_size 16
if %errorlevel% neq 0 (
    echo ERROR: Robustness failed! Check logs.
    pause
    exit /b 1
)
echo [4/5] Corruption Robustness COMPLETE
echo.

echo [5/5] Running Layer-wise Probing...
python main.py --scenario probing --model all ^
  --data_path ./dataloader/AID ^
  --output_dir outputs/ ^
  --seed 42 --epochs 30 --batch_size 16
if %errorlevel% neq 0 (
    echo ERROR: Probing failed! Check logs.
    pause
    exit /b 1
)
echo [5/5] Layer-wise Probing COMPLETE
echo.

echo ==========================================
echo ALL 5 SCENARIOS COMPLETE
echo Finished at %date% %time%
echo Check outputs/ folder for all results.
echo ==========================================
pause
