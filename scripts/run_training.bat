@echo off
:: scripts/run_training.bat
:: Launch MiniMedNet training on Windows

echo ============================================
echo   Transparent Mini-Med — Training Launcher
echo ============================================
echo.

:: Check Python
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

:: Check if weights folder exists
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist "outputs" mkdir outputs

:: Check dataset
if not exist "data\chest_xrays\train" (
    echo [ERROR] Training data not found at data\chest_xrays\train
    echo.
    echo Please download the dataset first:
    echo   python scripts\download_dataset.py
    echo.
    pause
    exit /b 1
)

echo [INFO] Starting training...
echo.
python core\training\train.py --config config\training_config.yaml --model-config config\model_config.yaml

if errorlevel 1 (
    echo.
    echo [ERROR] Training failed. See output above.
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Training complete!
echo   Best weights: models\mini_med_net_best.pth
echo   Logs:         logs\training_history.json
echo.
echo To evaluate: python core\training\evaluate.py --weights models\mini_med_net_best.pth --data-dir data\chest_xrays\test
echo To run UI:   python app.py
echo.
pause
