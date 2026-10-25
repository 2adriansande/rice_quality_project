@echo off
REM Runs all four steps in order (Windows). Usage: run_all.bat "C:\path\to\dataset"
if "%~1"=="" (
  echo Usage: run_all.bat "C:\path\to\dataset"
  exit /b 1
)
python 01_data_preparation.py %1 || exit /b 1
python 02_data_splitting.py || exit /b 1
python 03_model_training.py || exit /b 1
python 04_model_evaluation.py || exit /b 1
echo Done. See the rice_quality_output folder.
