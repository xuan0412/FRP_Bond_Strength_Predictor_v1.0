@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title FRP Bond Predictor Python 3.11 Launcher
set "PYTHONUTF8=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

echo ================================================================
echo  FRP Bond Predictor v1.0.0 - Python 3.11 Launcher
echo ================================================================
echo.

set "BASEPY="

where py >nul 2>nul
if not errorlevel 1 (
    py -3.11 -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) and sys.maxsize>2**32 else 1)" >nul 2>nul
    if not errorlevel 1 set "BASEPY=py -3.11"
)

if not defined BASEPY (
    where python >nul 2>nul
    if not errorlevel 1 (
        python -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) and sys.maxsize>2**32 else 1)" >nul 2>nul
        if not errorlevel 1 set "BASEPY=python"
    )
)

if not defined BASEPY goto python_missing

echo [1/6] Python 3.11 detected:
%BASEPY% --version

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) else 1)" >nul 2>nul
    if errorlevel 1 (
        echo       Existing .venv is not Python 3.11. Recreating it...
        rmdir /s /q ".venv"
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo [2/6] Creating isolated Python 3.11 environment...
    %BASEPY% -m venv .venv
    if errorlevel 1 goto venv_failed
) else (
    echo [2/6] Python 3.11 virtual environment found.
)

set "PY=.venv\Scripts\python.exe"

echo [3/6] Checking dependencies...
"%PY%" -c "import numpy,pandas,sklearn,xgboost,catboost,joblib" >nul 2>nul
if errorlevel 1 (
    echo [4/6] Installing prebuilt packages. Internet is needed on the first run.
    echo       This may take several minutes because CatBoost and XGBoost are large.
    "%PY%" -m pip install --upgrade pip wheel setuptools
    if errorlevel 1 goto install_failed
    "%PY%" -m pip install --only-binary=:all: -r requirements.txt
    if errorlevel 1 goto install_failed
) else (
    echo [4/6] Dependencies are already installed.
)

set "RETRAIN=0"
if not exist "models\model_meta.json" set "RETRAIN=1"
if exist "models\model_meta.json" (
    "%PY%" -c "import json; m=json.load(open(r'models\model_meta.json',encoding='utf-8')); raise SystemExit(0 if m.get('python_major_minor')=='3.11' else 1)" >nul 2>nul
    if errorlevel 1 set "RETRAIN=1"
)

if "%RETRAIN%"=="1" (
    echo [5/6] Training 9 models locally with Python 3.11...
    echo       This is required only once and can take a few minutes.
    "%PY%" train_models.py
    if errorlevel 1 goto train_failed
) else (
    echo [5/6] Python 3.11 model files found.
)

echo [6/6] Starting desktop and mobile web application...
echo.
"%PY%" app.py
if errorlevel 1 goto app_failed
goto end

:python_missing
echo.
echo [ERROR] 64-bit Python 3.11 was not found.
echo Install Python 3.11 x64, then run this file again.
echo If Python 3.11 is already installed, verify with:
echo     py -3.11 --version
echo or:
echo     python --version
pause
exit /b 1

:venv_failed
echo.
echo [ERROR] Failed to create the Python 3.11 virtual environment.
pause
exit /b 1

:install_failed
echo.
echo [ERROR] Dependency installation failed.
echo Copy the complete error message above for diagnosis.
pause
exit /b 1

:train_failed
echo.
echo [ERROR] Model training failed.
echo Copy the complete error message above for diagnosis.
pause
exit /b 1

:app_failed
echo.
echo [ERROR] The local web application stopped unexpectedly.
echo Copy the complete error message above for diagnosis.
pause
exit /b 1

:end
endlocal
