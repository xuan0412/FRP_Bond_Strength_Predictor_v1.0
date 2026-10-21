@echo off
setlocal
cd /d "%~dp0"
title FRP Bond Predictor Repair
echo This will rebuild the Python 3.11 environment and retrain all 9 models.
echo.
if exist ".venv" rmdir /s /q ".venv"
if exist "models" rmdir /s /q "models"
mkdir models
echo Clean completed.
echo Starting fresh installation...
call start_windows.bat
endlocal
