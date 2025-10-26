@echo off
REM Setup script for AI Image Captioner with GPU support (Windows)

setlocal enabledelayedexpansion

echo Setting up AI Image Captioner with GPU support...
echo.

REM Detect CUDA version from nvidia-smi if available
set CUDA_VERSION=12.1
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=4" %%i in ('nvidia-smi ^| findstr "CUDA Version"') do (
        echo Detected CUDA driver version: %%i
        REM Extract major.minor version and compare
        for /f "tokens=1,2 delims=." %%a in ("%%i") do (
            set MAJOR=%%a
            set MINOR=%%b
        )
        REM Use CUDA 12.8 for version >= 12.4
        if !MAJOR! gtr 12 (
            set CUDA_VERSION=12.8
        ) else if !MAJOR! equ 12 (
            if !MINOR! geq 4 (
                set CUDA_VERSION=12.8
            )
        )
    )
)

echo Using CUDA version: %CUDA_VERSION%
echo.

REM Check if conda is available
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: conda is not installed or not in PATH
    echo Please install Miniconda or Anaconda from:
    echo   https://docs.conda.io/en/latest/miniconda.html
    exit /b 1
)

REM Check if environment already exists
conda env list | findstr /r "^captioner-gpu " >nul 2>&1
if %errorlevel% equ 0 (
    echo Environment 'captioner-gpu' already exists.
    set /p RECREATE="Do you want to remove and recreate it? (y/N): "
    if /i "!RECREATE!"=="y" (
        echo Removing existing environment...
        call conda env remove -n captioner-gpu -y
    ) else (
        echo Aborting setup.
        exit /b 1
    )
)

REM Create conda environment with PyTorch and CUDA
echo Creating conda environment with Python 3.10...
if "%CUDA_VERSION%"=="12.8" (
    REM For CUDA 12.8 (RTX 50 series and newer)
    call conda create -n captioner-gpu python=3.10 -y
    call conda activate captioner-gpu
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
) else (
    REM For CUDA 12.1 (RTX 20/30/40 series)
    call conda create -n captioner-gpu python=3.10 pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
    call conda activate captioner-gpu
)

REM Install project dependencies
echo.
echo Installing project dependencies...
pip install -r requirements.txt

echo.
echo ✅ Setup complete!
echo.
echo To activate the environment, run:
echo   conda activate captioner-gpu
echo.
echo To start the application, run:
echo   cd backend ^&^& python app.py
echo.
echo Then open http://localhost:5000 in your browser
