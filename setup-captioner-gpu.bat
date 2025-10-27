@echo off
REM ============================================================
REM  AI Image Captioner GPU Environment Setup (Windows)
REM ============================================================
setlocal enabledelayedexpansion

echo Setting up AI Image Captioner with GPU support...
echo.

REM ------------------------------------------------------------
REM Detect CUDA version (via nvidia-smi)
REM ------------------------------------------------------------
set CUDA_VERSION=12.1
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=4" %%i in ('nvidia-smi ^| findstr "CUDA Version"') do (
        echo Detected CUDA driver version: %%i
        for /f "tokens=1,2 delims=." %%a in ("%%i") do (
            set MAJOR=%%a
            set MINOR=%%b
        )
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

REM ------------------------------------------------------------
REM Verify that conda is installed
REM ------------------------------------------------------------
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Error: conda is not installed or not in PATH.
    echo Please install Miniconda or Anaconda:
    echo   https://docs.conda.io/en/latest/miniconda.html
    exit /b 1
)

REM ------------------------------------------------------------
REM Remove existing environment if desired
REM ------------------------------------------------------------
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

REM ------------------------------------------------------------
REM Environment creation (CUDA-specific logic)
REM ------------------------------------------------------------
echo Creating conda environment with Python 3.10...
if "%CUDA_VERSION%"=="12.8" (
    echo 🐉 Configuring for CUDA 12.8 (RTX 50-series or newer)...
    call conda create -n captioner-gpu python=3.10 -y
    call conda activate captioner-gpu

    echo Installing PyTorch 2.7.1 stack...
    pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128

    echo Installing helpers...
    pip install packaging ninja

    echo Installing FlashAttention 2.8.2 (CUDA12.x + Torch2.7)...
    powershell -Command "Invoke-WebRequest -Uri https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.2/flash_attn-2.8.2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl -OutFile flash_attn-2.8.2.whl"
    pip install flash_attn-2.8.2.whl --no-build-isolation
    del flash_attn-2.8.2.whl

) else (
    echo ⚡ Configuring for CUDA 12.1 (RTX 20/30/40-series)...
    call conda create -n captioner-gpu python=3.10 -y
    call conda activate captioner-gpu

    echo Installing PyTorch 2.5.1 stack...
    call conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

    echo Installing helpers...
    pip install packaging ninja

    echo Installing FlashAttention 2.8.3 (CUDA12.x + Torch2.5)...
    powershell -Command "Invoke-WebRequest -Uri https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl -OutFile flash_attn-2.8.3.whl"
    pip install flash_attn-2.8.3.whl --no-build-isolation
    del flash_attn-2.8.3.whl
)

REM ------------------------------------------------------------
REM Install project dependencies (if file exists)
REM ------------------------------------------------------------
if exist requirements.txt (
    echo Installing project dependencies...
    pip install -r requirements.txt
) else (
    echo No requirements.txt found, skipping dependency installation.
)

REM ------------------------------------------------------------
REM Verify everything
REM ------------------------------------------------------------
echo.
echo ✅ Verifying installation...
python -c "import torch, flash_attn; print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda, '| FlashAttention:', flash_attn.__version__)"

echo.
echo 🎉 Setup complete!
echo To activate environment: conda activate captioner-gpu
echo To start app: cd backend && python app.py
echo.
endlocal
