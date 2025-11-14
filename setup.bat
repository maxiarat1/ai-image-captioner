@echo off
REM ============================================================
REM  AI Image Captioner Unified Setup (Windows)
REM  Supports both GPU and CPU installations
REM ============================================================
setlocal enabledelayedexpansion

REM Parse command-line arguments
set MODE=
set CUDA_VERSION_ARG=

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="/gpu" (
    set MODE=gpu
    shift
    goto parse_args
)
if /i "%~1"=="/cpu" (
    set MODE=cpu
    shift
    goto parse_args
)
if /i "%~1"=="/cuda" (
    set CUDA_VERSION_ARG=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="/?" goto usage
if /i "%~1"=="/h" goto usage
if /i "%~1"=="/help" goto usage

echo Error: Unknown option: %~1
goto usage

:args_done

REM Validate that a mode was selected
if "%MODE%"=="" (
    echo Error: You must specify either /gpu or /cpu
    goto usage
)

echo ============================================================
echo   AI Image Captioner Setup
echo ============================================================
echo.

REM Check if version.json exists
if not exist version.json (
    echo Error: version.json not found in the current directory.
    exit /b 1
)

REM Check for Python availability (needed for JSON parsing)
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    echo Python is required to parse version.json.
    exit /b 1
)

REM Check if conda is available
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: conda is not installed or not in PATH.
    echo Please install Miniconda or Anaconda:
    echo   https://docs.conda.io/en/latest/miniconda.html
    exit /b 1
)

REM ============================================================
REM GPU Installation
REM ============================================================
if "%MODE%"=="gpu" (
    echo Setting up with GPU support...
    echo.

    REM Detect or use provided CUDA version
    if not "%CUDA_VERSION_ARG%"=="" (
        set CUDA_VERSION=%CUDA_VERSION_ARG%
        echo Using specified CUDA version: !CUDA_VERSION!
    ) else (
        set CUDA_VERSION=12.1
        where nvidia-smi >nul 2>&1
        if !errorlevel! equ 0 (
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
        echo Auto-detected CUDA version: !CUDA_VERSION!
    )
    echo.

    REM Read build configuration from version.json
    for /f "usebackq delims=" %%i in (`python -c "import json; data = json.load(open('version.json')); cfg = next((v for k, v in data['build_configs'].items() if v.get('cuda_version_display') == '!CUDA_VERSION!'), None); print(cfg['python']) if cfg else print('ERROR')"`) do set PYTHON_VERSION=%%i
    for /f "usebackq delims=" %%i in (`python -c "import json; data = json.load(open('version.json')); cfg = next((v for k, v in data['build_configs'].items() if v.get('cuda_version_display') == '!CUDA_VERSION!'), None); print(cfg['cuda']) if cfg else print('ERROR')"`) do set CUDA_LABEL=%%i

    if "!PYTHON_VERSION!"=="ERROR" (
        echo Error: No matching build config found in version.json for CUDA !CUDA_VERSION!.
        echo Available CUDA versions in version.json:
        python -c "import json; data = json.load(open('version.json')); print('\n'.join(sorted({v.get('cuda_version_display','') for v in data['build_configs'].values()})))"
        exit /b 1
    )

    echo Selected build config -^> Python: !PYTHON_VERSION!, CUDA label: !CUDA_LABEL!
    echo.

    REM Get additional configuration values
    for /f "tokens=1,2,3,4,5,6,7 delims=|" %%a in ('python -c "import json; cfg = next((v for k, v in json.load(open('version.json'))['build_configs'].items() if v.get('cuda_version_display') == '!CUDA_VERSION!'), None); print('|'.join([cfg['pytorch']['install_method'], cfg['pytorch']['version'], cfg['pytorch']['torchvision'], cfg['pytorch']['torchaudio'], str(cfg['pytorch'].get('index_url', '')), cfg['flash_attention']['version'], cfg['flash_attention']['cuda_suffix']]))"') do (
        set "PYTORCH_METHOD=%%a"
        set "PYTORCH_VERSION=%%b"
        set "TORCHVISION_VERSION=%%c"
        set "TORCHAUDIO_VERSION=%%d"
        set "PYTORCH_INDEX_URL=%%e"
        set "FA_VERSION=%%f"
        set "FA_CUDA_SUFFIX=%%g"
    )

    REM Convert Python version to CPython ABI tag
    set PY_ABI_TAG=cp!PYTHON_VERSION:.=!

    REM Check if environment already exists
    set ENV_NAME=captioner-gpu
    conda env list | findstr /r "^!ENV_NAME! " >nul 2>&1
    if !errorlevel! equ 0 (
        echo Environment '!ENV_NAME!' already exists.
        set /p RECREATE="Do you want to remove and recreate it? (y/N): "
        if /i "!RECREATE!"=="y" (
            echo Removing existing environment...
            call conda env remove -n !ENV_NAME! -y
        ) else (
            echo Aborting setup.
            exit /b 1
        )
    )

    REM Create conda environment
    echo Creating conda environment with Python !PYTHON_VERSION!...
    call conda create -n !ENV_NAME! python=!PYTHON_VERSION! -y
    call conda activate !ENV_NAME!

    REM Install PyTorch stack
    echo Installing PyTorch stack...
    if "!PYTORCH_METHOD!"=="pip" (
        pip install torch==!PYTORCH_VERSION!+!CUDA_LABEL! torchvision==!TORCHVISION_VERSION!+!CUDA_LABEL! torchaudio==!TORCHAUDIO_VERSION!+!CUDA_LABEL! --index-url !PYTORCH_INDEX_URL!
    ) else if "!PYTORCH_METHOD!"=="conda" (
        set PYTORCH_CUDA=!CUDA_LABEL:cu=!
        call conda install pytorch==!PYTORCH_VERSION! torchvision==!TORCHVISION_VERSION! torchaudio==!TORCHAUDIO_VERSION! pytorch-cuda=!PYTORCH_CUDA! -c pytorch -c nvidia -y
    )

    REM Install build helpers
    echo Installing build helpers...
    python -c "import json; data = json.load(open('version.json')); cfg = next((v for k, v in data['build_configs'].items() if v.get('cuda_version_display') == '!CUDA_VERSION!'), None); [print(p) for p in cfg['additional_packages']['build_helpers']]" > helpers.tmp
    for /f "delims=" %%i in (helpers.tmp) do pip install %%i
    del helpers.tmp

    REM Install FlashAttention
    set FA_WHEEL=flash_attn-!FA_VERSION!+!FA_CUDA_SUFFIX!-!PY_ABI_TAG!-!PY_ABI_TAG!-win_amd64.whl
    set FA_URL=https://github.com/Dao-AILab/flash-attention/releases/download/v!FA_VERSION!/!FA_WHEEL!
    echo Downloading FlashAttention wheel: !FA_WHEEL!
    powershell -Command "try { Invoke-WebRequest -Uri '!FA_URL!' -OutFile '!FA_WHEEL!' } catch { Write-Host 'Warning: Failed to download FlashAttention'; exit 0 }"
    if exist !FA_WHEEL! (
        pip install !FA_WHEEL! --no-build-isolation
        del !FA_WHEEL!
    )

    REM Install additional packages
    for /f "usebackq delims=" %%i in (`python -c "import json; data = json.load(open('version.json')); cfg = next((v for k, v in data['build_configs'].items() if v.get('cuda_version_display') == '!CUDA_VERSION!'), None); print(cfg['additional_packages'].get('doctr', ''))"`) do set DOCTR_PKG=%%i
    for /f "usebackq delims=" %%i in (`python -c "import json; data = json.load(open('version.json')); cfg = next((v for k, v in data['build_configs'].items() if v.get('cuda_version_display') == '!CUDA_VERSION!'), None); print(cfg['additional_packages'].get('onnxruntime', ''))"`) do set ONNX_PKG=%%i

    if not "!DOCTR_PKG!"=="" pip install !DOCTR_PKG!
    if not "!ONNX_PKG!"=="" pip install !ONNX_PKG!

    REM Install GPU-specific packages
    echo Installing GPU-specific packages...
    python -c "import json; data = json.load(open('version.json')); cfg = next((v for k, v in data['build_configs'].items() if v.get('cuda_version_display') == '!CUDA_VERSION!'), None); pkgs = cfg.get('gpu_specific_packages', []); [print(p) for p in pkgs]" > gpu_pkgs.tmp
    for /f "delims=" %%i in (gpu_pkgs.tmp) do (
        if not "%%i"=="" pip install %%i
    )
    del gpu_pkgs.tmp

    REM Install project dependencies
    echo.
    echo Installing project dependencies...
    pip install -r requirements.txt

    REM Verify installation
    echo.
    echo Verifying installation...
    python -c "import sys, platform; print(f'Python: {sys.version.split()[0]} ^| Platform: {platform.platform()}'); import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '')"
    python -c "try: import flash_attn; print(f'FlashAttention: {flash_attn.__version__}'); except: print('FlashAttention: Not installed (optional)')"

    echo.
    echo ============================================================
    echo   GPU Setup complete!
    echo ============================================================
    echo.
    echo To activate the environment, run:
    echo   conda activate !ENV_NAME!
    echo.
    echo To start the application, run:
    echo   cd backend ^&^& python app.py
    echo.
    echo Then open http://localhost:5000 in your browser
    echo.

REM ============================================================
REM CPU Installation
REM ============================================================
) else if "%MODE%"=="cpu" (
    echo Setting up with CPU support...
    echo.

    REM Read Python version from cpu_config
    for /f "usebackq delims=" %%i in (`python -c "import json; data = json.load(open('version.json')); py = data.get('cpu_config', {}).get('python', ''); print(py if py else next(iter(data['build_configs'].values()))['python'])"`) do set PYTHON_VERSION=%%i

    if "!PYTHON_VERSION!"=="" (
        echo Warning: Could not read Python version from version.json, defaulting to 3.12
        set PYTHON_VERSION=3.12
    )

    echo Using Python version: !PYTHON_VERSION!
    echo.

    REM Check if environment already exists
    set ENV_NAME=captioner-cpu
    conda env list | findstr /r "^!ENV_NAME! " >nul 2>&1
    if !errorlevel! equ 0 (
        echo Environment '!ENV_NAME!' already exists.
        set /p RECREATE="Do you want to remove and recreate it? (y/N): "
        if /i "!RECREATE!"=="y" (
            echo Removing existing environment...
            call conda env remove -n !ENV_NAME! -y
        ) else (
            echo Aborting setup.
            exit /b 1
        )
    )

    REM Create conda environment
    echo Creating conda environment with Python !PYTHON_VERSION!...
    call conda create -n !ENV_NAME! python=!PYTHON_VERSION! -y
    call conda activate !ENV_NAME!

    REM Read PyTorch version from cpu_config if available
    for /f "tokens=1,2,3 delims=|" %%a in ('python -c "import json; cfg = json.load(open('version.json')).get('cpu_config', {}).get('pytorch', {}); print('|'.join([cfg.get('version', ''), cfg.get('torchvision', ''), cfg.get('torchaudio', '')]))"') do (
        set "PYTORCH_VERSION=%%a"
        set "TORCHVISION_VERSION=%%b"
        set "TORCHAUDIO_VERSION=%%c"
    )

    REM Install PyTorch CPU version
    echo Installing PyTorch (CPU version^)...
    if not "!PYTORCH_VERSION!"=="" (
        pip install torch==!PYTORCH_VERSION! torchvision==!TORCHVISION_VERSION! torchaudio==!TORCHAUDIO_VERSION! --index-url https://download.pytorch.org/whl/cpu
    ) else (
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    )

    REM Install additional CPU packages from config
    for /f "usebackq delims=" %%i in (`python -c "import json; cfg = json.load(open('version.json')).get('cpu_config', {}).get('additional_packages', {}); print(cfg.get('doctr', 'python-doctr[torch]'))"`) do set DOCTR_PKG=%%i
    for /f "usebackq delims=" %%i in (`python -c "import json; cfg = json.load(open('version.json')).get('cpu_config', {}).get('additional_packages', {}); print(cfg.get('onnxruntime', 'onnxruntime'))"`) do set ONNX_PKG=%%i

    pip install !DOCTR_PKG!
    pip install !ONNX_PKG!

    REM Install project dependencies
    echo.
    echo Installing project dependencies...
    pip install -r requirements.txt

    REM Verify installation
    echo.
    echo Verifying installation...
    python -c "import sys, platform; print(f'Python: {sys.version.split()[0]} ^| Platform: {platform.platform()}'); import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print('Installation verified successfully (CPU mode^)')"

    echo.
    echo ============================================================
    echo   CPU Setup complete!
    echo ============================================================
    echo.
    echo To activate the environment, run:
    echo   conda activate !ENV_NAME!
    echo.
    echo To start the application, run:
    echo   cd backend ^&^& python app.py
    echo.
    echo Then open http://localhost:5000 in your browser
    echo.
    echo Note: CPU mode will be slower than GPU mode, especially for large models.
    echo.
)

endlocal
goto :eof

:usage
echo.
echo Usage: setup.bat [/gpu [/cuda VERSION]] ^| [/cpu]
echo.
echo Options:
echo   /gpu          Install with GPU support (CUDA required^)
echo   /cuda VERSION Specify CUDA version (12.1 or 12.8, auto-detected if not provided^)
echo   /cpu          Install with CPU-only support
echo.
echo Examples:
echo   setup.bat /gpu                # Auto-detect CUDA version
echo   setup.bat /gpu /cuda 12.8    # Use CUDA 12.8
echo   setup.bat /cpu               # CPU-only installation
echo.
exit /b 1
