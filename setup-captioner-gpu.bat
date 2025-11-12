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
REM Read build configuration from version.json
REM ------------------------------------------------------------
if not exist version.json (
    echo Error: version.json not found in the current directory.
    exit /b 1
)

REM Check for Python availability
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    echo Python is required to parse version.json.
    exit /b 1
)

REM Parse version.json using Python
for /f "usebackq delims=" %%i in (`python -c "import json; data = json.load(open('version.json')); cfg = next((v for k, v in data['build_configs'].items() if v.get('cuda_version_display') == '%CUDA_VERSION%'), None); print(cfg['python']) if cfg else print('ERROR')"`) do set PYTHON_VERSION=%%i
for /f "usebackq delims=" %%i in (`python -c "import json; data = json.load(open('version.json')); cfg = next((v for k, v in data['build_configs'].items() if v.get('cuda_version_display') == '%CUDA_VERSION%'), None); print(cfg['cuda']) if cfg else print('ERROR')"`) do set CUDA_LABEL=%%i

if "%PYTHON_VERSION%"=="ERROR" (
    echo Error: No matching build config found in version.json for CUDA %CUDA_VERSION%.
    echo Available CUDA versions in version.json:
    python -c "import json; data = json.load(open('version.json')); print('\n'.join(sorted({v.get('cuda_version_display','') for v in data['build_configs'].values()})))"
    exit /b 1
)

echo Selected build config from version.json -^> Python: %PYTHON_VERSION%, CUDA label: %CUDA_LABEL% (display: %CUDA_VERSION%)
echo.

REM Get additional configuration values
for /f "usebackq delims=" %%i in (`python -c "import json; data = json.load(open('version.json')); cfg = next((v for k, v in data['build_configs'].items() if v.get('cuda_version_display') == '%CUDA_VERSION%'), None); print(cfg['pytorch']['install_method'])"`) do set PYTORCH_METHOD=%%i
for /f "usebackq delims=" %%i in (`python -c "import json; data = json.load(open('version.json')); cfg = next((v for k, v in data['build_configs'].items() if v.get('cuda_version_display') == '%CUDA_VERSION%'), None); print(cfg['pytorch']['version'])"`) do set PYTORCH_VERSION=%%i
for /f "usebackq delims=" %%i in (`python -c "import json; data = json.load(open('version.json')); cfg = next((v for k, v in data['build_configs'].items() if v.get('cuda_version_display') == '%CUDA_VERSION%'), None); print(cfg['pytorch']['torchvision'])"`) do set TORCHVISION_VERSION=%%i
for /f "usebackq delims=" %%i in (`python -c "import json; data = json.load(open('version.json')); cfg = next((v for k, v in data['build_configs'].items() if v.get('cuda_version_display') == '%CUDA_VERSION%'), None); print(cfg['pytorch']['torchaudio'])"`) do set TORCHAUDIO_VERSION=%%i
for /f "usebackq delims=" %%i in (`python -c "import json; data = json.load(open('version.json')); cfg = next((v for k, v in data['build_configs'].items() if v.get('cuda_version_display') == '%CUDA_VERSION%'), None); print(cfg['pytorch'].get('index_url', ''))"`) do set PYTORCH_INDEX_URL=%%i
for /f "usebackq delims=" %%i in (`python -c "import json; data = json.load(open('version.json')); cfg = next((v for k, v in data['build_configs'].items() if v.get('cuda_version_display') == '%CUDA_VERSION%'), None); print(cfg['pytorch'].get('pytorch_cuda', ''))"`) do set PYTORCH_CUDA=%%i
for /f "usebackq delims=" %%i in (`python -c "import json; data = json.load(open('version.json')); cfg = next((v for k, v in data['build_configs'].items() if v.get('cuda_version_display') == '%CUDA_VERSION%'), None); print(cfg['flash_attention']['version'])"`) do set FA_VERSION=%%i
for /f "usebackq delims=" %%i in (`python -c "import json; data = json.load(open('version.json')); cfg = next((v for k, v in data['build_configs'].items() if v.get('cuda_version_display') == '%CUDA_VERSION%'), None); print(cfg['flash_attention']['cuda_suffix'])"`) do set FA_CUDA_SUFFIX=%%i

REM Convert Python version to CPython ABI tag (e.g., 3.12 -> cp312)
set PY_ABI_TAG=cp%PYTHON_VERSION:.=%

REM ------------------------------------------------------------
REM Verify that conda is installed
REM ------------------------------------------------------------
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: conda is not installed or not in PATH.
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
REM Create conda environment
REM ------------------------------------------------------------
echo Creating conda environment with Python %PYTHON_VERSION%...
call conda create -n captioner-gpu python=%PYTHON_VERSION% -y
call conda activate captioner-gpu

REM ------------------------------------------------------------
REM Install PyTorch stack based on configuration
REM ------------------------------------------------------------
if "%PYTORCH_METHOD%"=="pip" (
    echo Checking if torch==%PYTORCH_VERSION%+%CUDA_LABEL% is available on %PYTORCH_INDEX_URL%...
    pip install --dry-run --no-deps torch==%PYTORCH_VERSION%+%CUDA_LABEL% --index-url %PYTORCH_INDEX_URL% > pip_check.log 2>&1
    if %errorlevel% neq 0 (
        echo.
        echo ERROR: The specified torch version torch==%PYTORCH_VERSION%+%CUDA_LABEL% was not found on %PYTORCH_INDEX_URL%.
        echo Please check your version.json and ensure the version exists.
        echo See pip_check.log for details.
        exit /b 1
    )
    del pip_check.log
    echo Installing PyTorch stack via pip...
    pip install torch==%PYTORCH_VERSION%+%CUDA_LABEL% torchvision==%TORCHVISION_VERSION%+%CUDA_LABEL% torchaudio==%TORCHAUDIO_VERSION%+%CUDA_LABEL% --index-url %PYTORCH_INDEX_URL%
) else if "%PYTORCH_METHOD%"=="conda" (
    echo Installing PyTorch stack via conda...
    call conda install pytorch==%PYTORCH_VERSION% torchvision==%TORCHVISION_VERSION% torchaudio==%TORCHAUDIO_VERSION% pytorch-cuda=%PYTORCH_CUDA% -c pytorch -c nvidia -y
)

REM ------------------------------------------------------------
REM Install build helpers
REM ------------------------------------------------------------
echo Installing build helpers...
python -c "import json; data = json.load(open('version.json')); cfg = next((v for k, v in data['build_configs'].items() if v.get('cuda_version_display') == '%CUDA_VERSION%'), None); [print(p) for p in cfg['additional_packages']['build_helpers']]" > helpers.tmp
for /f "delims=" %%i in (helpers.tmp) do pip install %%i
del helpers.tmp

REM ------------------------------------------------------------
REM Install FlashAttention
REM ------------------------------------------------------------
set FA_WHEEL=flash_attn-%FA_VERSION%+%FA_CUDA_SUFFIX%-%PY_ABI_TAG%-%PY_ABI_TAG%-win_amd64.whl
set FA_URL=https://github.com/Dao-AILab/flash-attention/releases/download/v%FA_VERSION%/%FA_WHEEL%
echo Downloading FlashAttention wheel for Python %PYTHON_VERSION%: %FA_WHEEL%
powershell -Command "Invoke-WebRequest -Uri '%FA_URL%' -OutFile '%FA_WHEEL%'"
pip install %FA_WHEEL% --no-build-isolation
del %FA_WHEEL%

REM ------------------------------------------------------------
REM Install additional packages
REM ------------------------------------------------------------
for /f "usebackq delims=" %%i in (`python -c "import json; data = json.load(open('version.json')); cfg = next((v for k, v in data['build_configs'].items() if v.get('cuda_version_display') == '%CUDA_VERSION%'), None); print(cfg['additional_packages'].get('doctr', ''))"`) do set DOCTR_PKG=%%i
for /f "usebackq delims=" %%i in (`python -c "import json; data = json.load(open('version.json')); cfg = next((v for k, v in data['build_configs'].items() if v.get('cuda_version_display') == '%CUDA_VERSION%'), None); print(cfg['additional_packages'].get('onnxruntime', ''))"`) do set ONNX_PKG=%%i

if not "%DOCTR_PKG%"=="" pip install %DOCTR_PKG%
if not "%ONNX_PKG%"=="" pip install %ONNX_PKG%

REM ------------------------------------------------------------
REM Verify installation
REM ------------------------------------------------------------
echo.
echo ✅ Verifying installation...
python -c "import sys, traceback, platform; print(f'Python: {sys.version.split()[0]} ^| Platform: {platform.platform()}'); import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA runtime in PyTorch: {getattr(torch.version, \"cuda\", None)}'); cuda_ok = torch.cuda.is_available(); print(f'CUDA available: {cuda_ok}'); import flash_attn; print(f'FlashAttention: {getattr(flash_attn, \"__version__\", None)}')"

REM ------------------------------------------------------------
REM Install project dependencies
REM ------------------------------------------------------------
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
echo.
endlocal
