#!/bin/bash
# Setup script for AI Image Captioner with GPU support

set -e

echo "Setting up AI Image Captioner with GPU support..."
echo ""

# Detect CUDA version from nvidia-smi if available
CUDA_VERSION="12.1"
if command -v nvidia-smi &> /dev/null; then
    DRIVER_CUDA=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9.]+")
    if [[ ! -z "$DRIVER_CUDA" ]]; then
        echo "Detected CUDA driver version: $DRIVER_CUDA"
        # Use CUDA 12.8 for newer drivers, 12.1 for older
        if (( $(echo "$DRIVER_CUDA >= 12.4" | bc -l) )); then
            CUDA_VERSION="12.8"
        fi
    fi
fi

echo "Using CUDA version: $CUDA_VERSION"
echo ""

# Read build configuration from version.json matching the CUDA version
if [[ ! -f "version.json" ]]; then
    echo "Error: version.json not found in the current directory."
    exit 1
fi

PYTHON_VERSION=""
CUDA_LABEL=""

if command -v jq &> /dev/null; then
    # Prefer jq if available
    PYTHON_VERSION=$(jq -r --arg ver "$CUDA_VERSION" '.build_configs | to_entries[] | select(.value.cuda_version_display == $ver) | .value.python' version.json)
    CUDA_LABEL=$(jq -r --arg ver "$CUDA_VERSION" '.build_configs | to_entries[] | select(.value.cuda_version_display == $ver) | .value.cuda' version.json)
else
    # Fallback to Python for JSON parsing
    if command -v python3 &> /dev/null; then
        readarray -t __cfg_vals < <(python3 - "$CUDA_VERSION" <<'PY'
import json, sys
ver = sys.argv[1]
with open('version.json') as f:
    data = json.load(f)
cfg = None
for k, v in data.get('build_configs', {}).items():
    if v.get('cuda_version_display') == ver:
        cfg = v
        break
if not cfg:
    sys.exit(2)
print(cfg.get('python', ''))
print(cfg.get('cuda', ''))
PY
        )
        PYTHON_VERSION="${__cfg_vals[0]}"
        CUDA_LABEL="${__cfg_vals[1]}"
        unset __cfg_vals
    else
        echo "Error: Neither 'jq' nor 'python3' is available to parse version.json. Please install one of them."
        exit 1
    fi
fi

if [[ -z "$PYTHON_VERSION" || -z "$CUDA_LABEL" || "$PYTHON_VERSION" == "null" || "$CUDA_LABEL" == "null" ]]; then
    echo "Error: No matching build config found in version.json for CUDA $CUDA_VERSION."
    echo "Available CUDA versions in version.json:"
    if command -v jq &> /dev/null; then
        jq -r '.build_configs | to_entries[] | "- " + .value.cuda_version_display' version.json | sort -u
    else
        python3 - <<'PY'
import json
with open('version.json') as f:
    data = json.load(f)
print('\n'.join(sorted({v.get('cuda_version_display','') for v in data.get('build_configs',{}).values()})))
PY
    fi
    exit 1
fi

echo "Selected build config from version.json -> Python: $PYTHON_VERSION, CUDA label: $CUDA_LABEL (display: $CUDA_VERSION)"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda from:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Source conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Check if environment already exists
if conda env list | grep -q "^captioner-gpu "; then
    echo "Environment 'captioner-gpu' already exists."
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n captioner-gpu -y
    else
        echo "Aborting setup."
        exit 1
    fi
fi

# Create conda environment with PyTorch and CUDA using versions from version.json
echo "Creating conda environment with Python $PYTHON_VERSION..."
 
# Define a verification helper that runs inside the currently active environment
verify_stack() {
    echo ""
    echo "✅ Verifying PyTorch, CUDA, and FlashAttention installation..."
    python - <<'PY'
import sys, traceback, platform

def report(msg):
    print(msg)

report(f"Python: {sys.version.split()[0]} | Platform: {platform.platform()}")

try:
    import torch
except Exception as e:
    print("ERROR: Failed to import torch:", e, file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

report(f"PyTorch: {torch.__version__}")
report(f"CUDA runtime in PyTorch: {getattr(torch.version, 'cuda', None)}")
cuda_ok = False
try:
    cuda_ok = torch.cuda.is_available()
    report(f"CUDA available: {cuda_ok}")
    if cuda_ok:
        try:
            cnt = torch.cuda.device_count()
            cur = torch.cuda.current_device()
            name = torch.cuda.get_device_name(cur)
            report(f"CUDA device count: {cnt}")
            report(f"Current device: {cur} | Name: {name}")
        except Exception as e:
            print("WARNING: CUDA query failed:", e, file=sys.stderr)
except Exception as e:
    print("WARNING: torch.cuda.is_available() check failed:", e, file=sys.stderr)

try:
    import flash_attn
except Exception as e:
    print("ERROR: Failed to import flash_attn:", e, file=sys.stderr)
    traceback.print_exc()
    print("\nHint: Ensure FlashAttention wheel matches your Python and Torch/CUDA versions.")
    print("      If you changed Python from 3.12, you may need a different cp tag wheel.")
    sys.exit(2)

fa_ver = getattr(flash_attn, '__version__', None)
report(f"FlashAttention: {fa_ver}")
sys.exit(0)
PY
}

# Convert Python version to CPython ABI tag (e.g., "3.12" -> "cp312", "3.10" -> "cp310")
PY_ABI_TAG="cp${PYTHON_VERSION//./}"

if [[ "$CUDA_LABEL" == "cu128" ]]; then
    echo "🐉 Setting up environment for CUDA 12.8 (RTX 50-series or newer)"

    conda create -n captioner-gpu python="$PYTHON_VERSION" -y
    conda activate captioner-gpu

    # Install PyTorch stack (CUDA 12.8 wheels from PyTorch site)
    pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
      --index-url https://download.pytorch.org/whl/cu128

    # Recommended build helpers
    pip install packaging ninja

    # Install FlashAttention 2 (prebuilt binary for CUDA 12.x + Torch 2.7)
    FA_WHEEL="flash_attn-2.8.2+cu12torch2.7cxx11abiFALSE-${PY_ABI_TAG}-${PY_ABI_TAG}-linux_x86_64.whl"
    FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.2/${FA_WHEEL}"
    echo "Downloading FlashAttention wheel for Python ${PYTHON_VERSION}: ${FA_WHEEL}"
    wget "$FA_URL"
    pip install "$FA_WHEEL" --no-build-isolation
    rm "$FA_WHEEL"

    # Verify
    verify_stack

elif [[ "$CUDA_LABEL" == "cu121" ]]; then
    echo "⚡ Setting up environment for CUDA 12.1 (RTX 20/30/40-series)"

    conda create -n captioner-gpu python="$PYTHON_VERSION" -y
    conda activate captioner-gpu

    # Install official PyTorch 2.5 stack for CUDA 12.1
    conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 \
      -c pytorch -c nvidia -y

    # Build helpers
    pip install packaging ninja

    # Install FlashAttention 2.8.3 prebuilt wheel (CUDA 12.x + Torch 2.5)
    FA_WHEEL="flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-${PY_ABI_TAG}-${PY_ABI_TAG}-linux_x86_64.whl"
    FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/${FA_WHEEL}"
    echo "Downloading FlashAttention wheel for Python ${PYTHON_VERSION}: ${FA_WHEEL}"
    wget "$FA_URL"
    pip install "$FA_WHEEL" --no-build-isolation
    rm "$FA_WHEEL"

    # Verify
    verify_stack

else
    echo "❌ Unsupported or unknown CUDA label from version.json: $CUDA_LABEL"
    echo "Please ensure version.json contains a supported 'cuda' label (e.g., cu121 or cu128)."
    exit 1
fi


# Install project dependencies
echo ""
echo "Installing project dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate captioner-gpu"
echo ""
echo "To start the application, run:"
echo "  cd backend && python app.py"
echo ""
echo "Then open http://localhost:5000 in your browser"
