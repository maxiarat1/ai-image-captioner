#!/bin/bash
# Unified setup script for AI Image Captioner
# Supports both CPU and GPU installations

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print usage information
usage() {
    echo "Usage: $0 [--gpu [--cuda VERSION]] | [--cpu]"
    echo ""
    echo "Options:"
    echo "  --gpu          Install with GPU support (CUDA required)"
    echo "  --cuda VERSION Specify CUDA version (12.1 or 12.8, auto-detected if not provided)"
    echo "  --cpu          Install with CPU-only support"
    echo ""
    echo "Examples:"
    echo "  $0 --gpu                # Auto-detect CUDA version"
    echo "  $0 --gpu --cuda 12.8   # Use CUDA 12.8"
    echo "  $0 --cpu               # CPU-only installation"
    exit 1
}

# Parse command-line arguments
MODE=""
CUDA_VERSION_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            MODE="gpu"
            shift
            ;;
        --cpu)
            MODE="cpu"
            shift
            ;;
        --cuda)
            CUDA_VERSION_ARG="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Error: Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Validate that a mode was selected
if [[ -z "$MODE" ]]; then
    echo -e "${RED}Error: You must specify either --gpu or --cpu${NC}"
    usage
fi

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  AI Image Captioner Setup${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check if version.json exists
if [[ ! -f "version.json" ]]; then
    echo -e "${RED}Error: version.json not found in the current directory.${NC}"
    exit 1
fi

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed or not in PATH${NC}"
    echo "Please install Miniconda or Anaconda from:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Source conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# ============================================================================
# GPU Installation
# ============================================================================
if [[ "$MODE" == "gpu" ]]; then
    echo -e "${GREEN}Setting up with GPU support...${NC}"
    echo ""

    # Detect or use provided CUDA version
    if [[ -n "$CUDA_VERSION_ARG" ]]; then
        CUDA_VERSION="$CUDA_VERSION_ARG"
        echo "Using specified CUDA version: $CUDA_VERSION"
    else
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
        echo "Auto-detected CUDA version: $CUDA_VERSION"
    fi
    echo ""

    # Read build configuration from version.json
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
            echo -e "${RED}Error: Neither 'jq' nor 'python3' is available to parse version.json.${NC}"
            echo "Please install one of them."
            exit 1
        fi
    fi

    if [[ -z "$PYTHON_VERSION" || -z "$CUDA_LABEL" || "$PYTHON_VERSION" == "null" || "$CUDA_LABEL" == "null" ]]; then
        echo -e "${RED}Error: No matching build config found in version.json for CUDA $CUDA_VERSION.${NC}"
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

    echo "Selected build config -> Python: $PYTHON_VERSION, CUDA label: $CUDA_LABEL"
    echo ""

    # Check if environment already exists
    ENV_NAME="captioner-gpu"
    if conda env list | grep -q "^${ENV_NAME} "; then
        echo -e "${YELLOW}Environment '${ENV_NAME}' already exists.${NC}"
        read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Removing existing environment..."
            conda env remove -n ${ENV_NAME} -y
        else
            echo "Aborting setup."
            exit 1
        fi
    fi

    # Create conda environment
    echo "Creating conda environment with Python $PYTHON_VERSION..."
    conda create -n ${ENV_NAME} python="$PYTHON_VERSION" -y
    conda activate ${ENV_NAME}

    # Convert Python version to CPython ABI tag
    PY_ABI_TAG="cp${PYTHON_VERSION//./}"

    # Get additional configuration from version.json
    if command -v jq &> /dev/null; then
        PYTORCH_CONFIG=$(jq -r --arg ver "$CUDA_VERSION" '.build_configs | to_entries[] | select(.value.cuda_version_display == $ver) | .value.pytorch' version.json)
        FA_CONFIG=$(jq -r --arg ver "$CUDA_VERSION" '.build_configs | to_entries[] | select(.value.cuda_version_display == $ver) | .value.flash_attention' version.json)
        ADD_PACKAGES=$(jq -r --arg ver "$CUDA_VERSION" '.build_configs | to_entries[] | select(.value.cuda_version_display == $ver) | .value.additional_packages' version.json)
        GPU_PACKAGES=$(jq -r --arg ver "$CUDA_VERSION" '.build_configs | to_entries[] | select(.value.cuda_version_display == $ver) | .value.gpu_specific_packages[]' version.json)

        # Parse configuration values
        PYTORCH_METHOD=$(echo "$PYTORCH_CONFIG" | jq -r '.install_method')
        PYTORCH_VERSION=$(echo "$PYTORCH_CONFIG" | jq -r '.version')
        TORCHVISION_VERSION=$(echo "$PYTORCH_CONFIG" | jq -r '.torchvision')
        TORCHAUDIO_VERSION=$(echo "$PYTORCH_CONFIG" | jq -r '.torchaudio')
        PYTORCH_INDEX_URL=$(echo "$PYTORCH_CONFIG" | jq -r '.index_url // empty')

        FA_VERSION=$(echo "$FA_CONFIG" | jq -r '.version')
        FA_CUDA_SUFFIX=$(echo "$FA_CONFIG" | jq -r '.cuda_suffix')
    else
        echo -e "${RED}Error: jq is required for GPU setup. Please install jq.${NC}"
        exit 1
    fi

    # Install PyTorch stack
    echo "Installing PyTorch stack..."
    if [[ "$PYTORCH_METHOD" == "pip" ]]; then
        pip install torch==${PYTORCH_VERSION}+${CUDA_LABEL} \
                    torchvision==${TORCHVISION_VERSION}+${CUDA_LABEL} \
                    torchaudio==${TORCHAUDIO_VERSION}+${CUDA_LABEL} \
                    --index-url "$PYTORCH_INDEX_URL"
    elif [[ "$PYTORCH_METHOD" == "conda" ]]; then
        conda install pytorch==${PYTORCH_VERSION} \
                      torchvision==${TORCHVISION_VERSION} \
                      torchaudio==${TORCHAUDIO_VERSION} \
                      pytorch-cuda=${CUDA_LABEL#cu} \
                      -c pytorch -c nvidia -y
    fi

    # Install build helpers
    for helper in $(echo "$ADD_PACKAGES" | jq -r '.build_helpers[]'); do
        pip install "$helper"
    done

    # Install FlashAttention
    FA_WHEEL="flash_attn-${FA_VERSION}+${FA_CUDA_SUFFIX}-${PY_ABI_TAG}-${PY_ABI_TAG}-linux_x86_64.whl"
    FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FA_VERSION}/${FA_WHEEL}"
    echo "Downloading FlashAttention wheel: ${FA_WHEEL}"
    if ! wget -q "$FA_URL"; then
        echo -e "${YELLOW}Warning: Failed to download FlashAttention. Continuing without it.${NC}"
    else
        pip install "$FA_WHEEL" --no-build-isolation
        rm "$FA_WHEEL"
    fi

    # Install additional packages
    if [[ -n "$(echo "$ADD_PACKAGES" | jq -r '.doctr')" ]]; then
        pip install $(echo "$ADD_PACKAGES" | jq -r '.doctr')
    fi

    # Install GPU-specific packages (bitsandbytes, onnxruntime-gpu, etc.)
    if [[ -n "$GPU_PACKAGES" ]]; then
        echo "Installing GPU-specific packages..."
        for pkg in $GPU_PACKAGES; do
            pip install "$pkg"
        done
    fi

    # Install project dependencies
    echo ""
    echo "Installing project dependencies..."
    pip install -r requirements.txt

    # Verify installation
    echo ""
    echo -e "${GREEN}Verifying installation...${NC}"
    python - <<'PY'
import sys, platform

print(f"Python: {sys.version.split()[0]} | Platform: {platform.platform()}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"ERROR: Failed to verify torch: {e}")
    sys.exit(1)

try:
    import flash_attn
    print(f"FlashAttention: {flash_attn.__version__}")
except:
    print("FlashAttention: Not installed (optional)")

print("✅ Installation verified successfully")
PY

    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}✅ GPU Setup complete!${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "To activate the environment, run:"
    echo -e "  ${YELLOW}conda activate ${ENV_NAME}${NC}"
    echo ""
    echo "To start the application, run:"
    echo -e "  ${YELLOW}cd backend && python app.py${NC}"
    echo ""
    echo "Then open http://localhost:5000 in your browser"
    echo ""

# ============================================================================
# CPU Installation
# ============================================================================
elif [[ "$MODE" == "cpu" ]]; then
    echo -e "${GREEN}Setting up with CPU support...${NC}"
    echo ""

    # Read Python version from cpu_config in version.json
    PYTHON_VERSION=""

    if command -v jq &> /dev/null; then
        PYTHON_VERSION=$(jq -r '.cpu_config.python // empty' version.json)
        if [[ -z "$PYTHON_VERSION" ]]; then
            # Fallback to first build config
            PYTHON_VERSION=$(jq -r '.build_configs | to_entries[] | .value.python' version.json | head -n1)
        fi
    else
        if command -v python3 &> /dev/null; then
            PYTHON_VERSION=$(python3 - <<'PY'
import json
with open('version.json') as f:
    data = json.load(f)
# Try cpu_config first
py_ver = data.get('cpu_config', {}).get('python', '')
if not py_ver:
    # Fallback to first build config
    configs = data.get('build_configs', {})
    if configs:
        py_ver = next(iter(configs.values())).get('python', '')
print(py_ver)
PY
            )
        else
            echo -e "${RED}Error: Neither 'jq' nor 'python3' is available.${NC}"
            exit 1
        fi
    fi

    if [[ -z "$PYTHON_VERSION" || "$PYTHON_VERSION" == "null" ]]; then
        echo -e "${YELLOW}Warning: Could not read Python version from version.json, defaulting to 3.12${NC}"
        PYTHON_VERSION="3.12"
    fi

    echo "Using Python version: $PYTHON_VERSION"
    echo ""

    # Check if environment already exists
    ENV_NAME="captioner-cpu"
    if conda env list | grep -q "^${ENV_NAME} "; then
        echo -e "${YELLOW}Environment '${ENV_NAME}' already exists.${NC}"
        read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Removing existing environment..."
            conda env remove -n ${ENV_NAME} -y
        else
            echo "Aborting setup."
            exit 1
        fi
    fi

    # Create conda environment
    echo "Creating conda environment with Python $PYTHON_VERSION..."
    conda create -n ${ENV_NAME} python="$PYTHON_VERSION" -y
    conda activate ${ENV_NAME}

    # Read PyTorch version from cpu_config if available
    if command -v jq &> /dev/null; then
        PYTORCH_VERSION=$(jq -r '.cpu_config.pytorch.version // empty' version.json)
        TORCHVISION_VERSION=$(jq -r '.cpu_config.pytorch.torchvision // empty' version.json)
        TORCHAUDIO_VERSION=$(jq -r '.cpu_config.pytorch.torchaudio // empty' version.json)
    fi

    # Install PyTorch CPU version
    echo "Installing PyTorch (CPU version)..."
    if [[ -n "$PYTORCH_VERSION" && "$PYTORCH_VERSION" != "null" ]]; then
        pip install torch==${PYTORCH_VERSION} \
                    torchvision==${TORCHVISION_VERSION} \
                    torchaudio==${TORCHAUDIO_VERSION} \
                    --index-url https://download.pytorch.org/whl/cpu
    else
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi

    # Install additional CPU packages from config
    if command -v jq &> /dev/null; then
        DOCTR_PKG=$(jq -r '.cpu_config.additional_packages.doctr // empty' version.json)
        ONNX_PKG=$(jq -r '.cpu_config.additional_packages.onnxruntime // empty' version.json)

        if [[ -n "$DOCTR_PKG" && "$DOCTR_PKG" != "null" ]]; then
            pip install "$DOCTR_PKG"
        else
            pip install python-doctr
        fi

        if [[ -n "$ONNX_PKG" && "$ONNX_PKG" != "null" ]]; then
            pip install "$ONNX_PKG"
        else
            pip install onnxruntime
        fi
    else
        pip install python-doctr
        pip install onnxruntime
    fi

    # Install project dependencies
    echo ""
    echo "Installing project dependencies..."
    pip install -r requirements.txt

    # Verify installation
    echo ""
    echo -e "${GREEN}Verifying installation...${NC}"
    python - <<'PY'
import sys, platform

print(f"Python: {sys.version.split()[0]} | Platform: {platform.platform()}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"ERROR: Failed to import torch: {e}")
    sys.exit(1)

print("✅ PyTorch installed successfully (CPU mode)")
PY

    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}✅ CPU Setup complete!${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "To activate the environment, run:"
    echo -e "  ${YELLOW}conda activate ${ENV_NAME}${NC}"
    echo ""
    echo "To start the application, run:"
    echo -e "  ${YELLOW}cd backend && python app.py${NC}"
    echo ""
    echo "Then open http://localhost:5000 in your browser"
    echo ""
    echo -e "${YELLOW}Note: CPU mode will be slower than GPU mode, especially for large models.${NC}"
    echo ""
fi
