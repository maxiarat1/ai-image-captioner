#!/bin/bash
# Setup script for AI Image Captioner with CPU support

set -e

echo "Setting up AI Image Captioner with CPU support..."
echo ""

# Read Python version from version.json
if [[ ! -f "version.json" ]]; then
    echo "Error: version.json not found in the current directory."
    exit 1
fi

PYTHON_VERSION=""

if command -v jq &> /dev/null; then
    # Use jq if available - get Python version from first config
    PYTHON_VERSION=$(jq -r '.build_configs | to_entries[] | .value.python' version.json | head -n1)
else
    # Fallback to Python for JSON parsing
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 - <<'PY'
import json
with open('version.json') as f:
    data = json.load(f)
configs = data.get('build_configs', {})
if configs:
    first_config = next(iter(configs.values()))
    print(first_config.get('python', ''))
PY
        )
    else
        echo "Error: Neither 'jq' nor 'python3' is available to parse version.json. Please install one of them."
        exit 1
    fi
fi

if [[ -z "$PYTHON_VERSION" || "$PYTHON_VERSION" == "null" ]]; then
    echo "Warning: Could not read Python version from version.json, defaulting to 3.12"
    PYTHON_VERSION="3.12"
fi

echo "Using Python version: $PYTHON_VERSION"
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
if conda env list | grep -q "^captioner-cpu "; then
    echo "Environment 'captioner-cpu' already exists."
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n captioner-cpu -y
    else
        echo "Aborting setup."
        exit 1
    fi
fi

# Create conda environment with PyTorch CPU
echo "Creating conda environment with Python $PYTHON_VERSION..."

conda create -n captioner-cpu python="$PYTHON_VERSION" -y
conda activate captioner-cpu

# Install PyTorch CPU version
echo "Installing PyTorch (CPU version)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Doctr with Torch support
pip install python-doctr[torch]

# Install ONNX Runtime (CPU)
pip install onnxruntime

# Verify installation
echo ""
echo "✅ Verifying PyTorch installation..."
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

# Install project dependencies
echo ""
echo "Installing project dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate captioner-cpu"
echo ""
echo "To start the application, run:"
echo "  cd backend && python app.py"
echo ""
echo "Then open http://localhost:5000 in your browser"
echo ""
echo "Note: CPU mode will be slower than GPU mode, especially for large models."
