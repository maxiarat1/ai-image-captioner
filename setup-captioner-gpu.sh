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

# Create conda environment with PyTorch and CUDA
echo "Creating conda environment with Python 3.10..."
if [ "$CUDA_VERSION" = "12.8" ]; then
    echo "🐉 Setting up environment for CUDA 12.8 (RTX 50-series or newer)"

    conda create -n captioner-gpu python=3.10 -y
    conda activate captioner-gpu

    # Install PyTorch stack (CUDA 12.8 wheels from PyTorch site)
    pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
      --index-url https://download.pytorch.org/whl/cu128

    # Recommended build helpers
    pip install packaging ninja

    # Install FlashAttention 2 (prebuilt binary for CUDA 12.x + Torch 2.7)
    wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.2/flash_attn-2.8.2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    pip install flash_attn-2.8.2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-build-isolation
    rm flash_attn-2.8.2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

    # Verify
    echo ""
    echo "✅ Verifying PyTorch, CUDA, and FlashAttention installation..."
    python -c "import torch, flash_attn; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'FlashAttention:', flash_attn.__version__')"

elif [ "$CUDA_VERSION" = "12.1" ]; then
    echo "⚡ Setting up environment for CUDA 12.1 (RTX 20/30/40-series)"

    conda create -n captioner-gpu python=3.10 -y
    conda activate captioner-gpu

    # Install official PyTorch 2.5 stack for CUDA 12.1
    conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 \
      -c pytorch -c nvidia -y

    # Build helpers
    pip install packaging ninja

    # Install FlashAttention 2.8.3 prebuilt wheel (CUDA 12.x + Torch 2.5)
    wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    pip install flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-build-isolation
    rm flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

    # Verify
    echo ""
    echo "✅ Verifying PyTorch, CUDA, and FlashAttention installation..."
    python -c "import torch, flash_attn; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'FlashAttention:', flash_attn.__version__')"

else
    echo "❌ Unsupported CUDA version: $CUDA_VERSION"
    echo "Please set CUDA_VERSION to either 12.1 or 12.8."
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
