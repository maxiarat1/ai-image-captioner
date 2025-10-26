#!/bin/bash
# Build script for AI Image Captioner

set -e

# Read version info
APP_VERSION=$(jq -r '.app_version' version.json 2>/dev/null || echo "dev")
BUILD_CONFIG="${1:-default}"

# Validate config
if ! jq -e ".build_configs.${BUILD_CONFIG}" version.json > /dev/null 2>&1; then
    echo "Error: Invalid build config '${BUILD_CONFIG}'"
    echo "Available: $(jq -r '.build_configs | keys | join(", ")' version.json)"
    exit 1
fi

PYTHON_VER=$(jq -r ".build_configs.${BUILD_CONFIG}.python" version.json)
CUDA_VER=$(jq -r ".build_configs.${BUILD_CONFIG}.cuda" version.json)
CUDA_DISPLAY=$(jq -r ".build_configs.${BUILD_CONFIG}.cuda_version_display" version.json)

echo "Building AI Image Captioner v${APP_VERSION}"
echo "Config: ${BUILD_CONFIG} (Python ${PYTHON_VER}, CUDA ${CUDA_DISPLAY})"
echo ""

# Activate conda environment
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate captioner-gpu
fi

# Install PyInstaller if needed
pip install -q pyinstaller

# Clean and build
rm -rf dist build
cd backend
pyinstaller captioner.spec --distpath ../dist --workpath ../build
cd ..

echo "✅ Build complete: dist/ai-image-captioner/"
echo ""
echo "To run: ./dist/ai-image-captioner/ai-image-captioner"
echo "To package: tar -czf ai-image-captioner.tar.gz -C dist ai-image-captioner"
