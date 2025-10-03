#!/bin/bash
# Simple build script for AI Image Tagger

set -e

echo "Building AI Image Tagger..."

# Activate conda environment
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate tagger-gpu
fi

# Install PyInstaller if needed
pip install -q pyinstaller

# Clean and build
rm -rf dist build
cd backend
pyinstaller tagger.spec --distpath ../dist --workpath ../build
cd ..

echo "✅ Build complete: dist/ai-image-tagger/"
echo ""
echo "To run: ./dist/ai-image-tagger/ai-image-tagger"
echo "To package: tar -czf ai-image-tagger.tar.gz -C dist ai-image-tagger"
