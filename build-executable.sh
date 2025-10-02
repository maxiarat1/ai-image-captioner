#!/bin/bash
# Build script for AI Image Tagger executable (Linux/macOS)
# This script builds a standalone executable using PyInstaller

set -e  # Exit on error

echo "=========================================="
echo "AI Image Tagger - Build Script (Linux/macOS)"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first"
    exit 1
fi

# Activate the tagger-gpu environment
echo "📦 Activating tagger-gpu environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tagger-gpu

# Check if PyInstaller is installed
if ! python -c "import PyInstaller" 2>/dev/null; then
    echo "📥 Installing PyInstaller..."
    pip install pyinstaller
else
    echo "✅ PyInstaller is already installed"
fi

# Clean previous builds
echo ""
echo "🧹 Cleaning previous builds..."
rm -rf build-output backend/*.spec.backup

# Build the executable
echo ""
echo "🔨 Building executable with PyInstaller..."
echo "   (This may take 5-15 minutes depending on your system)"
echo ""

cd backend
pyinstaller tagger.spec --distpath ../build-output/dist --workpath ../build-output/build
cd ..

# Check if build was successful
if [ -d "build-output/dist/ai-image-tagger" ]; then
    echo ""
    echo "=========================================="
    echo "✅ Build successful!"
    echo "=========================================="
    echo ""
    echo "📍 Executable location: build-output/dist/ai-image-tagger/"
    echo "📍 Main executable: build-output/dist/ai-image-tagger/ai-image-tagger"
    echo ""
    echo "To run the application:"
    echo "  cd build-output/dist/ai-image-tagger"
    echo "  ./ai-image-tagger"
    echo ""
    echo "To create a distributable archive:"
    echo "  cd build-output/dist"
    echo "  tar -czf ai-image-tagger-linux.tar.gz ai-image-tagger/"
    echo ""

    # Calculate size
    SIZE=$(du -sh build-output/dist/ai-image-tagger | cut -f1)
    echo "📦 Package size: $SIZE"
    echo ""
    echo "Note: Models will be downloaded on first run (~500MB-2GB)"
    echo "      to ~/.cache/huggingface/"
else
    echo ""
    echo "❌ Build failed! Check the output above for errors."
    exit 1
fi
