# Build & Release Guide

## Overview

This project is now set up to create standalone executables for Windows and Linux using PyInstaller. Users can download and run the application without installing Python, conda, or any dependencies.

## What Was Set Up

### 1. PyInstaller Configuration (`backend/tagger.spec`)
- Configured to bundle PyTorch, Transformers, and all dependencies
- Handles complex imports and dynamic loading
- Excludes unnecessary packages to reduce size
- Creates ~3-5GB executable (large due to PyTorch + CUDA)

### 2. Build Scripts
- **`build-executable.sh`** (Linux/macOS): Automated build for Unix systems
- **`build-executable.bat`** (Windows): Automated build for Windows
- Both scripts:
  - Activate conda environment
  - Install PyInstaller if needed
  - Build the executable
  - Create distributable archives

### 3. GitHub Actions Workflow (`.github/workflows/build-release.yml`)
- Automatically builds executables when you create a release
- Builds for both Windows and Linux
- Uploads binaries to the release
- Can be manually triggered

### 4. Dependencies (`requirements.txt`)
- Lists all required packages
- Includes PyInstaller for building
- Documents optional packages (like flash-attn)

### 5. Documentation
- **README.md**: Updated with download and build instructions
- **QUICKSTART.md**: User guide for executable users
- **BUILD_GUIDE.md**: This file - for developers

## How to Create a Release

### Method 1: Automatic (Recommended)

1. **Create a GitHub Release:**
   ```bash
   # First, commit and push your changes
   git add .
   git commit -m "Prepare for release v1.0.0"
   git push

   # Tag the release
   git tag v1.0.0
   git push origin v1.0.0
   ```

2. **Go to GitHub:**
   - Navigate to your repository
   - Click "Releases" → "Create a new release"
   - Select the tag you just created (v1.0.0)
   - Add release notes describing changes
   - Click "Publish release"

3. **GitHub Actions will automatically:**
   - Build Windows executable
   - Build Linux executable
   - Upload both to the release
   - Files will be named: `ai-image-tagger-windows-v1.0.0.zip` and `ai-image-tagger-linux-v1.0.0.tar.gz`

### Method 2: Manual Build

#### On Linux/macOS:
```bash
# Ensure you're in the project directory
cd /path/to/ai-image-tagger

# Run the build script
./build-executable.sh

# The executable will be in: build-output/dist/ai-image-tagger/
# Create distributable archive:
cd build-output/dist
tar -czf ai-image-tagger-linux.tar.gz ai-image-tagger/
```

#### On Windows:
```batch
# Open Command Prompt in the project directory
cd C:\path\to\ai-image-tagger

# Run the build script
build-executable.bat

# The executable will be in: build-output\dist\ai-image-tagger\
# Create distributable ZIP:
cd build-output\dist
powershell Compress-Archive -Path ai-image-tagger -DestinationPath ai-image-tagger-windows.zip
```

## Build Requirements

### System Requirements
- **Windows**: Windows 10/11, 64-bit
- **Linux**: Ubuntu 18.04+ or equivalent (64-bit)
- **macOS**: macOS 10.14+ (untested but should work)

### Software Requirements
- Python 3.10 (must match your conda environment)
- Conda environment `tagger-gpu` set up
- All dependencies installed (see requirements.txt)
- ~10GB free disk space for build process

## Testing the Executable

1. **Build the executable** using one of the methods above

2. **Test locally:**
   ```bash
   # Navigate to the dist folder
   cd build-output/dist/ai-image-tagger

   # Run the executable
   ./ai-image-tagger  # Linux
   ai-image-tagger.exe  # Windows
   ```

3. **Verify:**
   - Server starts on port 5000
   - No Python/conda errors
   - Can access models
   - Frontend can connect

4. **Test on a clean machine** (important!):
   - Copy the entire `ai-image-tagger/` folder to another computer
   - Computer should NOT have Python or conda installed
   - Run the executable
   - Verify models download correctly
   - Test image captioning

## Known Limitations

1. **File Size**: Executables are large (3-5GB) due to PyTorch and CUDA libraries
2. **CUDA Required**: Users must have NVIDIA GPU with CUDA drivers installed
3. **Model Download**: AI models (~500MB-2GB) still download on first run to user's cache
4. **Platform-Specific**: Must build on target platform (Windows build on Windows, Linux on Linux)
5. **GPU Compatibility**: Only works with CUDA-capable NVIDIA GPUs

## Troubleshooting Builds

### "Module not found" errors:
- Add missing module to `hiddenimports` in `tagger.spec`
- Rebuild with: `pyinstaller backend/tagger.spec --clean`

### Executable crashes on startup:
- Check if all PyTorch CUDA libraries are included
- Verify CUDA version matches (currently 11.8)
- Test in verbose mode: `./ai-image-tagger --debug`

### Huge executable size:
- Executables are inherently large due to PyTorch
- Consider using UPX compression (already enabled)
- Cannot reduce much without breaking functionality

### GitHub Actions fails:
- Check CUDA version compatibility
- Verify all dependencies in requirements.txt
- Check Actions logs for specific errors

## Distribution Checklist

Before releasing to users:

- [ ] Test executable on clean Windows machine
- [ ] Test executable on clean Linux machine
- [ ] Verify GPU detection works
- [ ] Verify models download correctly
- [ ] Test both BLIP and R-4B models
- [ ] Test all precision modes (float16, 8-bit, etc.)
- [ ] Update version numbers in README
- [ ] Write clear release notes
- [ ] Update QUICKSTART.md if needed
- [ ] Tag release with semantic version (v1.0.0)

## Future Improvements

Potential enhancements:

1. **Docker Image**: Easier cross-platform distribution
2. **Model Bundling**: Include models in executable (would be 5-10GB total)
3. **Auto-Updater**: Add self-update functionality
4. **Installer**: Create proper installers (.msi, .deb)
5. **macOS Support**: Test and add macOS builds
6. **CPU-Only Version**: Smaller build for CPU inference

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):
- **Major (1.x.x)**: Breaking changes
- **Minor (x.1.x)**: New features, backward compatible
- **Patch (x.x.1)**: Bug fixes, backward compatible

Example:
- v1.0.0 - Initial release
- v1.1.0 - Add new model support
- v1.1.1 - Fix model loading bug
- v2.0.0 - Redesign API (breaking change)

## Support

For build issues:
- Check GitHub Actions logs
- Review PyInstaller documentation
- Test on the target platform directly
- Check CUDA/PyTorch compatibility

---

**Last Updated**: 2025-10-02
**PyInstaller Version**: 6.0.0+
**PyTorch Version**: 2.8.0+cu128
