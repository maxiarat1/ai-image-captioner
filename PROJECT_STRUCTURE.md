# Project Structure

## Overview

```
ai-image-tagger/
├── backend/                      # Backend Flask application
│   ├── app.py                    # Main Flask server
│   ├── models/                   # Model adapters
│   │   ├── base_adapter.py       # Abstract base class
│   │   ├── blip_adapter.py       # BLIP model implementation
│   │   └── r4b_adapter.py        # R-4B model implementation
│   ├── utils/                    # Utilities
│   │   └── image_utils.py        # Image processing functions
│   └── tagger.spec               # PyInstaller configuration
│
├── frontend/                     # Frontend web interface
│   ├── index.html                # Main UI
│   ├── script.js                 # JavaScript logic
│   └── styles.css                # Styling (dark/light themes)
│
├── build-output/                 # Build artifacts (gitignored)
│   ├── build/                    # PyInstaller temporary build files
│   └── dist/                     # Final executables
│       └── ai-image-tagger/      # Distributable application
│           ├── ai-image-tagger   # Executable (Linux)
│           └── ai-image-tagger.exe  # Executable (Windows)
│
├── assets/                       # Project assets
│   └── Image Tagger.png          # Screenshot for README
│
├── .github/                      # GitHub configuration
│   └── workflows/
│       └── build-release.yml     # CI/CD workflow for releases
│
├── build-executable.sh           # Linux/macOS build script
├── build-executable.bat          # Windows build script
├── setup-tagger-gpu.sh           # Environment setup script
├── requirements.txt              # Python dependencies
├── user_config.json              # User configuration (auto-generated)
│
├── README.md                     # Main documentation
├── QUICKSTART.md                 # User quick start guide
├── BUILD_GUIDE.md                # Developer build guide
├── PROJECT_STRUCTURE.md          # This file
├── CLAUDE.md                     # Instructions for Claude Code
│
└── .gitignore                    # Git ignore rules
```

## Directory Purposes

### Source Code
- **`backend/`**: Flask API server and AI model implementations
- **`frontend/`**: Static HTML/CSS/JS web interface

### Build System
- **`build-output/`**: All PyInstaller build artifacts (not committed to git)
  - `build/`: Temporary build files (large, ~10GB)
  - `dist/`: Final executables ready for distribution (~7-8GB uncompressed)

### Scripts
- **`build-executable.sh`**: Automated build for Linux/macOS
- **`build-executable.bat`**: Automated build for Windows
- **`setup-tagger-gpu.sh`**: Sets up conda environment with dependencies

### Documentation
- **`README.md`**: Main project documentation for users
- **`QUICKSTART.md`**: Guide for end users who download executables
- **`BUILD_GUIDE.md`**: Guide for developers building releases
- **`PROJECT_STRUCTURE.md`**: This structure overview
- **`CLAUDE.md`**: Instructions for AI code assistant

### Configuration
- **`.github/workflows/`**: GitHub Actions CI/CD pipelines
- **`requirements.txt`**: Python package dependencies
- **`user_config.json`**: User preferences (saved prompts, configs)
- **`.gitignore`**: Files to exclude from version control

## Build Workflow

### Local Development
1. Edit code in `backend/` or `frontend/`
2. Test with `python backend/app.py`
3. Open `frontend/index.html` in browser

### Building Executables
1. Run `./build-executable.sh` (Linux) or `build-executable.bat` (Windows)
2. PyInstaller reads `backend/tagger.spec`
3. Outputs to `build-output/dist/ai-image-tagger/`
4. Create archive: `tar -czf ai-image-tagger-linux.tar.gz ai-image-tagger/`

### Automated Releases (GitHub Actions)
1. Create git tag: `git tag v1.0.0 && git push origin v1.0.0`
2. Create GitHub release
3. GitHub Actions runs `build-release.yml`
4. Builds executables for Windows and Linux
5. Uploads to release automatically

## Key Files

| File | Purpose |
|------|---------|
| `backend/app.py` | Main Flask server entrypoint |
| `backend/tagger.spec` | PyInstaller build configuration |
| `build-executable.sh` | Build script for Unix systems |
| `.github/workflows/build-release.yml` | CI/CD automation |
| `requirements.txt` | Python dependencies list |
| `.gitignore` | Prevents committing build artifacts |

## Size Information

- **Source code**: ~500 KB
- **Build artifacts** (`build-output/build/`): ~10 GB (temporary)
- **Final executable** (`build-output/dist/`): ~7-8 GB uncompressed
- **Compressed archive**: ~2-3 GB (for distribution)
- **AI models** (downloaded on first run): ~500 MB - 2 GB

## Clean Build

To start fresh:

```bash
# Remove all build artifacts
rm -rf build-output/

# Remove user config (optional)
rm -f user_config.json

# Rebuild
./build-executable.sh
```

## Notes

- `build-output/` is gitignored and never committed
- Old `backend/build/` and `backend/dist/` directories are deprecated
- Models are NOT included in executable; downloaded to `~/.cache/huggingface/`
- Frontend can be bundled with backend by uncommenting line in `tagger.spec`

---

**Last Updated**: 2025-10-02
