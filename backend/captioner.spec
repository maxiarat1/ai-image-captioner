# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for AI Image Captioner Backend

This builds a standalone executable that includes:
- Flask backend server
- PyTorch with CUDA support
- Multiple AI vision models (BLIP, R-4B, WD-ViT, Janus, Qwen3-VL, DeepSeek-OCR)
- All required dependencies

Build command:
    pyinstaller backend/captioner.spec

Output directories:
    - Build artifacts: build-output/build/
    - Final executable: build-output/dist/ai-image-captioner/

Note: The executable will be large (~3-5GB) due to PyTorch and CUDA libraries.
Models will be downloaded on first run to ~/.cache/huggingface/
"""

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import os
from pathlib import Path

block_cipher = None

# Define output directories at project root
project_root = Path(os.getcwd())
build_dir = project_root / 'build-output' / 'build'
dist_dir = project_root / 'build-output' / 'dist'

# Collect all submodules for critical packages
hidden_imports = [
    # Flask and CORS
    'flask',
    'flask_cors',
    'werkzeug',

    # PIL/Pillow
    'PIL',
    'PIL.Image',
    'PIL.ImageFile',
    'PIL.JpegImagePlugin',
    'PIL.PngImagePlugin',
    'PIL.WebPImagePlugin',
    'PIL.BmpImagePlugin',

    # PyTorch and related
    'torch',
    'torch.cuda',
    'torch.nn',
    'torch.nn.functional',
    'torch.utils',
    'torch.utils.data',
    'torch.optim',
    'torchvision',
    'torchvision.transforms',

    # Transformers and dependencies
    'transformers',
    'transformers.models',
    'transformers.models.auto',
    'transformers.generation',
    'transformers.modeling_utils',
    'transformers.tokenization_utils',
    'transformers.tokenization_utils_base',
    'transformers.image_utils',
    'transformers.utils',

    # Accelerate
    'accelerate',
    'accelerate.utils',

    # BitsAndBytes
    'bitsandbytes',

    # Hugging Face Hub
    'huggingface_hub',
    'huggingface_hub.utils',

    # Other critical imports
    'packaging',
    'packaging.version',
    'filelock',
    'requests',
    'tqdm',
    'numpy',
    'regex',
    'safetensors',
    'sentencepiece',
    'tokenizers',

    # DuckDB
    'duckdb',
]

# Collect all submodules for transformers (it has many dynamic imports)
hidden_imports += collect_submodules('transformers')
hidden_imports += collect_submodules('accelerate')

# Collect data files
datas = []
datas += collect_data_files('transformers')
datas += collect_data_files('accelerate')
datas += collect_data_files('bitsandbytes')
datas += collect_data_files('torch', include_py_files=True)
datas += collect_data_files('torchvision', include_py_files=True)
datas += collect_data_files('duckdb')

# Add the frontend directory (optional - include if you want a single package)
# Uncomment the line below to bundle frontend with backend
datas += [('../frontend', 'frontend')]

# Binary files and libraries
binaries = []

# Add CUDA libraries if available (for Windows)
if sys.platform == 'win32':
    # PyTorch CUDA libraries are usually included automatically
    pass

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unused modules to reduce size
        'matplotlib',
        'scipy',
        'pandas',
        'jupyter',
        'notebook',
        'IPython',
        'pytest',
        'setuptools',
        'wheel',
        'pip',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Set custom build directory
a.SPECPATH = str(build_dir)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ai-image-captioner',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Keep console for server logs
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ai-image-captioner',
)

# Override default paths
import PyInstaller.config
PyInstaller.config.CONF['workpath'] = str(build_dir)
PyInstaller.config.CONF['distpath'] = str(dist_dir)
