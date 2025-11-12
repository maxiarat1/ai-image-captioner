# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for AI Image Captioner

Auto-discovers dependencies and handles PyTorch/CUDA/ML packages.
Usage: ./build.sh (recommended) or: cd backend && pyinstaller captioner.spec
"""

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

block_cipher = None

# ============================================================================
# Helper: Parse requirements.txt
# ============================================================================
def get_packages_from_requirements():
    """Extract package names from requirements.txt"""
    spec_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    req_path = spec_dir.parent / 'requirements.txt' if 'backend' in str(spec_dir) else spec_dir / 'requirements.txt'
    
    packages = set()
    if req_path.exists():
        for line in req_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Handle git URLs (extract package name)
            if line.startswith('git+'):
                packages.add(line.split('/')[-1].replace('.git', '').lower())
                continue
            # Extract base package name
            pkg = line.split('==')[0].split('>=')[0].split('[')[0].split('<')[0].strip()
            if pkg:
                packages.add(pkg.lower().replace('-', '_'))
    return sorted(packages)

# ============================================================================
# Configure Hidden Imports
# ============================================================================
# Critical submodules that PyInstaller can't auto-detect (dynamically loaded)
CRITICAL_IMPORTS = [
    'flask', 'flask_cors', 'werkzeug.security', 'werkzeug.routing', 'jinja2',
    'PIL.JpegImagePlugin', 'PIL.PngImagePlugin', 'PIL.WebPImagePlugin', 'PIL.BmpImagePlugin',
    'torch.cuda', 'torch.nn.functional', 'torch.utils.data',
    'transformers.models.auto', 'transformers.generation.utils', 
    'transformers.image_utils', 'transformers.modeling_utils',
    'accelerate.utils', 'huggingface_hub.utils', 'packaging.version',
    'duckdb', 'timm', 'doctr', 'importlib.metadata', 'importlib_metadata',
]

print("Discovering dependencies...")
auto_packages = get_packages_from_requirements()
hidden_imports = sorted(set(CRITICAL_IMPORTS + auto_packages))
print(f"  Packages to include: {len(hidden_imports)}")

# Collect submodules for complex ML packages
print("Collecting submodules for ML frameworks...")
for pkg in ['transformers', 'accelerate', 'timm', 'doctr']:
    if pkg in hidden_imports:
        try:
            hidden_imports += collect_submodules(pkg)
            print(f"  OK: {pkg}")
        except Exception as e:
            print(f"  Warning: {pkg}: {e}")

# ============================================================================
# Configure Data Files
# ============================================================================
datas = [('../frontend', 'frontend')]  # Include frontend

# Collect data files for packages that need them
print("Collecting data files...")
for pkg in ['transformers', 'tokenizers', 'duckdb', 'timm', 'doctr']:
    if pkg in hidden_imports:
        try:
            datas += collect_data_files(pkg, include_py_files=False)
            print(f"  OK: {pkg}")
        except Exception as e:
            print(f"  Warning: {pkg}: {e}")

# Collect metadata for packages that check versions at runtime
print("Collecting package metadata...")
metadata_pkgs = [
    'duckdb', 'transformers', 'tokenizers', 'torch', 'accelerate', 'timm', 
    'doctr', 'pillow', 'flask', 'huggingface-hub', 'safetensors', 
    'bitsandbytes', 'tqdm', 'requests', 'pandas', 'numpy', 'scipy', 'einops', 'packaging'
]
for pkg in metadata_pkgs:
    pkg_normalized = pkg.replace('-', '_')
    if pkg in hidden_imports or pkg_normalized in hidden_imports:
        try:
            datas += copy_metadata(pkg)
        except Exception:
            pass  # Silent - not all packages have metadata

print(f"Total data files collected: {len(datas)}")

# ============================================================================
# PyInstaller Analysis Configuration
# ============================================================================

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude large unused packages (auto-excluded if not in requirements.txt)
        'matplotlib' if 'matplotlib' not in auto_packages else None,
        'pandas' if 'pandas' not in auto_packages else None,
        # Development/testing tools
        'jupyter', 'notebook', 'IPython', 'pytest',
        # Build tools (avoid jaraco.text dependency)
        'setuptools', 'pkg_resources', 'wheel', 'pip', 'distutils',
        # Problematic transformers kernel
        'transformers.kernels.falcon_mamba',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

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
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
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
