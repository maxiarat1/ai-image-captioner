# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for AI Image Captioner Backend

Builds a standalone executable with Flask backend and PyTorch/CUDA support.
Auto-discovers dependencies from requirements.txt

Build from project root:
    cd backend && pyinstaller captioner.spec --distpath ../dist --workpath ../build

Or use build.sh script which handles paths automatically.
"""

import sys
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Auto-discover packages from requirements.txt
def get_packages_from_requirements():
    """Parse requirements.txt and extract package names"""
    requirements_path = Path(__file__).parent.parent / 'requirements.txt'
    packages = set()
    
    if requirements_path.exists():
        with open(requirements_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Skip git URLs
                if line.startswith('git+'):
                    # Extract package name from git URL (e.g., Janus from git+...Janus.git)
                    if '/' in line:
                        pkg_name = line.split('/')[-1].replace('.git', '').lower()
                        packages.add(pkg_name)
                    continue
                # Extract package name (before ==, >=, [, etc.)
                pkg = line.split('==')[0].split('>=')[0].split('[')[0].split('<')[0].strip()
                if pkg:
                    packages.add(pkg.lower().replace('-', '_'))
    
    return sorted(packages)

# Auto-discover imports from Python source files
def get_imports_from_source():
    """Scan Python files to find import statements"""
    backend_path = Path(__file__).parent
    imports = set()
    
    # Standard library modules to ignore
    stdlib = {
        'os', 'sys', 'json', 'pathlib', 'io', 're', 'time', 'datetime',
        'typing', 'collections', 'functools', 'itertools', 'base64',
        'hashlib', 'uuid', 'logging', 'warnings', 'asyncio', 'threading',
        'csv', 'shutil', 'tempfile', 'zipfile', 'webbrowser', 'gc',
        'concurrent', 'abc', 'random', 'string', 'urllib', 'http',
    }
    
    # Local modules to ignore (your own code, not external packages)
    local_modules = {
        'app', 'backend', 'config', 'database', 'graph_executor',
        'models', 'utils', 'workers',
    }
    
    # Scan all Python files in backend directory
    for py_file in backend_path.rglob('*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Match: import xxx or from xxx import yyy
                    if line.startswith('import ') or line.startswith('from '):
                        parts = line.replace('import ', ' ').replace('from ', ' ').split()
                        if parts:
                            # Get base module name, strip trailing commas
                            module = parts[0].split('.')[0].strip().strip(',')
                            # Only add if not stdlib or local module
                            if module and module not in stdlib and module not in local_modules:
                                imports.add(module)
        except Exception:
            pass
    
    return sorted(imports)

# Base hidden imports (critical submodules that PyInstaller misses)
# These are submodules that are dynamically imported at runtime
hidden_imports = [
    # Flask submodules (dynamically loaded)
    'flask',
    'flask_cors',
    'werkzeug.security',
    'werkzeug.routing',
    'jinja2',
    
    # PIL plugins (loaded dynamically based on image format)
    'PIL.JpegImagePlugin',
    'PIL.PngImagePlugin',
    'PIL.WebPImagePlugin',
    'PIL.BmpImagePlugin',
    
    # PyTorch CUDA (optional runtime import)
    'torch.cuda',
    'torch.nn.functional',
    'torch.utils.data',
    
    # Transformers submodules (many dynamic imports)
    'transformers.models.auto',
    'transformers.generation.utils',
    'transformers.image_utils',
    'transformers.modeling_utils',
    
    # Accelerate utilities
    'accelerate.utils',
    
    # Hugging Face Hub utilities
    'huggingface_hub.utils',
    
    # Essential utilities with dynamic imports
    'packaging.version',
]

# Auto-discover and merge all imports
print("Auto-discovering packages from requirements.txt...")
auto_packages = get_packages_from_requirements()

print("Auto-discovering imports from source code...")
source_imports = get_imports_from_source()

# Merge all discovered imports (avoid duplicates)
all_imports = set(hidden_imports)
all_imports.update(auto_packages)
all_imports.update(source_imports)

# Convert back to list
hidden_imports = sorted(all_imports)

print(f"Total packages to include: {len(hidden_imports)}")
print(f"  From requirements.txt: {len(auto_packages)}")
print(f"  From source code: {len(source_imports)}")
print(f"  Manual (critical submodules): {len(set(hidden_imports) - set(auto_packages) - set(source_imports))}")


# Auto-collect submodules for packages that need it
print("Collecting submodules for complex packages...")
submodule_packages = ['transformers', 'accelerate', 'timm', 'doctr']
for pkg in submodule_packages:
    if pkg in auto_packages or pkg in hidden_imports:
        try:
            print(f"  Collecting submodules for {pkg}...")
            hidden_imports += collect_submodules(pkg)
        except Exception as e:
            print(f"  Warning: Could not collect submodules for {pkg}: {e}")
            pass

# Collect data files (only essential ones to reduce size)
datas = [
    # Include frontend for standalone distribution
    ('../frontend', 'frontend'),
]

# Auto-collect data files for packages that need them
print("Collecting data files...")
data_packages = ['transformers', 'tokenizers', 'duckdb', 'timm', 'doctr']
for pkg in data_packages:
    if pkg in auto_packages or pkg in hidden_imports:
        try:
            print(f"  Collecting data files for {pkg}...")
            datas += collect_data_files(pkg, include_py_files=False)
        except Exception as e:
            print(f"  Warning: Could not collect data files for {pkg}: {e}")
            pass

print(f"Total data entries collected: {len(datas)}")

# Binary files (empty - let PyInstaller auto-detect)
binaries = []

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
        # Exclude large unused packages to reduce build size
        # (auto-excluded if not in requirements.txt)
        'matplotlib' if 'matplotlib' not in auto_packages else None,
        'scipy' if 'scipy' not in auto_packages else None,
        'pandas' if 'pandas' not in auto_packages else None,
        'jupyter',
        'notebook',
        'IPython',
        'pytest',
        'test',
        'tests',
        'testing',
        'unittest',
        'distutils',
        'setuptools',
        'wheel',
        'pip',
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
