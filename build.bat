@echo off
REM Build script for AI Image Captioner (Windows)

setlocal enabledelayedexpansion

REM Validate version.json exists
if not exist version.json (
    echo Error: version.json not found in current directory
    exit /b 1
)

REM Get default config from version.json (first entry)
for /f "delims=" %%i in ('jq -r ".build_configs | keys | first" version.json') do set DEFAULT_CONFIG=%%i

REM Read build config name (first argument or default from version.json)
set BUILD_CONFIG=%1
if "%BUILD_CONFIG%"=="" set BUILD_CONFIG=%DEFAULT_CONFIG%

REM Validate config exists
jq -e ".build_configs[\"%BUILD_CONFIG%\"]" version.json >nul 2>&1
if errorlevel 1 (
    echo Error: Invalid build config '%BUILD_CONFIG%'
    for /f "delims=" %%i in ('jq -r ".build_configs | keys | join(\", \")" version.json') do echo Available: %%i
    exit /b 1
)

REM Read config values
for /f "delims=" %%i in ('jq -r ".build_configs[\"%BUILD_CONFIG%\"].python" version.json') do set PYTHON_VER=%%i
for /f "delims=" %%i in ('jq -r ".build_configs[\"%BUILD_CONFIG%\"].cuda_version_display" version.json') do set CUDA_DISPLAY=%%i

echo Building AI Image Captioner
echo Config: %BUILD_CONFIG% (Python %PYTHON_VER%, CUDA %CUDA_DISPLAY%)
echo.

REM Activate conda environment
call conda activate captioner-gpu 2>nul

REM Install PyInstaller if needed
pip install -q pyinstaller

REM Clean and build
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
cd backend
pyinstaller captioner.spec --distpath ..\dist --workpath ..\build
cd ..

echo.
echo ✅ Build complete: dist\ai-image-captioner\
echo.
echo To run: dist\ai-image-captioner\ai-image-captioner.exe
echo To package: tar -czf ai-image-captioner.tar.gz -C dist ai-image-captioner
