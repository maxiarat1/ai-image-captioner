@echo off
REM Build script for AI Image Captioner (Windows)

setlocal enabledelayedexpansion

REM Read version info (requires jq on Windows)
for /f "delims=" %%i in ('jq -r .app_version version.json 2^>nul') do set APP_VERSION=%%i
if "%APP_VERSION%"=="" set APP_VERSION=dev

set BUILD_CONFIG=%1
if "%BUILD_CONFIG%"=="" set BUILD_CONFIG=default

REM Validate config
jq -e ".build_configs.%BUILD_CONFIG%" version.json >nul 2>&1
if errorlevel 1 (
    echo Error: Invalid build config '%BUILD_CONFIG%'
    for /f "delims=" %%i in ('jq -r ".build_configs | keys | join(\", \")" version.json') do echo Available: %%i
    exit /b 1
)

for /f "delims=" %%i in ('jq -r ".build_configs.%BUILD_CONFIG%.python" version.json') do set PYTHON_VER=%%i
for /f "delims=" %%i in ('jq -r ".build_configs.%BUILD_CONFIG%.cuda_version_display" version.json') do set CUDA_DISPLAY=%%i

echo Building AI Image Captioner v%APP_VERSION%
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
echo Build complete: dist\ai-image-captioner\
echo.
echo To run: dist\ai-image-captioner\ai-image-captioner.exe
