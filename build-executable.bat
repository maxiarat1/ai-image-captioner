@echo off
REM Build script for AI Image Tagger executable (Windows)
REM This script builds a standalone executable using PyInstaller

echo ==========================================
echo AI Image Tagger - Build Script (Windows)
echo ==========================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: conda is not installed or not in PATH
    echo Please install Miniconda or Anaconda first
    pause
    exit /b 1
)

REM Activate the tagger-gpu environment
echo Activating tagger-gpu environment...
call conda activate tagger-gpu
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to activate tagger-gpu environment
    echo Please run setup-tagger-gpu.sh first to create the environment
    pause
    exit /b 1
)

REM Check if PyInstaller is installed
python -c "import PyInstaller" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Installing PyInstaller...
    pip install pyinstaller
) else (
    echo PyInstaller is already installed
)

REM Clean previous builds
echo.
echo Cleaning previous builds...
if exist build-output rmdir /s /q build-output

REM Build the executable
echo.
echo Building executable with PyInstaller...
echo (This may take 5-15 minutes depending on your system)
echo.

cd backend
pyinstaller tagger.spec --distpath ..\build-output\dist --workpath ..\build-output\build
cd ..

REM Check if build was successful
if exist "build-output\dist\ai-image-tagger" (
    echo.
    echo ==========================================
    echo Build successful!
    echo ==========================================
    echo.
    echo Executable location: build-output\dist\ai-image-tagger\
    echo Main executable: build-output\dist\ai-image-tagger\ai-image-tagger.exe
    echo.
    echo To run the application:
    echo   cd build-output\dist\ai-image-tagger
    echo   ai-image-tagger.exe
    echo.
    echo To create a distributable ZIP:
    echo   cd build-output\dist
    echo   powershell Compress-Archive -Path ai-image-tagger -DestinationPath ai-image-tagger-windows.zip
    echo.

    REM Calculate size (PowerShell command)
    for /f "tokens=*" %%i in ('powershell -command "(Get-ChildItem -Path 'build-output\dist\ai-image-tagger' -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB"') do set SIZE=%%i
    echo Package size: ~%SIZE% MB
    echo.
    echo Note: Models will be downloaded on first run (~500MB-2GB^)
    echo       to %USERPROFILE%\.cache\huggingface\
) else (
    echo.
    echo Build failed! Check the output above for errors.
    pause
    exit /b 1
)

pause
