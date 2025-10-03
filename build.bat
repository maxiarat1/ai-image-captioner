@echo off
REM Simple build script for AI Image Tagger (Windows)

echo Building AI Image Tagger...

REM Activate conda environment
call conda activate tagger-gpu 2>nul

REM Install PyInstaller if needed
pip install -q pyinstaller

REM Clean and build
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
cd backend
pyinstaller tagger.spec --distpath ..\dist --workpath ..\build
cd ..

echo.
echo Build complete: dist\ai-image-tagger\
echo.
echo To run: dist\ai-image-tagger\ai-image-tagger.exe
