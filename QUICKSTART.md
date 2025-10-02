# Quick Start Guide - AI Image Tagger

## For Pre-built Executable Users

### First Time Setup

1. **Download the executable** for your platform from [Releases](https://github.com/yourusername/ai-image-captioner/releases):
   - Windows: `ai-image-tagger-windows-vX.X.X.zip`
   - Linux: `ai-image-tagger-linux-vX.X.X.tar.gz`

2. **Extract the archive**:
   - Windows: Right-click → Extract All
   - Linux: `tar -xzf ai-image-tagger-linux-vX.X.X.tar.gz`

3. **GPU Requirements** (IMPORTANT):
   - NVIDIA GPU with CUDA support required
   - Minimum 4GB VRAM (8GB+ recommended for R-4B model)
   - CUDA drivers must be installed on your system

### Running the Application

#### Windows
1. Navigate to the extracted folder
2. Double-click `ai-image-tagger.exe`
3. A console window will open - keep it running
4. Open `frontend/index.html` in your browser

#### Linux
1. Open terminal in the extracted folder
2. Run: `./ai-image-tagger`
3. Keep the terminal open
4. Open `frontend/index.html` in your browser

**Note:** On first run, the application will download AI models (~500MB-2GB) to `~/.cache/huggingface/`. This is a one-time download.

### Using the Interface

1. **Upload Tab**
   - Drag & drop images or click "Choose Files"
   - Supported formats: JPG, PNG, WebP, BMP
   - Can select multiple files or folders

2. **Options Tab**
   - **Model Selection**:
     - BLIP: Fast, basic captions (~1-2s per image)
     - R-4B: Detailed descriptions (~5-10s per image)
   - **Custom Prompt** (optional): Guide what the AI describes
   - **Advanced Parameters** (R-4B only):
     - Precision: Lower = faster, less VRAM
     - Thinking Mode: Controls detail level
   - **Save/Load Configs**: Store your favorite settings

3. **Results Tab**
   - View generated captions
   - Click images to enlarge
   - Edit captions if needed
   - Click "Download ZIP" to save all images with captions

### Troubleshooting

**"CUDA out of memory" error:**
- Switch to R-4B with 8-bit or 4-bit precision
- Use BLIP model instead
- Close other GPU-intensive programs

**Models downloading slowly:**
- First run downloads ~500MB-2GB from Hugging Face
- Future runs are instant (models cached)

**Backend not connecting:**
- Ensure the executable is running (console window open)
- Check if another program is using port 5000
- Try closing and restarting the executable

**Frontend shows errors:**
- Make sure backend is running first
- Check browser console (F12) for errors
- Try a different browser (Chrome/Firefox recommended)

### Model Precision Guide (R-4B)

| Precision | VRAM | Speed | Quality | Best For |
|-----------|------|-------|---------|----------|
| float32   | ~16GB | Slowest | Best | Maximum quality |
| float16   | ~8GB | Fast | Excellent | Balanced (recommended) |
| bfloat16  | ~8GB | Fast | Excellent | Alternative to float16 |
| 8-bit     | ~4GB | Faster | Very Good | Limited VRAM |
| 4-bit     | ~2GB | Fastest | Good | Low VRAM systems |

### Tips

- **Batch Processing**: Upload all images at once for efficiency
- **Custom Prompts**: Be specific for better results
  - Good: "Describe the colors, objects, and mood"
  - Better: "List all visible objects and their colors in detail"
- **Save Configs**: Save your tested settings for repeated use
- **Keyboard Shortcuts**:
  - Tab: Switch between tabs
  - Enter (in Options): Start processing

### Next Steps

- Read the full [README.md](README.md) for advanced features
- Check [GitHub Issues](https://github.com/yourusername/ai-image-captioner/issues) for help
- Star the repo if you find it useful!

---

**Need to build from source?** See the main README.md for developer instructions.
