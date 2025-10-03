# AI Image Tagger

AI-powered batch image captioning with BLIP and R-4B models. Generate accurate descriptions for multiple images with customizable prompts.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.4+-green.svg)

## Features

- 🚀 **Batch Processing** - Process multiple images simultaneously
- 🤖 **Dual AI Models** - BLIP (fast) and R-4B (advanced reasoning)
- ⚙️ **Configurable** - Custom prompts, precision modes, generation parameters
- 💾 **Smart Memory** - On-demand model loading
- 📦 **Export** - Download results as ZIP
- 🎨 **Modern UI** - Dark/light themes, responsive design

![AI Image Tagger Interface](assets/Image%20Tagger.png)

## Quick Start

### Download & Run (Easiest)

1. Download the latest release from [Releases](https://github.com/yourusername/ai-image-tagger/releases)
2. Extract and run:
   - **Windows:** `ai-image-tagger.exe`
   - **Linux:** `./ai-image-tagger`
3. Open `http://localhost:5000` in your browser

**Requirements:** NVIDIA GPU with CUDA 12.4+ drivers

### From Source

```bash
# Clone and setup
git clone https://github.com/yourusername/ai-image-tagger.git
cd ai-image-tagger
./setup-tagger-gpu.sh

# Run
conda activate tagger-gpu
cd backend && python app.py
```

Open `frontend/index.html` in your browser.

## Usage

1. **Upload** - Drag & drop images (JPG, PNG, WebP, BMP)
2. **Configure** - Select model (BLIP/R-4B), add prompts, adjust settings
3. **Generate** - Process and review captions
4. **Export** - Download as ZIP

### Models

**BLIP** - Fast captioning (~1-2s per image, ~2GB VRAM)
**R-4B** - Detailed descriptions (~5-10s per image, 2-16GB VRAM depending on precision)

| Precision | VRAM | Speed | Quality |
|-----------|------|-------|---------|
| float32   | 16GB | Slow  | Best    |
| float16   | 8GB  | Fast  | Excellent |
| 8-bit     | 4GB  | Faster | Very Good |
| 4-bit     | 2GB  | Fastest | Good |

## API

```python
import requests

with open('image.jpg', 'rb') as f:
    response = requests.post('http://localhost:5000/generate',
        files={'image': f},
        data={'model': 'blip', 'prompt': 'Describe this image'}
    )
print(response.json()['caption'])
```

**Endpoints:**
- `POST /generate` - Generate caption
- `GET /models` - List models
- `POST /model/unload` - Free memory
- `GET/POST /config` - Manage saved configs

## Building

**Local build:**
```bash
./build.sh    # Linux
build.bat     # Windows
```

**Release:** Push a tag (`v1.0.0`) to trigger automated builds for both platforms via GitHub Actions.

## Project Structure

```
ai-image-tagger/
├── backend/
│   ├── app.py              # Flask API
│   ├── models/             # BLIP & R-4B adapters
│   └── utils/              # Image processing
├── frontend/               # HTML/CSS/JS interface
├── build.sh / build.bat    # Build scripts
└── .github/workflows/      # Automated releases
```

## Troubleshooting

**Out of Memory:** Use lower precision (8-bit/4-bit) or BLIP model

**CUDA Error:** Update to CUDA 12.4+ drivers for modern GPUs (RTX 40/50 series)

**Models Downloading:** First run downloads ~500MB-2GB to `~/.cache/huggingface/`

## License

MIT License - See LICENSE file

## Credits

- [Salesforce BLIP](https://github.com/salesforce/BLIP)
- [YannQi/R-4B](https://huggingface.co/YannQi/R-4B)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
