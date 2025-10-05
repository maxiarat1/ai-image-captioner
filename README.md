# AI Image Tagger

Batch image captioning using BLIP and R-4B models. Generate descriptions for multiple images with customizable prompts.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.1%20%7C%2012.8-green.svg)

## Features

- Batch image processing
- Dual AI models: BLIP (fast) and R-4B (advanced reasoning)
- Custom prompts, precision modes, generation parameters
- On-demand model loading
- ZIP export
- Dark/light themes

![AI Image Tagger Interface](assets/Image%20Tagger.png)

## Quick Start

### Download & Run

1. Download from [Releases](https://github.com/maxiarat1/ai-image-captioner/releases)
2. Extract and run:
   - Windows: `ai-image-tagger.exe`
   - Linux: `./ai-image-tagger`
3. Navigate to `http://localhost:5000`

Requirements: CUDA 12.1+ (RTX 20/30/40) or CUDA 12.8+ (RTX 50)

### From Source

```bash
git clone https://github.com/maxiarat1/ai-image-captioner.git
cd ai-image-captioner
./setup-tagger-gpu.sh
conda activate tagger-gpu
cd backend && python app.py
```

Navigate to `http://localhost:5000`

## Usage

1. Upload images (JPG, PNG, WebP, BMP)
2. Select model (BLIP/R-4B), configure prompts and settings
3. Generate and review captions
4. Export as ZIP

### Models

**BLIP:** Fast captioning (1-2s/image, 2GB VRAM)

**R-4B:** Detailed descriptions (5-10s/image, VRAM varies by precision)

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

Endpoints:
- `POST /generate` - Generate caption
- `GET /models` - List models
- `POST /model/unload` - Free memory
- `GET/POST /config` - Manage configs

## Building

Local build:
```bash
./build.sh python310-cuda121      # Linux - RTX 20/30/40
./build.sh python312-cuda128      # Linux - RTX 50
build.bat python310-cuda121       # Windows - RTX 20/30/40
build.bat python312-cuda128       # Windows - RTX 50
```

Release all configs:
```bash
./release.sh -v 1.0.2
```

Builds all Python/CUDA combinations from `version.json`:
- 2 configs × 2 platforms = 4 executables
- 2 Docker images
- GitHub release with all artifacts

Configs:
- `python310-cuda121` - RTX 20/30/40 (CUDA 12.1)
- `python312-cuda128` - RTX 50 (CUDA 12.8, sm_120)

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

**Out of Memory:** Use lower precision (8-bit/4-bit) or BLIP

**CUDA Error:**
- RTX 20/30/40: CUDA 12.1+ drivers, `cuda121` build
- RTX 50: CUDA 12.8+ drivers, `cuda128` build

**Models Downloading:** First run downloads 500MB-2GB to `~/.cache/huggingface/`

**Wrong GPU Architecture:** RTX 50 "sm_120 not supported" error requires `python312-cuda128` build

## License

MIT License - See LICENSE file

## Credits

- [Salesforce BLIP](https://github.com/salesforce/BLIP)
- [YannQi/R-4B](https://huggingface.co/YannQi/R-4B)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
