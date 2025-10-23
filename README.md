# AI Image Tagger

Image captioning using BLIP and R-4B models. Generate descriptions for images with customizable prompts.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.1%20%7C%2012.8-green.svg)

## Features

- **Visual Node Editor** - Build AI pipelines with drag-and-drop nodes
- **Dual AI Models** - BLIP (fast) & R-4B (detailed reasoning)
- **Smart Templates** - Combine prompts and AI outputs with Conjunction nodes
- **Real-time Stats** - Live progress tracking with speed & ETA
- **Flexible Export** - ZIP with embedded metadata

![AI Image Tagger Interface](assets/Image%20Tagger.png)

## Quick Start

### Download & Run

**Backend (API Server):**

Choose one method:

**Executable (Easiest):**
1. Download from [Releases](https://github.com/maxiarat1/ai-image-captioner/releases)
2. Extract and run:
   - Windows: `ai-image-tagger.exe`
   - Linux: `./ai-image-tagger`

**Docker:**
```bash
# RTX 20/30/40 series
docker run --gpus all -p 5000:5000 ghcr.io/maxiarat1/ai-image-captioner:latest-python310-cuda121

# RTX 50 series
docker run --gpus all -p 5000:5000 ghcr.io/maxiarat1/ai-image-captioner:latest-python312-cuda128
```

API runs at `http://localhost:5000`

**Frontend:**

Serve the `frontend/` directory with any web server or open `index.html` directly. Configure API endpoint to point to `http://localhost:5000`.

Requirements: CUDA 12.1+ (RTX 20/30/40) or CUDA 12.8+ (RTX 50)

### From Source

**Backend:**
```bash
git clone https://github.com/maxiarat1/ai-image-captioner.git
cd ai-image-captioner
./setup-tagger-gpu.sh
conda activate tagger-gpu
cd backend && python app.py
```

API available at `http://localhost:5000`

**Frontend:**
```bash
# Any web server, e.g.:
cd frontend
python -m http.server 8080
# Or open index.html directly in browser
```

## Usage

### Node Editor Workflow

1. **Upload** - Add images to the Input node
2. **Connect** - Link Input → AI Model → Output nodes
3. **Customize** - Add Prompt or Conjunction nodes for templates
4. **Process** - Execute and watch real-time stats in Output node
5. **Export** - Download ZIP with embedded captions

**Example Pipeline:**
```
Input → Prompt → AI Model (BLIP) → Conjunction → Output
                                         ↑
                                    Prompt (style guide)
```
Conjunction template: `Caption: {AI_Model}. Style: {Prompt}`

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

The backend exposes a REST API at `http://localhost:5000`:

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
- `POST /model/reload` - Reload with new settings
- `POST /model/unload` - Free memory
- `GET/POST /config` - Manage configs
- `GET /health` - Health check

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

| Issue | Solution |
|-------|----------|
| Out of Memory | Use 8-bit/4-bit precision or BLIP model |
| CUDA Error | RTX 20/30/40: `cuda121` build, RTX 50: `cuda128` build |
| First Run Slow | Models download ~2GB to `~/.cache/huggingface/` |
| Docker GPU Error | Install `nvidia-container-toolkit`, use Docker Engine (not Desktop) |

**Docker GPU Setup (Linux):**
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit && sudo systemctl restart docker
```

## License

MIT License - See LICENSE file

## Credits

- [Salesforce BLIP](https://github.com/salesforce/BLIP)
- [YannQi/R-4B](https://huggingface.co/YannQi/R-4B)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
