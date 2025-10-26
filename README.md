# AI Image Captioner

Build and test image captioning pipelines with diverse AI models using a flexible, node-based visual interface.

![License](https://img.shields.io/badge/license-MIT-purple.svg)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.1%20%7C%2012.8-*.svg)


![AI Image Captioner Interface](assets/Image%20Tagger.png)

## Features

- **Visual Node Editor** - Build AI pipelines with drag-and-drop nodes
- **Dual AI Models** - BLIP (fast) & R-4B (detailed reasoning) and many more
- **Smart Templates** - Combine prompts and AI outputs with Conjunction nodes
- **Real-time Stats** - Live progress tracking with speed & ETA
- **Flexible Export** - ZIP with embedded EXIF/PNG metadata

## Quick Start

### Option 1: Docker (Recommended)

```bash
CUDA 12.1
docker run --gpus all -p 5000:5000 ghcr.io/maxiarat1/ai-image-captioner:latest-python310-cuda121

CUDA 12.8
docker run --gpus all -p 5000:5000 ghcr.io/maxiarat1/ai-image-captioner:latest-python312-cuda128
```
Note: It needs [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Option 2: Download Executable

1. Download from [Releases](https://github.com/maxiarat1/ai-image-captioner/releases)
2. Run the executable:
   - Windows: `ai-image-captioner.exe`
   - Linux: `./ai-image-captioner`
3. Open `frontend/index.html` in your browser

### Option 3: From Source

```bash
git clone https://github.com/maxiarat1/ai-image-captioner.git
cd ai-image-captioner
./setup-captioner-gpu.sh
conda activate captioner-gpu
cd backend && python app.py
```

## Usage

1. Upload images to the Input node
2. Connect nodes: Input → AI Model → Output
3. Add Prompt or Conjunction nodes for custom templates
4. Click Process and monitor progress
5. Export results as ZIP with embedded captions

**Example pipeline:**
```
Input → Prompt → BLIP → Conjunction → Output
```

## [Flash Attention Instalation](https://github.com/Dao-AILab/flash-attention)
```
conda activate captioner-gpu 
sudo apt install nvidia-cuda-toolkit
nvcc --version
pip install flash-attn --use-pep517 --no-build-isolation
```
Building flash-attn from source can take a long time, typically 30–60 minutes on mid-range hardware.
If you want a much faster installation, use a prebuilt wheel matching your CUDA and PyTorch version (recommended when available).

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Use 4-bit/8-bit precision or BLIP |
| CUDA error | Match build to GPU: RTX 20/30/40 use cuda121, RTX 50 use cuda128 |
| Slow first run | Models download to `~/.cache/huggingface/` (~2GB) |
| Docker GPU error | Install nvidia-container-toolkit |