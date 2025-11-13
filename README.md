# AI Image Captioner

Build and test image captioning pipelines with diverse AI models using a flexible, node-based visual interface.

![License](https://img.shields.io/badge/license-MIT-purple.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.8-green.svg)


![AI Image Captioner Interface](assets/Image%20Tagger.png)

## Features

- **Visual Node Editor** - Build AI pipelines with drag-and-drop nodes
- **AI Models** - BLIP & R-4B (detailed reasoning) and many more options
- **Smart Templates** - Combine prompts and AI outputs with Conjunction nodes
- **Real-time Stats** - Live progress tracking with speed & ETA
- **Flexible Export** - ZIP with embedded EXIF/PNG metadata

## Quick Start

### Option 1: Docker (Recommended)
- Pull and run pre-built image
```bash
docker run --gpus all -p 5000:5000 \
  -v ai-captioner-data:/app/backend/data \
  -v ai-captioner-thumbnails:/app/backend/thumbnails \
  -v huggingface-cache:/root/.cache/huggingface \
  ghcr.io/maxiarat1/ai-image-captioner:latest-python312-cuda128
```
- OR build locally with docker-compose
```bash
git clone https://github.com/maxiarat1/ai-image-captioner.git
cd ai-image-captioner
docker-compose up
```
Note: Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Option 2: Download Executable

1. Download from [Releases](https://github.com/maxiarat1/ai-image-captioner/releases)
2. Extract the archive:
   - **Windows**: If split archives (`.zip.001`, `.zip.002`), use 7-Zip to extract `.zip.001`
   - **Linux**: If split archives (`.tar.gz.partaa`, `.tar.gz.partab`):
     ```bash
     cat ai-image-captioner-linux-*.tar.gz.part* > ai-image-captioner.tar.gz
     tar -xzf ai-image-captioner.tar.gz
     ```
3. Run the executable:
   - Windows: `ai-image-captioner.exe`
   - Linux: `./ai-image-captioner/ai-image-captioner`
4. Open `frontend/index.html` in your browser

### Option 3: From Source

```bash
git clone https://github.com/maxiarat1/ai-image-captioner.git
cd ai-image-captioner
./setup-captioner-gpu.sh
conda activate captioner-gpu
python backend/app.py
```

## Usage

https://github.com/user-attachments/assets/5d59ddf6-b471-4e3b-a7dd-146893297a79

1. Upload images to the Input node
2. Connect nodes: Input → AI Model → Output
3. Add Prompt or Conjunction nodes for custom templates
4. Click Process and monitor progress
5. Export results

**Example pipeline:**
```
Images ┐  
       │—→ BLIP —→ Output
Prompt ┘
```
## Project Overview

Hi! This project is alive and growing. I’m building it to make it easy to connect small, efficient models into simple pipelines, so together they can pull more detail out of images than any single model could on its own.

### A few notes

* There are still some rough edges while I refactor a fairly large codebase (~60k lines).
* Performance and memory usage are improving steadily. The first run may be slow while models download.
* It runs fastest on GPU, but CPU works too. If you’re short on VRAM, try 4-bit or 8-bit modes.

### What’s coming next

* More nodes and adapters (OCR, VLMs, and utility helpers)
* Better multi-model workflows and templates for richer image captions
* Cleaner docs, more example pipelines, and a few quality-of-life fixes

### How to contribute

* Open an issue, please include steps, logs, and screenshots
* Submit a PR for a bug fix, performance tweak, doc improvement, or a new node/adapter
* Share any pipeline setups that work well for you

Thanks for checking this out! I read the issues and try to ship improvements regularly, your feedback genuinely shapes where the project goes next.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Use 4-bit/8-bit precision|
| Slow first run | Models download to `~/.cache/huggingface/` (~2GB) |
| Docker GPU error | Install nvidia-container-toolkit |
