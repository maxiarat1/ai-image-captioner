# AI Image Captioner

AI-powered batch image captioning tool with BLIP and R-4B models. Generate accurate descriptions for multiple images with customizable prompts and advanced parameters.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)

## Features

- 🚀 **Batch Processing** - Process multiple images simultaneously
- 🤖 **Dual Model Support**
  - **BLIP** - Fast, efficient image captioning
  - **R-4B** - Advanced reasoning with detailed descriptions
- ⚙️ **Highly Configurable**
  - Custom prompts for guided caption generation
  - Precision modes (FP32, FP16, BFloat16, 4-bit, 8-bit quantization)
  - Adjustable generation parameters
  - Flash Attention 2 support for faster inference
- 💾 **Smart Memory Management** - Load/unload models on-demand
- 📦 **Export Options** - Download results as ZIP with captions
- 🎨 **Modern UI** - Clean, responsive interface with dark/light themes
- 🔧 **Configuration Presets** - Save and load your favorite settings

## Screenshots

![AI Image Tagger Interface](assets/Image%20Tagger.png)

*The application features a clean, tab-based interface for uploading images, configuring models, and viewing results.*

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ GPU VRAM recommended for R-4B model
- Conda (recommended for environment management)

### Setup Script (Recommended)

The easiest way to set up the project:

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-image-captioner.git
cd ai-image-captioner

# Run the automated setup script
chmod +x setup-tagger-gpu.sh
./setup-tagger-gpu.sh
```

The setup script will:
- Create a conda environment named `tagger-gpu`
- Install PyTorch with CUDA support
- Install all required dependencies
- Optionally install Flash Attention 2 for faster inference

### Manual Installation

If you prefer manual setup:

```bash
# Create conda environment
conda create -n tagger-gpu python=3.10 -y
conda activate tagger-gpu

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install dependencies
pip install flask flask-cors pillow transformers accelerate bitsandbytes

# Optional: Install Flash Attention 2 (requires compatible GPU)
pip install flash-attn --no-build-isolation
```

## Usage

### Starting the Application

1. **Activate the environment:**
   ```bash
   conda activate tagger-gpu
   ```

2. **Start the backend server:**
   ```bash
   cd backend
   python app.py
   ```
   The Flask server will start on `http://localhost:5000`

3. **Open the frontend:**
   - Simply open `frontend/index.html` in your browser
   - Or serve it with a local server:
     ```bash
     cd frontend
     python -m http.server 8000
     ```
     Then visit `http://localhost:8000`

### Workflow

1. **Upload Tab**
   - Drag & drop images or click to browse
   - Support for JPG, PNG, WebP, BMP formats
   - Add individual files or entire folders

2. **Options Tab**
   - Select AI model (BLIP or R-4B)
   - Add custom prompts (optional)
   - Configure advanced parameters:
     - Precision mode (for R-4B)
     - Generation settings (temperature, max tokens, etc.)
     - Thinking mode (auto/short/long)
   - Save/load configuration presets

3. **Results Tab**
   - View generated captions
   - Preview images in modal view
   - Edit captions if needed
   - Download all as ZIP file

### API Endpoints

The backend provides a RESTful API:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Generate caption for an image |
| `/models` | GET | List available models and status |
| `/model/reload` | POST | Reload model with new settings |
| `/model/unload` | POST | Unload model to free memory |
| `/config` | GET | Get saved configurations |
| `/config` | POST | Save new configuration |

### Example API Usage

```python
import requests

# Generate caption
with open('image.jpg', 'rb') as f:
    response = requests.post('http://localhost:5000/generate',
        files={'image': f},
        data={
            'model': 'r4b',
            'prompt': 'Describe this image in detail',
            'parameters': '{"precision": "float16"}'
        }
    )

result = response.json()
print(result['caption'])
```

## Models

### BLIP (Salesforce/blip-image-captioning-base)
- **Speed:** Fast (~1-2s per image)
- **Use Case:** Quick captioning, batch processing
- **Memory:** ~2GB VRAM
- **Features:** Optional text prompts for guided generation

### R-4B (YannQi/R-4B)
- **Speed:** Slower (~5-10s per image, varies by precision)
- **Use Case:** Detailed descriptions, reasoning tasks
- **Memory:** 4GB-16GB VRAM (depends on precision)
- **Features:**
  - Multiple precision modes for speed/quality tradeoff
  - Thinking modes (includes reasoning process)
  - Flash Attention 2 support
  - Highly configurable generation parameters

### Precision Modes (R-4B)

| Mode | VRAM Usage | Speed | Quality |
|------|------------|-------|---------|
| float32 | ~16GB | Slowest | Best |
| float16 | ~8GB | Fast | Excellent |
| bfloat16 | ~8GB | Fast | Excellent |
| 8-bit | ~4GB | Faster | Very Good |
| 4-bit | ~2GB | Fastest | Good |

## Configuration

### User Config (`user_config.json`)

The application automatically saves your preferences:

```json
{
  "saved_prompts": {
    "detailed": "Describe this image in detail, including objects, colors, and atmosphere",
    "simple": "What is in this image?"
  },
  "saved_configs": {
    "fast": {
      "model": "blip",
      "parameters": {}
    },
    "quality": {
      "model": "r4b",
      "parameters": {
        "precision": "float16",
        "max_new_tokens": 200
      }
    }
  }
}
```

## Troubleshooting

### CUDA Out of Memory
- Switch to lower precision (8-bit or 4-bit for R-4B)
- Reduce `max_new_tokens` parameter
- Use BLIP model instead of R-4B
- Unload unused models via `/model/unload` endpoint

### Flash Attention Installation Fails
- Ensure CUDA toolkit is properly installed
- Check GPU compatibility (compute capability 7.5+)
- Flash Attention is optional - the app works without it

### Model Loading Slow
- Models are downloaded on first use (~500MB-2GB)
- Subsequent loads are much faster
- Models load on-demand, not at startup

### CORS Errors
- Ensure backend is running on `http://localhost:5000`
- Check that CORS is enabled in `backend/app.py`

## Project Structure

```
ai-image-captioner/
├── backend/
│   ├── app.py                 # Flask API server
│   ├── models/
│   │   ├── base_adapter.py    # Abstract model interface
│   │   ├── blip_adapter.py    # BLIP model adapter
│   │   └── r4b_adapter.py     # R-4B model adapter
│   └── utils/
│       └── image_utils.py     # Image processing utilities
├── frontend/
│   ├── index.html             # Main UI
│   ├── script.js              # Client-side logic
│   └── styles.css             # Styling (dark/light themes)
├── setup-tagger-gpu.sh        # Automated setup script
├── user_config.json           # User preferences (auto-generated)
└── README.md
```

## Development

### Adding a New Model

1. Create a new adapter in `backend/models/`:
   ```python
   from models.base_adapter import BaseModelAdapter

   class CustomModelAdapter(BaseModelAdapter):
       def load_model(self):
           # Load your model
           pass

       def generate_caption(self, image_path, prompt=None, **kwargs):
           # Generate caption
           pass
   ```

2. Register in `backend/app.py`:
   ```python
   'custom': CustomModelAdapter()
   ```

### Frontend Customization

The UI uses CSS variables for easy theming:
- Edit `frontend/styles.css` to modify colors, spacing, etc.
- Theme switching is built-in (dark/light modes)

## Performance Tips

- **GPU Acceleration:** Ensure CUDA is properly configured
- **Batch Processing:** Process multiple images in sequence for efficiency
- **Precision:** Use float16/bfloat16 for best speed/quality balance
- **Flash Attention:** Install for 2-4x faster inference on compatible GPUs
- **Model Preloading:** Models load on first request - expect initial delay

## License

MIT License - see LICENSE file for details

## Acknowledgments

- [Salesforce BLIP](https://github.com/salesforce/BLIP) - Image captioning model
- [YannQi/R-4B](https://huggingface.co/YannQi/R-4B) - Advanced reasoning model
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - Model framework
- [Flask](https://flask.palletsprojects.com/) - Backend framework

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Support

If you encounter any issues or have questions:
- Open an [issue](https://github.com/yourusername/ai-image-captioner/issues)
- Check existing issues for solutions
- Review the troubleshooting section above

---

**Note:** This project requires a CUDA-capable GPU for optimal performance. CPU inference is possible but significantly slower.
