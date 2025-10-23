import io
import base64
import json
import zipfile
import os
import logging
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
from PIL.PngImagePlugin import PngInfo
"""
Set CUDA visibility before importing any modules that might import torch/transformers.
Users can force CPU by exporting TAGGER_FORCE_CPU=1 before launching the app.
This prevents PyTorch from attempting forward-compat CUDA init on unsupported GPUs.
"""
if os.environ.get("TAGGER_FORCE_CPU", "0") == "1":
    # Hide all CUDA devices from PyTorch/Transformers
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    # Be tolerant on Apple MPS or other backends if present
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from models.blip_adapter import BlipAdapter
from models.r4b_adapter import R4BAdapter
from models.qwen3vl_adapter import Qwen3VLAdapter
from models.wdvit_adapter import WdVitAdapter
from utils.image_utils import load_image, image_to_base64
from utils.logging_utils import setup_logging, compact_json

# Configure logging early
setup_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Constants
SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
THUMBNAIL_SIZE = (150, 150)
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Configuration file path
CONFIG_FILE = Path(__file__).parent.parent / "user_config.json"

# Default configuration template
DEFAULT_CONFIG = {
    "version": "1.0",
    "savedConfigurations": {},
    "customPrompts": [
        {
            "id": "default-detailed",
            "name": "Detailed Description",
            "text": "Provide a detailed description of this image, including objects, people, colors, setting, and any notable details.",
            "createdAt": "2024-01-01T00:00:00.000Z"
        },
        {
            "id": "default-simple",
            "name": "Simple Description",
            "text": "Describe this image in a few simple sentences.",
            "createdAt": "2024-01-01T00:00:00.000Z"
        },
        {
            "id": "default-artistic",
            "name": "Artistic Analysis",
            "text": "Analyze this image from an artistic perspective, describing composition, lighting, style, and mood.",
            "createdAt": "2024-01-01T00:00:00.000Z"
        }
    ],
    "lastModified": "2024-01-01T00:00:00.000Z"
}

# Precision parameter defaults
PRECISION_DEFAULTS = {
    'r4b': {'precision': 'float32', 'use_flash_attention': False},
    'qwen3vl-4b': {'precision': 'auto', 'use_flash_attention': False},
    'qwen3vl-8b': {'precision': 'auto', 'use_flash_attention': False}
}

# Model registry and metadata
MODEL_METADATA = {
    'blip': {
        'description': "Fast, basic image captioning",
        'adapter': BlipAdapter,
        'adapter_args': {}
    },
    'r4b': {
        'description': "Advanced reasoning model with configurable parameters",
        'adapter': R4BAdapter,
        'adapter_args': {}
    },
    'qwen3vl-4b': {
        'description': "Qwen3-VL 4B - Compact vision-language model with strong performance",
        'adapter': Qwen3VLAdapter,
        'adapter_args': {'model_id': "Qwen/Qwen3-VL-4B-Instruct"}
    },
    'qwen3vl-8b': {
        'description': "Qwen3-VL 8B - Advanced vision-language model with superior image understanding",
        'adapter': Qwen3VLAdapter,
        'adapter_args': {'model_id': "Qwen/Qwen3-VL-8B-Instruct"}
    },
    'wdvit': {
        'description': "WD-ViT Large Tagger v3 - Anime-style image tagging model with ViT backbone",
        'adapter': WdVitAdapter,
        'adapter_args': {'model_id': "SmilingWolf/wd-vit-large-tagger-v3"}
    },
    'wdeva02': {
        'description': "WD-EVA02 Large Tagger v3 - Anime-style image tagging model with EVA02 backbone (improved accuracy)",
        'adapter': WdVitAdapter,
        'adapter_args': {'model_id': "SmilingWolf/wd-eva02-large-tagger-v3"}
    }
}

# Global model instances (loaded on-demand)
models = {name: None for name in MODEL_METADATA.keys()}

# Cache for model info to avoid recreating adapters
_model_info_cache = {}

def validate_model_name(model_name: str) -> bool:
    """Validate if model name exists"""
    return model_name in MODEL_METADATA

def init_models():
    """Initialize model registry (models will be loaded on-demand)"""
    logger.info("Model registry ready. Models load on first use.")
    logger.info("Available models: %s", ", ".join(MODEL_METADATA.keys()))

def get_model(model_name, precision_params=None, force_reload=False):
    """Get or load a specific model with optional precision parameters"""
    if not validate_model_name(model_name):
        raise ValueError(f"Unknown model: {model_name}")

    # Unload other models to free VRAM when switching
    for other_name, other_model in models.items():
        if other_name != model_name and other_model is not None:
            logger.info("Switching model: unloading %s", other_name)
            other_model.unload()
            models[other_name] = None

    # Check if reload needed due to precision changes
    should_reload = force_reload or _needs_precision_reload(model_name, precision_params)
    
    if models[model_name] is None or should_reload:
        _load_model(model_name, precision_params, should_reload)

    return models[model_name]

def _needs_precision_reload(model_name: str, precision_params: dict) -> bool:
    """Check if model needs reload due to precision parameter changes"""
    if not precision_params or model_name not in PRECISION_DEFAULTS:
        return False
        
    current_model = models[model_name]
    if current_model is None or not hasattr(current_model, 'current_precision_params'):
        return False
        
    return current_model.current_precision_params != precision_params

def _load_model(model_name: str, precision_params: dict, is_reload: bool):
    """Load or reload a model with given parameters"""
    action = "Reloading" if is_reload else "Loading"
    
    # Unload existing model if reloading
    if is_reload and models[model_name] is not None:
        models[model_name].unload()
        models[model_name] = None

    try:
        metadata = MODEL_METADATA[model_name]
        logger.info("%s %s model on-demand…", action, model_name)
        
        # Create adapter instance
        adapter_class = metadata['adapter']
        adapter_args = metadata['adapter_args']
        models[model_name] = adapter_class(**adapter_args)
        
        # Load model with precision parameters
        if precision_params:
            models[model_name].load_model(**precision_params)
            models[model_name].current_precision_params = precision_params.copy()
        else:
            models[model_name].load_model()
            # Store default precision params if applicable
            if model_name in PRECISION_DEFAULTS:
                models[model_name].current_precision_params = PRECISION_DEFAULTS[model_name]
                
    except Exception as e:
        logger.exception("Failed to load %s model: %s", model_name, e)
        models[model_name] = None
        raise

def load_user_config():
    """Load user configuration from file"""
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Create and save default config
        default_config = DEFAULT_CONFIG.copy()
        save_user_config(default_config)
        return default_config
        
    except Exception as e:
        logger.exception("Error loading user config: %s", e)
        return DEFAULT_CONFIG.copy()

def save_user_config(config_data):
    """Save user configuration to file"""
    try:
        # Ensure parent directory exists
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Add timestamp
        config_data["lastModified"] = datetime.now().isoformat()

        # Write to file with proper formatting
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        logger.info("User config saved to: %s", CONFIG_FILE)
        return True
    except Exception as e:
        logger.exception("Error saving user config: %s", e)
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_status = {
        f"{name}_loaded": model is not None and model.is_loaded()
        for name, model in models.items()
    }
    
    return jsonify({
        "status": "ok",
        "models_available": list(models.keys()),
        **model_status
    })

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information including available parameters"""
    model_name = request.args.get('model', 'blip')

    if not validate_model_name(model_name):
        return jsonify({"error": f"Unknown model: {model_name}"}), 400

    # Use cached info if available
    if model_name not in _model_info_cache:
        try:
            metadata = MODEL_METADATA[model_name]
            adapter = metadata['adapter'](**metadata['adapter_args'])
            _model_info_cache[model_name] = {
                "name": model_name,
                "parameters": adapter.get_available_parameters()
            }
        except Exception as e:
            logger.exception("Error getting model info: %s", e)
            return jsonify({"error": str(e)}), 500

    return jsonify(_model_info_cache[model_name])

@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    available_models = [
        {
            "name": name,
            "loaded": models[name] is not None and models[name].is_loaded(),
            "description": MODEL_METADATA[name]['description']
        }
        for name in MODEL_METADATA.keys()
    ]

    return jsonify({"models": available_models})

@app.route('/model/reload', methods=['POST'])
def reload_model():
    """Force reload a model with new settings"""
    try:
        data = request.get_json() or {}
        model_name = data.get('model', 'r4b')

        if not validate_model_name(model_name):
            return jsonify({"error": f"Unknown model: {model_name}"}), 400

        precision_params = data.get('precision_params')
        model_adapter = get_model(model_name, precision_params, force_reload=True)

        return jsonify({
            "success": True,
            "message": f"Model {model_name} reloaded successfully",
            "loaded": model_adapter.is_loaded()
        })

    except Exception as e:
        logger.exception("Error reloading model: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/model/unload', methods=['POST'])
def unload_model():
    """Unload a model to free memory"""
    try:
        data = request.get_json() or {}
        model_name = data.get('model', 'r4b')

        if not validate_model_name(model_name):
            return jsonify({"error": f"Unknown model: {model_name}"}), 400

        if models[model_name] is not None:
            models[model_name].unload()
            models[model_name] = None

        return jsonify({
            "success": True,
            "message": f"Model {model_name} unloaded successfully"
        })

    except Exception as e:
        logger.exception("Error unloading model: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/config', methods=['GET', 'POST'])
def user_config():
    """Get or save user configuration"""
    try:
        if request.method == 'GET':
            config = load_user_config()
            return jsonify({
                "success": True,
                "config": config,
                "configFile": str(CONFIG_FILE)
            })
        
        # POST - Save config
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Validate required fields
        required_fields = ['savedConfigurations', 'customPrompts']
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        if not save_user_config(data):
            return jsonify({"error": "Failed to save configuration"}), 500

        return jsonify({
            "success": True,
            "message": "Configuration saved successfully",
            "configFile": str(CONFIG_FILE),
            "lastModified": data.get("lastModified")
        })

    except Exception as e:
        logger.exception("Error handling config: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/config/backup', methods=['POST'])
def backup_user_config():
    """Create a backup of the current config file"""
    try:
        if not CONFIG_FILE.exists():
            return jsonify({"error": "No config file exists to backup"}), 404

        backup_name = f"user_config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        backup_path = CONFIG_FILE.parent / backup_name

        # Copy config to backup
        import shutil
        shutil.copy2(CONFIG_FILE, backup_path)

        return jsonify({
            "success": True,
            "message": "Backup created successfully",
            "backupFile": str(backup_path)
        })

    except Exception as e:
        logger.exception("Error creating backup: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/scan-folder', methods=['POST'])
def scan_folder():
    """Scan a folder and return image metadata"""
    try:
        folder_path = request.get_json().get('folder_path', '')
        if not folder_path:
            return jsonify({"error": "No folder path provided"}), 400

        folder = Path(folder_path).resolve()

        # Validate path
        if not folder.exists():
            return jsonify({"error": "Folder does not exist"}), 404
        if not folder.is_dir():
            return jsonify({"error": "Path is not a directory"}), 400

        # Scan for supported images
        images = [
            {
                'filename': f.name,
                'path': str(f),
                'size': f.stat().st_size
            }
            for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_FORMATS
        ]

        # Sort by filename
        images.sort(key=lambda x: x['filename'])

        return jsonify({
            "success": True,
            "folder": str(folder),
            "images": images,
            "count": len(images)
        })

    except Exception as e:
        logger.exception("Error scanning folder: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/image/thumbnail', methods=['GET'])
def get_thumbnail():
    """Generate thumbnail for an image"""
    try:
        image_path = request.args.get('path', '')
        if not image_path:
            return jsonify({"error": "No image path provided"}), 400

        # Load and create thumbnail
        image = load_image(image_path)
        image.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)

        return jsonify({
            "success": True,
            "thumbnail": image_to_base64(image)
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.exception("Error generating thumbnail: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate_caption():
    """Generate caption for uploaded image or image path"""
    try:
        # Get image source (path or uploaded file)
        image_path = request.form.get('image_path', '')
        if image_path:
            image_source = image_path
            filename = Path(image_path).name
        elif 'image' in request.files:
            image_file = request.files['image']
            if not image_file.filename:
                return jsonify({"error": "No image file selected"}), 400
            image_source = image_file.read()
            filename = image_file.filename
        else:
            return jsonify({"error": "No image file or path provided"}), 400

        # Parse request parameters
        model_name = request.form.get('model', 'blip')
        prompt = request.form.get('prompt', '') or None
        
        try:
            parameters = json.loads(request.form.get('parameters', '{}'))
        except json.JSONDecodeError:
            parameters = {}

        # Extract precision parameters for model loading
        precision_params = _extract_precision_params(model_name, parameters)
        
        # Load image and model
        image = load_image(image_source)
        model_adapter = get_model(model_name, precision_params)
        
        if not model_adapter or not model_adapter.is_loaded():
            return jsonify({"error": f"Model {model_name} not available"}), 500

        # Generate caption
        caption = model_adapter.generate_caption(image, prompt, parameters)
        
        # Prepare response
        return jsonify({
            "caption": caption,
            "image_preview": image_to_base64(image),
            "model": model_adapter.model_name,
            "parameters_used": parameters,
            "image_id": request.form.get('image_id', ''),
            "image_filename": filename or request.form.get('image_filename', '')
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Error generating caption: %s", e)
        return jsonify({"error": str(e)}), 500

def _extract_precision_params(model_name: str, parameters: dict) -> dict:
    """Extract precision parameters for models that support them"""
    if not parameters or model_name not in PRECISION_DEFAULTS:
        return None
        
    defaults = PRECISION_DEFAULTS[model_name]
    return {
        'precision': parameters.get('precision', defaults['precision']),
        'use_flash_attention': parameters.get('use_flash_attention', defaults['use_flash_attention'])
    }

def _embed_caption_in_image(image: Image.Image, caption: str, filename: str) -> bytes:
    """Embed caption in image metadata and return bytes"""
    output = io.BytesIO()
    file_ext = Path(filename).suffix.lower()
    
    # Determine format and embed metadata
    if image.format == 'JPEG' or file_ext in ('.jpg', '.jpeg'):
        exif = image.getexif()
        exif[0x010E] = caption  # ImageDescription tag
        image.save(output, format='JPEG', exif=exif, quality=95)
    elif image.format == 'PNG' or file_ext == '.png':
        metadata = PngInfo()
        metadata.add_text("Description", caption)
        image.save(output, format='PNG', pnginfo=metadata)
    else:
        # Default to JPEG with EXIF for other formats
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        exif = Image.Exif()
        exif[0x010E] = caption
        image.save(output, format='JPEG', exif=exif, quality=95)
    
    return output.getvalue()

@app.route('/export/metadata', methods=['POST'])
def export_with_metadata():
    """Export images with embedded EXIF metadata"""
    try:
        data = request.get_json() if request.is_json else {}

        # Get images and captions from different sources
        if 'image_paths' in data and 'captions' in data:
            # Method 1: File paths
            image_sources = [(Path(p), None) for p in data['image_paths']]
            captions = data['captions']
        elif 'images' in request.files and 'captions' in request.form:
            # Method 2: Uploaded files
            image_sources = [(None, f) for f in request.files.getlist('images')]
            captions = request.form.getlist('captions')
        else:
            return jsonify({"error": "Missing images or captions"}), 400

        if len(image_sources) != len(captions):
            return jsonify({"error": "Images and captions count mismatch"}), 400

        # Create ZIP with embedded metadata
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for (img_path, img_file), caption in zip(image_sources, captions):
                # Load image from path or file
                if img_path:
                    if not img_path.exists():
                        continue
                    image = load_image(img_path)
                    filename = img_path.name
                else:
                    image = load_image(img_file.read())
                    filename = img_file.filename

                # Embed caption and add to ZIP
                image_bytes = _embed_caption_in_image(image, caption, filename)
                zip_file.writestr(filename, image_bytes)

        # Return ZIP file
        zip_buffer.seek(0)
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name='images_with_metadata.zip'
        )

    except Exception as e:
        logger.exception("Error in export_with_metadata: %s", e)
        return jsonify({"error": str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large errors"""
    return jsonify({"error": "File too large"}), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.exception("Internal server error: %s", e)
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Configure app
    app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

    # Initialize models
    init_models()

    # Run app
    flask_debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=flask_debug, host='0.0.0.0', port=5000)