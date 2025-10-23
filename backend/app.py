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

# Debug mode via env (default: off)
DEBUG_MODE = os.environ.get("TAGGER_DEBUG", "0") == "1"

def debug_log(message, data=None):
    """Debug helper that logs a single compact line when TAGGER_DEBUG=1."""
    if DEBUG_MODE:
        if data is not None:
            logger.debug("%s | data=%s", message, compact_json(data))
        else:
            logger.debug("%s", message)

app = Flask(__name__)
CORS(app)

# Configuration file path
CONFIG_FILE = Path(__file__).parent.parent / "user_config.json"

# Global model instances
models = {
    'blip': None,
    'r4b': None,
    'qwen3vl-4b': None,
    'qwen3vl-8b': None,
    'wdvit': None,
    'wdeva02': None
}
current_model = 'blip'

def init_models():
    """Initialize model registry (models will be loaded on-demand)"""
    global models

    # Note: All models are now loaded on-demand to save memory and startup time
    logger.info("Model registry ready. Models load on first use.")
    logger.info("Available models: BLIP (fast), R-4B (reasoning), Qwen3-VL-4B/8B (vision-language), WD-ViT/EVA02 (anime tagging)")

def get_model(model_name, precision_params=None, force_reload=False):
    """Get or load a specific model with optional precision parameters"""
    global models

    if model_name not in models:
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
    if not precision_params or model_name not in ['r4b', 'qwen3vl-4b', 'qwen3vl-8b']:
        return False
        
    current_model = models[model_name]
    if current_model is None:
        return False
        
    if not hasattr(current_model, 'current_precision_params'):
        return True
        
    return current_model.current_precision_params != precision_params

def _load_model(model_name: str, precision_params: dict, is_reload: bool):
    """Load or reload a model with given parameters"""
    action = "Reloading" if is_reload else "Loading"
    
    # Unload existing model if reloading
    if is_reload and models[model_name] is not None:
        models[model_name].unload()
        models[model_name] = None

    try:
        # Model configurations
        model_configs = {
            'blip': {
                'adapter': BlipAdapter,
                'params': {},
                'log': "BLIP"
            },
            'r4b': {
                'adapter': R4BAdapter,
                'params': precision_params or {},
                'defaults': {'precision': 'float32', 'use_flash_attention': False},
                'log': "R-4B"
            },
            'qwen3vl-4b': {
                'adapter': lambda: Qwen3VLAdapter(model_id="Qwen/Qwen3-VL-4B-Instruct"),
                'params': precision_params or {},
                'defaults': {'precision': 'auto', 'use_flash_attention': False},
                'log': "Qwen3-VL-4B"
            },
            'qwen3vl-8b': {
                'adapter': lambda: Qwen3VLAdapter(model_id="Qwen/Qwen3-VL-8B-Instruct"),
                'params': precision_params or {},
                'defaults': {'precision': 'auto', 'use_flash_attention': False},
                'log': "Qwen3-VL-8B"
            },
            'wdvit': {
                'adapter': lambda: WdVitAdapter(model_id="SmilingWolf/wd-vit-large-tagger-v3"),
                'params': {},
                'log': "WD-ViT"
            },
            'wdeva02': {
                'adapter': lambda: WdVitAdapter(model_id="SmilingWolf/wd-eva02-large-tagger-v3"),
                'params': {},
                'log': "WD-EVA02"
            }
        }
        
        config = model_configs[model_name]
        logger.info("%s %s model on-demand…", action, config['log'])
        
        # Create adapter
        adapter_factory = config['adapter']
        models[model_name] = adapter_factory() if callable(adapter_factory) else adapter_factory()
        
        # Load model with parameters
        load_params = config['params']
        if load_params:
            models[model_name].load_model(**load_params)
            # Store parameters for future comparison
            models[model_name].current_precision_params = load_params.copy()
        else:
            models[model_name].load_model()
            if 'defaults' in config:
                models[model_name].current_precision_params = config['defaults']
                
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
        else:
            # Return default configuration
            default_config = {
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
            save_user_config(default_config)
            return default_config
    except Exception as e:
        logger.exception("Error loading user config: %s", e)
        return {"savedConfigurations": {}, "customPrompts": [], "lastModified": None}

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
    return jsonify({
        "status": "ok",
        "models_available": list(models.keys()),
        "blip_loaded": models['blip'] is not None and models['blip'].is_loaded(),
        "r4b_loaded": models['r4b'] is not None and models['r4b'].is_loaded(),
        "qwen3vl_4b_loaded": models['qwen3vl-4b'] is not None and models['qwen3vl-4b'].is_loaded(),
        "qwen3vl_8b_loaded": models['qwen3vl-8b'] is not None and models['qwen3vl-8b'].is_loaded(),
        "wdvit_loaded": models['wdvit'] is not None and models['wdvit'].is_loaded(),
        "wdeva02_loaded": models['wdeva02'] is not None and models['wdeva02'].is_loaded()
    })

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information including available parameters"""
    model_name = request.args.get('model', 'blip')

    try:
        # Get model adapter (don't need to load the full model just to get parameters)
        if model_name == 'blip':
            from models.blip_adapter import BlipAdapter
            adapter = BlipAdapter()
        elif model_name == 'r4b':
            from models.r4b_adapter import R4BAdapter
            adapter = R4BAdapter()
        elif model_name == 'qwen3vl-4b':
            from models.qwen3vl_adapter import Qwen3VLAdapter
            adapter = Qwen3VLAdapter(model_id="Qwen/Qwen3-VL-4B-Instruct")
        elif model_name == 'qwen3vl-8b':
            from models.qwen3vl_adapter import Qwen3VLAdapter
            adapter = Qwen3VLAdapter(model_id="Qwen/Qwen3-VL-8B-Instruct")
        elif model_name == 'wdvit':
            from models.wdvit_adapter import WdVitAdapter
            adapter = WdVitAdapter(model_id="SmilingWolf/wd-vit-large-tagger-v3")
        elif model_name == 'wdeva02':
            from models.wdvit_adapter import WdVitAdapter
            adapter = WdVitAdapter(model_id="SmilingWolf/wd-eva02-large-tagger-v3")
        else:
            return jsonify({"error": f"Unknown model: {model_name}"}), 400

        return jsonify({
            "name": model_name,
            "parameters": adapter.get_available_parameters()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    available_models = []

    for model_name, model_instance in models.items():
        available_models.append({
            "name": model_name,
            "loaded": model_instance is not None and model_instance.is_loaded(),
            "description": {
                "blip": "Fast, basic image captioning",
                "r4b": "Advanced reasoning model with configurable parameters",
                "qwen3vl-4b": "Qwen3-VL 4B - Compact vision-language model with strong performance",
                "qwen3vl-8b": "Qwen3-VL 8B - Advanced vision-language model with superior image understanding",
                "wdvit": "WD-ViT Large Tagger v3 - Anime-style image tagging model with ViT backbone",
                "wdeva02": "WD-EVA02 Large Tagger v3 - Anime-style image tagging model with EVA02 backbone (improved accuracy)"
            }.get(model_name, "Unknown model")
        })

    return jsonify({"models": available_models})

@app.route('/model/reload', methods=['POST'])
def reload_model():
    """Force reload a model with new settings"""
    try:
        data = request.get_json() or {}
        model_name = data.get('model', 'r4b')
        precision_params = data.get('precision_params')

        if model_name not in models:
            return jsonify({"error": f"Unknown model: {model_name}"}), 400

        # Force reload the model
        model_adapter = get_model(model_name, precision_params, force_reload=True)

        return jsonify({
            "success": True,
            "message": f"Model {model_name} reloaded successfully",
            "loaded": model_adapter.is_loaded()
        })

    except Exception as e:
        return jsonify({"error": f"Failed to reload model: {str(e)}"}), 500

@app.route('/model/unload', methods=['POST'])
def unload_model():
    """Unload a model to free memory"""
    try:
        data = request.get_json() or {}
        model_name = data.get('model', 'r4b')

        if model_name not in models:
            return jsonify({"error": f"Unknown model: {model_name}"}), 400

        if models[model_name] is not None:
            # Use the model's unload method for proper cleanup
            models[model_name].unload()
            models[model_name] = None
            logger.info("%s model unloaded successfully", model_name)

        return jsonify({
            "success": True,
            "message": f"Model {model_name} unloaded successfully"
        })

    except Exception as e:
        return jsonify({"error": f"Failed to unload model: {str(e)}"}), 500

@app.route('/config', methods=['GET'])
def get_user_config():
    """Get user configuration"""
    try:
        config = load_user_config()
        return jsonify({
            "success": True,
            "config": config,
            "configFile": str(CONFIG_FILE)
        })
    except Exception as e:
        return jsonify({"error": f"Failed to load config: {str(e)}"}), 500

@app.route('/config', methods=['POST'])
def save_user_config_endpoint():
    """Save user configuration"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No configuration data provided"}), 400

        # Validate required fields
        if 'savedConfigurations' not in data or 'customPrompts' not in data:
            return jsonify({"error": "Invalid configuration format"}), 400

        success = save_user_config(data)
        if success:
            return jsonify({
                "success": True,
                "message": "Configuration saved successfully",
                "configFile": str(CONFIG_FILE),
                "lastModified": data.get("lastModified")
            })
        else:
            return jsonify({"error": "Failed to save configuration"}), 500

    except Exception as e:
        return jsonify({"error": f"Failed to save config: {str(e)}"}), 500

@app.route('/config/backup', methods=['POST'])
def backup_user_config():
    """Create a backup of the current config file"""
    try:
        if CONFIG_FILE.exists():
            backup_name = f"user_config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            backup_path = CONFIG_FILE.parent / backup_name

            # Copy current config to backup
            import shutil
            shutil.copy2(CONFIG_FILE, backup_path)

            return jsonify({
                "success": True,
                "message": "Backup created successfully",
                "backupFile": str(backup_path)
            })
        else:
            return jsonify({"error": "No config file exists to backup"}), 404

    except Exception as e:
        return jsonify({"error": f"Failed to create backup: {str(e)}"}), 500

@app.route('/scan-folder', methods=['POST'])
def scan_folder():
    """Scan a folder and return image metadata"""
    try:
        data = request.get_json()
        folder_path = data.get('folder_path', '')

        if not folder_path:
            return jsonify({"error": "No folder path provided"}), 400

        folder = Path(folder_path)

        # Security: Validate path exists and is a directory
        if not folder.exists():
            return jsonify({"error": "Folder does not exist"}), 404

        if not folder.is_dir():
            return jsonify({"error": "Path is not a directory"}), 400

        # Security: Resolve to absolute path to prevent traversal
        folder = folder.resolve()

        # Scan for supported images
        supported_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        images = []

        for file_path in folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_exts:
                images.append({
                    'filename': file_path.name,
                    'path': str(file_path),
                    'size': file_path.stat().st_size
                })

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
        image.thumbnail((150, 150), Image.Resampling.LANCZOS)

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
    if not parameters:
        return None
        
    if model_name == 'r4b':
        return {
            'precision': parameters.get('precision', 'float32'),
            'use_flash_attention': parameters.get('use_flash_attention', False)
        }
    elif model_name in ['qwen3vl-4b', 'qwen3vl-8b']:
        return {
            'precision': parameters.get('precision', 'auto'),
            'use_flash_attention': parameters.get('use_flash_attention', False)
        }
    return None

@app.route('/export/metadata', methods=['POST'])
def export_with_metadata():
    """Export images with embedded EXIF metadata"""
    try:
        data = request.get_json()

        # Support both old (uploaded files) and new (paths) methods
        if 'image_paths' in data and 'captions' in data:
            # New method: use paths
            image_paths = data['image_paths']
            captions = data['captions']

            if len(image_paths) != len(captions):
                return jsonify({"error": "Images and captions count mismatch"}), 400

            # Create ZIP file in memory
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for img_path, caption in zip(image_paths, captions):
                    # Read image from filesystem
                    img_file_path = Path(img_path)
                    if not img_file_path.exists():
                        continue

                    with open(img_file_path, 'rb') as f:
                        img_data = f.read()

                    image = process_uploaded_image(img_data)

                    # Embed caption in EXIF
                    output = io.BytesIO()

                    if image.format == 'JPEG' or img_file_path.suffix.lower() in ('.jpg', '.jpeg'):
                        exif = image.getexif()
                        exif[0x010E] = caption
                        image.save(output, format='JPEG', exif=exif, quality=95)
                    elif image.format == 'PNG' or img_file_path.suffix.lower() == '.png':
                        metadata = PngInfo()
                        metadata.add_text("Description", caption)
                        image.save(output, format='PNG', pnginfo=metadata)
                    else:
                        if image.mode in ('RGBA', 'LA', 'P'):
                            image = image.convert('RGB')
                        exif = Image.Exif()
                        exif[0x010E] = caption
                        image.save(output, format='JPEG', exif=exif, quality=95)

                    # Add to ZIP
                    zip_file.writestr(img_file_path.name, output.getvalue())

        elif 'images' in request.files and 'captions' in request.form:
            # Old method: uploaded files (backward compatibility)
            images = request.files.getlist('images')
            captions = request.form.getlist('captions')

            if len(images) != len(captions):
                return jsonify({"error": "Images and captions count mismatch"}), 400

            # Create ZIP file in memory
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for img_file, caption in zip(images, captions):
                    img_data = img_file.read()
                    image = process_uploaded_image(img_data)

                    output = io.BytesIO()

                    if image.format == 'JPEG' or img_file.filename.lower().endswith(('.jpg', '.jpeg')):
                        exif = image.getexif()
                        exif[0x010E] = caption
                        image.save(output, format='JPEG', exif=exif, quality=95)
                    elif image.format == 'PNG' or img_file.filename.lower().endswith('.png'):
                        metadata = PngInfo()
                        metadata.add_text("Description", caption)
                        image.save(output, format='PNG', pnginfo=metadata)
                    else:
                        if image.mode in ('RGBA', 'LA', 'P'):
                            image = image.convert('RGB')
                        exif = Image.Exif()
                        exif[0x010E] = caption
                        image.save(output, format='JPEG', exif=exif, quality=95)

                    zip_file.writestr(img_file.filename, output.getvalue())
        else:
            return jsonify({"error": "Missing images or captions"}), 400

        # Prepare for download
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
    return jsonify({"error": "File too large"}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Set max file size to 16MB
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

    # Initialize models
    init_models()

    # Run app (Flask debug mode can cause double logging via reloader)
    flask_debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=flask_debug, host='0.0.0.0', port=5000)