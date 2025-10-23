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
from utils.image_utils import process_uploaded_image, validate_image_format
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

    # IMPORTANT: Unload other models to free VRAM when switching models
    for other_model_name in models.keys():
        if other_model_name != model_name and models[other_model_name] is not None:
            logger.info("Switching model %s -> %s (unloading %s)", other_model_name, model_name, other_model_name)
            models[other_model_name].unload()
            models[other_model_name] = None

    # Check if we need to reload the model due to precision changes
    should_reload = force_reload
    if model_name in ['r4b', 'qwen3vl-4b', 'qwen3vl-8b'] and models[model_name] is not None and precision_params:
        # Check if current model settings differ from requested settings
        current_adapter = models[model_name]
        if hasattr(current_adapter, 'current_precision_params'):
            if current_adapter.current_precision_params != precision_params:
                logger.info("Model settings changed, reloading %s…", model_name)
                should_reload = True
        else:
            # First time with parameters, need to track them
            should_reload = True

    if models[model_name] is None or should_reload:
        if model_name == 'blip':
            try:
                action = "reloading" if should_reload else "loading"
                logger.info("%s BLIP model on-demand…", action.capitalize())

                # Clear existing model if reloading
                if should_reload and models['blip'] is not None:
                    models['blip'].unload()  # Properly unload the model
                    models['blip'] = None

                models['blip'] = BlipAdapter()
                models['blip'].load_model()
            except Exception as e:
                logger.exception("Failed to load BLIP model: %s", e)
                raise

        elif model_name == 'r4b':
            try:
                action = "reloading" if should_reload else "loading"
                logger.info("%s R-4B model on-demand…", action.capitalize())

                # Clear existing model if reloading
                if should_reload and models['r4b'] is not None:
                    models['r4b'].unload()  # Properly unload the model
                    models['r4b'] = None

                models['r4b'] = R4BAdapter()

                # Extract precision parameters if provided
                if precision_params:
                    models['r4b'].load_model(
                        precision=precision_params.get('precision', 'float32'),
                        use_flash_attention=precision_params.get('use_flash_attention', False)
                    )
                    # Store current parameters for future comparison
                    models['r4b'].current_precision_params = precision_params.copy()
                else:
                    models['r4b'].load_model()
                    models['r4b'].current_precision_params = {
                        'precision': 'float32',
                        'use_flash_attention': False
                    }
            except Exception as e:
                logger.exception("Failed to load R-4B model: %s", e)
                raise

        elif model_name == 'qwen3vl-4b':
            try:
                action = "reloading" if should_reload else "loading"
                logger.info("%s Qwen3-VL-4B model on-demand…", action.capitalize())

                # Clear existing model if reloading
                if should_reload and models['qwen3vl-4b'] is not None:
                    models['qwen3vl-4b'].unload()  # Properly unload the model
                    models['qwen3vl-4b'] = None

                models['qwen3vl-4b'] = Qwen3VLAdapter(model_id="Qwen/Qwen3-VL-4B-Instruct")

                # Extract precision parameters if provided
                if precision_params:
                    models['qwen3vl-4b'].load_model(
                        precision=precision_params.get('precision', 'auto'),
                        use_flash_attention=precision_params.get('use_flash_attention', False)
                    )
                    # Store current parameters for future comparison
                    models['qwen3vl-4b'].current_precision_params = precision_params.copy()
                else:
                    models['qwen3vl-4b'].load_model()
                    models['qwen3vl-4b'].current_precision_params = {
                        'precision': 'auto',
                        'use_flash_attention': False
                    }
            except Exception as e:
                logger.exception("Failed to load Qwen3-VL-4B model: %s", e)
                raise

        elif model_name == 'qwen3vl-8b':
            try:
                action = "reloading" if should_reload else "loading"
                logger.info("%s Qwen3-VL-8B model on-demand…", action.capitalize())

                # Clear existing model if reloading
                if should_reload and models['qwen3vl-8b'] is not None:
                    models['qwen3vl-8b'].unload()  # Properly unload the model
                    models['qwen3vl-8b'] = None

                models['qwen3vl-8b'] = Qwen3VLAdapter(model_id="Qwen/Qwen3-VL-8B-Instruct")

                # Extract precision parameters if provided
                if precision_params:
                    models['qwen3vl-8b'].load_model(
                        precision=precision_params.get('precision', 'auto'),
                        use_flash_attention=precision_params.get('use_flash_attention', False)
                    )
                    # Store current parameters for future comparison
                    models['qwen3vl-8b'].current_precision_params = precision_params.copy()
                else:
                    models['qwen3vl-8b'].load_model()
                    models['qwen3vl-8b'].current_precision_params = {
                        'precision': 'auto',
                        'use_flash_attention': False
                    }
            except Exception as e:
                logger.exception("Failed to load Qwen3-VL-8B model: %s", e)
                raise

        elif model_name == 'wdvit':
            try:
                action = "reloading" if should_reload else "loading"
                logger.info("%s WD-ViT tagger model on-demand…", action.capitalize())

                # Clear existing model if reloading
                if should_reload and models['wdvit'] is not None:
                    models['wdvit'].unload()  # Properly unload the model
                    models['wdvit'] = None

                models['wdvit'] = WdVitAdapter(model_id="SmilingWolf/wd-vit-large-tagger-v3")
                models['wdvit'].load_model()
            except Exception as e:
                logger.exception("Failed to load WD-ViT model: %s", e)
                raise

        elif model_name == 'wdeva02':
            try:
                action = "reloading" if should_reload else "loading"
                logger.info("%s WD-EVA02 tagger model on-demand…", action.capitalize())

                # Clear existing model if reloading
                if should_reload and models['wdeva02'] is not None:
                    models['wdeva02'].unload()  # Properly unload the model
                    models['wdeva02'] = None

                models['wdeva02'] = WdVitAdapter(model_id="SmilingWolf/wd-eva02-large-tagger-v3")
                models['wdeva02'].load_model()
            except Exception as e:
                logger.exception("Failed to load WD-EVA02 model: %s", e)
                raise

    return models[model_name]

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

        img_file = Path(image_path)

        # Security: Validate path
        if not img_file.exists() or not img_file.is_file():
            return jsonify({"error": "Image not found"}), 404

        # Read and create thumbnail
        image = Image.open(img_file)

        # Create thumbnail (150x150 max, maintain aspect ratio)
        image.thumbnail((150, 150), Image.Resampling.LANCZOS)

        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            "success": True,
            "thumbnail": f"data:image/jpeg;base64,{img_str}"
        })

    except Exception as e:
        logger.exception("Error generating thumbnail: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate_caption():
    """Generate caption for uploaded image or image path"""
    try:
        debug_log("=== NEW CAPTION GENERATION REQUEST ===")

        # Check if image_path is provided (new method)
        image_path = request.form.get('image_path', '')

        if image_path:
            # Read image from filesystem
            img_file = Path(image_path)
            if not img_file.exists() or not img_file.is_file():
                return jsonify({"error": "Image not found"}), 404

            with open(img_file, 'rb') as f:
                image_data = f.read()

            filename = img_file.name
            debug_log("Image loaded from path", {
                "filename": filename,
                "path": image_path,
                "size_bytes": len(image_data)
            })
        else:
            # Old method: uploaded file
            if 'image' not in request.files:
                return jsonify({"error": "No image file or path provided"}), 400

            image_file = request.files['image']
            if image_file.filename == '':
                return jsonify({"error": "No image file selected"}), 400

            debug_log("Image file received", {
                "filename": image_file.filename,
                "size_bytes": len(image_file.read())
            })
            image_file.seek(0)
            image_data = image_file.read()
            filename = image_file.filename

        # Get model and parameters
        model_name = request.form.get('model', 'blip')
        parameters_str = request.form.get('parameters', '{}')
        prompt = request.form.get('prompt', '')
        # Optional client-provided identifiers for tracing
        req_image_id = request.form.get('image_id', '')
        req_image_filename = request.form.get('image_filename', '')

        debug_log("Raw request data", {
            "model_name": model_name,
            "parameters_str": parameters_str,
            "prompt": prompt,
            "prompt_length": len(prompt) if prompt else 0,
            "req_image_id": req_image_id,
            "req_image_filename": req_image_filename
        })

        try:
            parameters = json.loads(parameters_str)
            debug_log("Parsed parameters successfully", parameters)
        except json.JSONDecodeError as e:
            debug_log("Failed to parse parameters", {"error": str(e), "raw": parameters_str})
            parameters = {}

        # Extract precision parameters for models that support it
        precision_params = None
        if model_name == 'r4b' and parameters:
            precision_params = {
                'precision': parameters.get('precision', 'float32'),
                'use_flash_attention': parameters.get('use_flash_attention', False)
            }
            debug_log("Extracted precision parameters for R-4B", precision_params)
        elif model_name in ['qwen3vl-4b', 'qwen3vl-8b'] and parameters:
            precision_params = {
                'precision': parameters.get('precision', 'auto'),
                'use_flash_attention': parameters.get('use_flash_attention', False)
            }
            debug_log(f"Extracted precision parameters for {model_name}", precision_params)

        # Get model instance
        debug_log(f"Loading/getting model: {model_name}")
        try:
            model_adapter = get_model(model_name, precision_params)
            debug_log(f"Model {model_name} loaded successfully", {
                "is_loaded": model_adapter.is_loaded() if model_adapter else False
            })
        except Exception as e:
            debug_log(f"Failed to load model {model_name}", {"error": str(e)})
            return jsonify({"error": f"Failed to load model {model_name}: {str(e)}"}), 500

        if model_adapter is None or not model_adapter.is_loaded():
            debug_log(f"Model {model_name} not available or not loaded")
            return jsonify({"error": f"Model {model_name} not available"}), 500

        # Process image
        image = process_uploaded_image(image_data)

        debug_log("Image processed", {
            "mode": image.mode,
            "size": image.size,
            "format": image.format
        })

        # Validate image format
        if not validate_image_format(image):
            debug_log("Image format validation failed")
            return jsonify({"error": "Unsupported image format"}), 400

        # Generate caption with model-specific parameters
        debug_log("Starting caption generation", {
            "model_name": model_name,
            "prompt": prompt,
            "prompt_is_empty": not bool(prompt),
            "parameters_count": len(parameters) if parameters else 0
        })

        if hasattr(model_adapter, 'generate_caption'):
            if model_name == 'r4b':
                # R-4B supports parameters
                debug_log("Calling R-4B generate_caption", {
                    "parameters_passed": parameters,
                    "specific_params_of_interest": {
                        "thinking_mode": parameters.get('thinking_mode'),
                        "temperature": parameters.get('temperature'),
                        "max_new_tokens": parameters.get('max_new_tokens')
                    }
                })
                caption = model_adapter.generate_caption(
                    image,
                    prompt if prompt else None,
                    parameters
                )
            else:
                # BLIP also supports parameters
                debug_log("Calling BLIP generate_caption", {
                    "prompt": prompt if prompt else "(no prompt)",
                    "parameters": parameters
                })
                caption = model_adapter.generate_caption(
                    image,
                    prompt if prompt else None,
                    parameters
                )

            debug_log("Caption generated successfully", {
                "caption_length": len(caption) if caption else 0,
                "caption_preview": caption[:100] + "..." if caption and len(caption) > 100 else caption
            })
        else:
            debug_log("Model does not support caption generation")
            return jsonify({"error": "Model does not support caption generation"}), 500

        # Convert image to base64 for frontend display
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            "caption": caption,
            "image_preview": f"data:image/jpeg;base64,{img_str}",
            "model": model_adapter.model_name,
            "parameters_used": parameters,
            # Echo identifiers back to the client for robust pairing
            "image_id": req_image_id,
            # Prefer the server-derived filename when available
            "image_filename": filename or req_image_filename
        })

    except Exception as e:
        logger.exception("Error in generate_caption: %s", e)
        return jsonify({"error": str(e)}), 500

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