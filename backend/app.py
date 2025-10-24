import io
import json
import zipfile
import os
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
from PIL.PngImagePlugin import PngInfo

if os.environ.get("TAGGER_FORCE_CPU", "0") == "1":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from models.blip_adapter import BlipAdapter
from models.r4b_adapter import R4BAdapter
from models.qwen3vl_adapter import Qwen3VLAdapter
from models.wdvit_adapter import WdVitAdapter
from utils.image_utils import load_image, image_to_base64
from utils.logging_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
THUMBNAIL_SIZE = (150, 150)
MAX_FILE_SIZE = 16 * 1024 * 1024

PRECISION_DEFAULTS = {
    'r4b': {'precision': 'float32', 'use_flash_attention': False},
    'qwen3vl-4b': {'precision': 'auto', 'use_flash_attention': False},
    'qwen3vl-8b': {'precision': 'auto', 'use_flash_attention': False}
}

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

models = {name: None for name in MODEL_METADATA.keys()}
_model_info_cache = {}

def validate_model_name(model_name: str) -> bool:
    return model_name in MODEL_METADATA

def init_models():
    logger.info("Model registry ready. Models load on first use.")
    logger.info("Available models: %s", ", ".join(MODEL_METADATA.keys()))

def get_model(model_name, precision_params=None, force_reload=False):
    if not validate_model_name(model_name):
        raise ValueError(f"Unknown model: {model_name}")

    for other_name, other_model in models.items():
        if other_name != model_name and other_model is not None:
            logger.info("Switching model: unloading %s", other_name)
            other_model.unload()
            models[other_name] = None

    should_reload = force_reload or _needs_precision_reload(model_name, precision_params)

    if models[model_name] is None or should_reload:
        _load_model(model_name, precision_params, should_reload)

    return models[model_name]

def _needs_precision_reload(model_name: str, precision_params: dict) -> bool:
    if not precision_params or model_name not in PRECISION_DEFAULTS:
        return False
    current_model = models[model_name]
    return current_model and hasattr(current_model, 'current_precision_params') and \
           current_model.current_precision_params != precision_params

def _load_model(model_name: str, precision_params: dict, is_reload: bool):
    if is_reload and models[model_name]:
        models[model_name].unload()
        models[model_name] = None

    try:
        metadata = MODEL_METADATA[model_name]
        action = "Reloading" if is_reload else "Loading"
        logger.info("%s %s model on-demand…", action, model_name)

        adapter = metadata['adapter'](**metadata['adapter_args'])

        if precision_params:
            adapter.load_model(**precision_params)
            adapter.current_precision_params = precision_params.copy()
        else:
            adapter.load_model()
            if model_name in PRECISION_DEFAULTS:
                adapter.current_precision_params = PRECISION_DEFAULTS[model_name]

        models[model_name] = adapter

    except Exception as e:
        logger.exception("Failed to load %s model: %s", model_name, e)
        models[model_name] = None
        raise

@app.route('/health', methods=['GET'])
def health_check():
    model_status = {f"{name}_loaded": model is not None and model.is_loaded()
                    for name, model in models.items()}
    return jsonify({"status": "ok", "models_available": list(models.keys()), **model_status})

@app.route('/model/info', methods=['GET'])
def model_info():
    model_name = request.args.get('model', 'blip')
    if not validate_model_name(model_name):
        return jsonify({"error": f"Unknown model: {model_name}"}), 400

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
    available_models = [
        {"name": name, "loaded": models[name] is not None and models[name].is_loaded(),
         "description": MODEL_METADATA[name]['description']}
        for name in MODEL_METADATA.keys()
    ]
    return jsonify({"models": available_models})

@app.route('/model/reload', methods=['POST'])
def reload_model():
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
    try:
        data = request.get_json() or {}
        model_name = data.get('model', 'r4b')
        if not validate_model_name(model_name):
            return jsonify({"error": f"Unknown model: {model_name}"}), 400

        if models[model_name]:
            models[model_name].unload()
            models[model_name] = None

        return jsonify({"success": True, "message": f"Model {model_name} unloaded successfully"})
    except Exception as e:
        logger.exception("Error unloading model: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/scan-folder', methods=['POST'])
def scan_folder():
    try:
        folder_path = request.get_json().get('folder_path', '')
        if not folder_path:
            return jsonify({"error": "No folder path provided"}), 400

        folder = Path(folder_path).resolve()
        if not folder.exists():
            return jsonify({"error": "Folder does not exist"}), 404
        if not folder.is_dir():
            return jsonify({"error": "Path is not a directory"}), 400

        images = sorted([
            {'filename': f.name, 'path': str(f), 'size': f.stat().st_size}
            for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_FORMATS
        ], key=lambda x: x['filename'])

        return jsonify({"success": True, "folder": str(folder), "images": images, "count": len(images)})
    except Exception as e:
        logger.exception("Error scanning folder: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/image/thumbnail', methods=['GET'])
def get_thumbnail():
    try:
        image_path = request.args.get('path', '')
        if not image_path:
            return jsonify({"error": "No image path provided"}), 400

        image = load_image(image_path)
        image.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
        return jsonify({"success": True, "thumbnail": image_to_base64(image)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.exception("Error generating thumbnail: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate_caption():
    try:
        image_path = request.form.get('image_path', '')
        if image_path:
            image_source, filename = image_path, Path(image_path).name
        elif 'image' in request.files:
            image_file = request.files['image']
            if not image_file.filename:
                return jsonify({"error": "No image file selected"}), 400
            image_source, filename = image_file.read(), image_file.filename
        else:
            return jsonify({"error": "No image file or path provided"}), 400

        model_name = request.form.get('model', 'blip')
        prompt = request.form.get('prompt', '') or None

        try:
            parameters = json.loads(request.form.get('parameters', '{}'))
        except json.JSONDecodeError:
            parameters = {}

        precision_params = _extract_precision_params(model_name, parameters)
        image = load_image(image_source)
        model_adapter = get_model(model_name, precision_params)

        if not model_adapter or not model_adapter.is_loaded():
            return jsonify({"error": f"Model {model_name} not available"}), 500

        caption = model_adapter.generate_caption(image, prompt, parameters)

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
    if not parameters or model_name not in PRECISION_DEFAULTS:
        return None
    defaults = PRECISION_DEFAULTS[model_name]
    return {
        'precision': parameters.get('precision', defaults['precision']),
        'use_flash_attention': parameters.get('use_flash_attention', defaults['use_flash_attention'])
    }

def _embed_caption_in_image(image: Image.Image, caption: str, filename: str) -> bytes:
    output = io.BytesIO()
    file_ext = Path(filename).suffix.lower()

    if image.format == 'JPEG' or file_ext in ('.jpg', '.jpeg'):
        exif = image.getexif()
        exif[0x010E] = caption
        image.save(output, format='JPEG', exif=exif, quality=95)
    elif image.format == 'PNG' or file_ext == '.png':
        metadata = PngInfo()
        metadata.add_text("Description", caption)
        image.save(output, format='PNG', pnginfo=metadata)
    else:
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        exif = Image.Exif()
        exif[0x010E] = caption
        image.save(output, format='JPEG', exif=exif, quality=95)

    return output.getvalue()

@app.route('/export/metadata', methods=['POST'])
def export_with_metadata():
    try:
        data = request.get_json() if request.is_json else {}

        if 'image_paths' in data and 'captions' in data:
            image_sources = [(Path(p), None) for p in data['image_paths']]
            captions = data['captions']
        elif 'images' in request.files and 'captions' in request.form:
            image_sources = [(None, f) for f in request.files.getlist('images')]
            captions = request.form.getlist('captions')
        else:
            return jsonify({"error": "Missing images or captions"}), 400

        if len(image_sources) != len(captions):
            return jsonify({"error": "Images and captions count mismatch"}), 400

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for (img_path, img_file), caption in zip(image_sources, captions):
                if img_path:
                    if not img_path.exists():
                        continue
                    image, filename = load_image(img_path), img_path.name
                else:
                    image, filename = load_image(img_file.read()), img_file.filename

                image_bytes = _embed_caption_in_image(image, caption, filename)
                zip_file.writestr(filename, image_bytes)

        zip_buffer.seek(0)
        return send_file(zip_buffer, mimetype='application/zip', as_attachment=True,
                        download_name='images_with_metadata.zip')
    except Exception as e:
        logger.exception("Error in export_with_metadata: %s", e)
        return jsonify({"error": str(e)}), 500

@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "File too large"}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.exception("Internal server error: %s", e)
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
    init_models()
    flask_debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=flask_debug, host='0.0.0.0', port=5000)