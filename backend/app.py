import io
import json
import zipfile
import os
import sys
import logging
import asyncio
import webbrowser
import threading
from pathlib import Path
from flask import Flask, request, jsonify, send_file, Response, stream_with_context
from flask_cors import CORS
from PIL import Image
from PIL.PngImagePlugin import PngInfo

if os.environ.get("TAGGER_FORCE_CPU", "0") == "1":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from models.blip_adapter import BlipAdapter
from models.blip2_adapter import Blip2Adapter
from models.r4b_adapter import R4BAdapter
from models.wdvit_adapter import WdVitAdapter
from models.janus_adapter import JanusAdapter
from models.nanonets_ocr_adapter import NanonetsOCRAdapter
from models.chandra_adapter import ChandraAdapter
from models.trocr_adapter import TrOCRAdapter
from models.llava_phi3_adapter import LlavaPhiAdapter
from models.lfm2_adapter import LFM2Adapter
from models.wd14_convnext_adapter import Wd14ConvNextAdapter
from models.vit_classifier_adapter import VitClassifierAdapter
from utils.image_utils import load_image, image_to_base64
from utils.logging_utils import setup_logging
from database import SessionManager, AsyncSessionManager, ExecutionManager
from graph_executor import GraphExecutor
from config import (
    SUPPORTED_IMAGE_FORMATS,
    THUMBNAIL_SIZE,
    MAX_FILE_SIZE,
    PRECISION_DEFAULTS,
)

setup_logging()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

app = Flask(__name__)
CORS(app)

# Session managers: sync for simple CRUD, async for AI inference paths
session_manager = SessionManager()
async_session_manager = AsyncSessionManager()
execution_manager = ExecutionManager()

# Active graph executors (job_id -> GraphExecutor)
active_executors = {}

# Category definitions
CATEGORIES = {
    'general': {
        'name': 'General Captioning',
        'icon': '⚡',
        'color': '#6366f1',
        'description': 'Fast & versatile image captioning'
    },
    'anime': {
        'name': 'Anime & Art',
        'icon': '🎨',
        'color': '#ec4899',
        'description': 'Specialized for anime and artwork'
    },
    'multimodal': {
        'name': 'Multimodal Vision',
        'icon': '👁️',
        'color': '#8b5cf6',
        'description': 'Advanced vision-language models'
    },
    'ocr': {
        'name': 'OCR & Text',
        'icon': '📝',
        'color': '#10b981',
        'description': 'Text extraction and analysis'
    }
}

MODEL_METADATA = {
    'blip': {
        'category': 'general',
        'description': "Fast, basic image captioning",
        'adapter': BlipAdapter,
        'adapter_args': {}
    },
    'blip2': {
        'category': 'general',
        'description': "BLIP2-OPT-2.7B - Enhanced captioning",
        'adapter': Blip2Adapter,
        'adapter_args': {'model_id': "Salesforce/blip2-opt-2.7b"}
    },
    'r4b': {
        'category': 'general',
        'description': "Advanced reasoning model with configurable parameters",
        'adapter': R4BAdapter,
        'adapter_args': {}
    },
    'wdvit': {
        'category': 'anime',
        'description': "WD-ViT Large Tagger v3 - Anime-style image tagging model with ViT backbone",
        'adapter': WdVitAdapter,
        'adapter_args': {'model_id': "SmilingWolf/wd-vit-large-tagger-v3"}
    },
    'wdeva02': {
        'category': 'anime',
        'description': "WD-EVA02 Large Tagger v3 - Anime-style image tagging model with EVA02 backbone (improved accuracy)",
        'adapter': WdVitAdapter,
        'adapter_args': {'model_id': "SmilingWolf/wd-eva02-large-tagger-v3"}
    },
    'wd14-convnext': {
        'category': 'anime',
        'description': "WD v1.4 ConvNext Tagger v2 - Fast ONNX-based anime tagging (optimized inference)",
        'adapter': Wd14ConvNextAdapter,
        'adapter_args': {'model_id': "SmilingWolf/wd-v1-4-convnext-tagger-v2"}
    },
    'janus-1.3b': {
        'category': 'multimodal',
        'description': "Janus 1.3B - Multimodal vision-language model with efficient architecture",
        'adapter': JanusAdapter,
        'adapter_args': {'model_id': "deepseek-ai/Janus-1.3B"}
    },
    'janus-pro-1b': {
        'category': 'multimodal',
        'description': "Janus Pro 1B - Compact professional-grade vision model",
        'adapter': JanusAdapter,
        'adapter_args': {'model_id': "deepseek-ai/Janus-Pro-1B"}
    },
    'janus-pro-7b': {
        'category': 'multimodal',
        'description': "Janus Pro 7B - Advanced multimodal model with superior reasoning capabilities",
        'adapter': JanusAdapter,
        'adapter_args': {'model_id': "deepseek-ai/Janus-Pro-7B"}
    },
    'lfm2-vl-3b': {
        'category': 'multimodal',
        'description': "LFM2-VL-3B - LiquidAI's vision-language model with chat capabilities",
        'adapter': LFM2Adapter,
        'adapter_args': {'model_id': "LiquidAI/LFM2-VL-3B"}
    },
    'nanonets-ocr-s': {
        'category': 'ocr',
        'description': "Nanonets OCR-S - Lightweight OCR (tables/equations/HTML) via Transformers",
        'adapter': NanonetsOCRAdapter,
        'adapter_args': {'model_id': "nanonets/Nanonets-OCR-s"}
    },
    'chandra-ocr': {
        'category': 'ocr',
        'description': "Chandra OCR - Advanced layout-aware text extraction with table/equation support",
        'adapter': ChandraAdapter,
        'adapter_args': {'model_id': "datalab-to/chandra"}
    },
    'trocr-large-printed': {
        'category': 'ocr',
        'description': "TrOCR Large Printed - Microsoft's transformer-based OCR for printed text",
        'adapter': TrOCRAdapter,
        'adapter_args': {'model_id': "microsoft/trocr-large-printed"}
    },
    'llava-phi3': {
        'category': 'multimodal',
        'description': "LLaVA-Phi-3-Mini - Compact and efficient vision-language model",
        'adapter': LlavaPhiAdapter,
        'adapter_args': {'model_id': "xtuner/llava-phi-3-mini-hf"}
    },
    'vit-classifier': {
        'category': 'general',
        'description': "Google ViT Base - ImageNet classification model (1000 object classes)",
        'adapter': VitClassifierAdapter,
        'adapter_args': {'model_id': "google/vit-base-patch16-224"}
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

@app.route('/models/categories', methods=['GET'])
def get_model_categories():
    """Get models organized by categories with category metadata."""
    categories_data = []

    for category_id, category_info in CATEGORIES.items():
        # Find all models in this category
        category_models = [
            {
                "name": name,
                "description": MODEL_METADATA[name]['description'],
                "loaded": models[name] is not None and models[name].is_loaded()
            }
            for name, metadata in MODEL_METADATA.items()
            if metadata.get('category') == category_id
        ]

        if category_models:  # Only include categories that have models
            categories_data.append({
                "id": category_id,
                "name": category_info['name'],
                "icon": category_info['icon'],
                "color": category_info['color'],
                "description": category_info['description'],
                "models": category_models
            })

    return jsonify({"categories": categories_data})

@app.route('/models/metadata', methods=['GET'])
def models_metadata():
    """Get comprehensive metadata about all models for documentation."""
    model_details = {
        'blip': {
            'display_name': 'BLIP',
            'full_name': 'Salesforce BLIP',
            'description': 'Fast image captioning',
            'speed_score': 80,  # 0-100
            'vram_gb': 1.6,
            'vram_label': '1.6GB',
            'speed_label': 'Fast',
            'features': ['Fast processing', 'Low VRAM usage', 'General-purpose captions'],
            'use_cases': ['Batch processing', 'Quick previews', 'Resource-constrained systems']
        },
        'blip2': {
            'display_name': 'BLIP2',
            'full_name': 'BLIP2-OPT-2.7B',
            'description': 'Enhanced image captioning',
            'speed_score': 80,
            'vram_gb': 8.2,
            'vram_label': '8.2GB',
            'speed_label': 'Fast',
            'features': ['Enhanced accuracy', 'Detailed captions', 'General-purpose'],
            'use_cases': ['High-quality captions', 'Detailed descriptions']
        },
        'r4b': {
            'display_name': 'R-4B',
            'full_name': 'R-4B Advanced Reasoning',
            'description': 'Advanced reasoning model with configurable parameters',
            'speed_score': 30,
            'vram_gb': 21.9,
            'vram_label': '21.9GB',
            'speed_label': 'Slow',
            'features': ['Advanced reasoning', 'Configurable precision', 'Detailed captions'],
            'use_cases': ['High-quality descriptions', 'Complex scenes', 'Fine-grained control']
        },
        'wdvit': {
            'display_name': 'WD-ViT',
            'full_name': 'WD-ViT Large Tagger v3',
            'description': 'Anime-style image tagging model with ViT backbone',
            'speed_score': 80,
            'vram_gb': 2,
            'vram_label': '2GB',
            'speed_label': 'Fast',
            'features': ['Anime/manga specialized', 'Tag-based output', 'Character recognition'],
            'use_cases': ['Anime artwork', 'Tag generation', 'Booru-style tags']
        },
        'wdeva02': {
            'display_name': 'WD-EVA02',
            'full_name': 'WD-EVA02 Large Tagger v3',
            'description': 'Anime-style image tagging model with EVA02 backbone (improved accuracy)',
            'speed_score': 80,
            'quality_score': 60,
            'vram_gb': 2,
            'vram_label': '2GB',
            'speed_label': 'Fast',
            'quality_label': 'Good',
            'features': ['Enhanced accuracy', 'Anime/manga specialized', 'Advanced tagging'],
            'use_cases': ['Professional anime tagging', 'Dataset creation', 'High-accuracy needs']
        },
        'wd14-convnext': {
            'display_name': 'WD14 ConvNext',
            'full_name': 'WD v1.4 ConvNext Tagger v2',
            'description': 'Fast ONNX-based anime tagging',
            'speed_score': 80,
            'vram_gb': 1.7,
            'vram_label': '1.7GB',
            'speed_label': 'Fast',
            'features': ['ONNX optimized', 'Fast inference', 'Anime tagging'],
            'use_cases': ['Batch tagging', 'Fast anime classification']
        },
        'janus-1.3b': {
            'display_name': 'Janus 1.3B',
            'full_name': 'Janus 1.3B',
            'description': 'Multimodal vision-language model',
            'speed_score': 30,
            'vram_gb': 5.4,
            'vram_label': '5.4GB',
            'speed_label': 'Slow',
            'features': ['Multimodal', 'Efficient architecture', 'Vision-language'],
            'use_cases': ['Detailed descriptions', 'Multimodal understanding']
        },
        'janus-pro-1b': {
            'display_name': 'Janus Pro 1B',
            'full_name': 'Janus Pro 1B',
            'description': 'Compact professional-grade vision model',
            'speed_score': 30,
            'vram_gb': 5.4,
            'vram_label': '5.4GB',
            'speed_label': 'Slow',
            'features': ['Professional grade', 'Compact', 'Vision understanding'],
            'use_cases': ['Professional captions', 'Detailed analysis']
        },
        'janus-pro-7b': {
            'display_name': 'Janus Pro 7B',
            'full_name': 'Janus Pro 7B',
            'description': 'Advanced multimodal model with superior reasoning',
            'speed_score': 30,
            'vram_gb': 16.1,
            'vram_label': '16.1GB',
            'speed_label': 'Slow',
            'features': ['Superior reasoning', 'Advanced multimodal', 'High quality'],
            'use_cases': ['Complex scenes', 'Advanced reasoning', 'High-quality output']
        },
        'lfm2-vl-3b': {
            'display_name': 'LFM2-VL-3B',
            'full_name': 'LiquidAI LFM2-VL-3B',
            'description': 'Vision-language model with chat capabilities',
            'speed_score': 30,
            'vram_gb': 7.2,
            'vram_label': '7.2GB',
            'speed_label': 'Slow',
            'features': ['Chat capabilities', 'Vision-language', 'Conversational'],
            'use_cases': ['Interactive descriptions', 'Chat-based captions']
        },
        'llava-phi3': {
            'display_name': 'LLaVA-Phi3',
            'full_name': 'LLaVA-Phi-3-Mini',
            'description': 'Compact efficient vision-language model',
            'speed_score': 30,
            'vram_gb': 8.9,
            'vram_label': '8.9GB',
            'speed_label': 'Slow',
            'features': ['Compact', 'Efficient', 'Vision-language'],
            'use_cases': ['Detailed captions', 'Multimodal tasks']
        },
        'nanonets-ocr-s': {
            'display_name': 'Nanonets OCR-S',
            'full_name': 'Nanonets OCR-S',
            'description': 'Lightweight OCR for tables, equations, and HTML',
            'speed_score': 50,
            'vram_gb': 10.7,
            'vram_label': '10.7GB',
            'speed_label': 'Medium',
            'features': ['Table extraction', 'Equation recognition', 'HTML output'],
            'use_cases': ['Document OCR', 'Table extraction', 'Academic papers']
        },
        'chandra-ocr': {
            'display_name': 'Chandra OCR',
            'full_name': 'Chandra OCR',
            'description': 'Advanced layout-aware text extraction',
            'speed_score': 30,
            'vram_gb': 18.6,
            'vram_label': '18.6GB',
            'speed_label': 'Slow',
            'features': ['Layout-aware', 'Table support', 'Equation recognition'],
            'use_cases': ['Complex documents', 'Academic papers', 'Technical documents']
        },
        'trocr-large-printed': {
            'display_name': 'TrOCR Large Printed',
            'full_name': 'Microsoft TrOCR Large Printed',
            'description': 'Transformer-based OCR for printed text',
            'speed_score': 50,
            'vram_gb': 1.9,
            'vram_label': '1.9GB',
            'speed_label': 'Medium',
            'features': ['Transformer-based', 'Printed text', 'High accuracy'],
            'use_cases': ['Printed documents', 'Book scanning', 'Text extraction']
        },
        'vit-classifier': {
            'display_name': 'ViT Classifier',
            'full_name': 'Google ViT Base',
            'description': 'ImageNet classification model',
            'speed_score': 80,
            'vram_gb': 1.1,
            'vram_label': '1.1GB',
            'speed_label': 'Fast',
            'features': ['1000 object classes', 'Fast classification', 'ImageNet trained'],
            'use_cases': ['Object classification', 'Quick categorization', 'General objects']
        }
    }

    available_models = [name for name in MODEL_METADATA.keys()]
    active_models = {name: details for name, details in model_details.items() if name in available_models}

    return jsonify({
        'model_count': len(available_models),
        'models': active_models,
        'export_formats': 4,
        'vram_range': f"{min(m['vram_gb'] for m in active_models.values())}-{max(m['vram_gb'] for m in active_models.values())}" if active_models else "2-16",
        'tech_stack': [
            {'name': 'Salesforce BLIP', 'description': 'Fast image captioning'},
            {'name': 'R-4B', 'description': 'Advanced reasoning model'},
            {'name': 'WD Taggers', 'description': 'Anime-style tagging'},
            {'name': 'PyTorch', 'description': 'Deep learning framework'},
            {'name': 'Flask', 'description': 'REST API backend'},
            {'name': 'Vanilla JavaScript', 'description': 'No-build frontend'},
            {'name': 'CUDA', 'description': 'GPU acceleration'},
            {'name': 'DuckDB', 'description': 'Embedded analytics database'}
        ]
    })

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

@app.route('/session/register-folder', methods=['POST'])
def register_folder():
    """Register all images from a folder path."""
    try:
        folder_path = request.get_json().get('folder_path', '')
        if not folder_path:
            return jsonify({"error": "No folder path provided"}), 400

        images = session_manager.register_folder(folder_path)
        return jsonify({
            "success": True,
            "images": images,
            "total": len(images)
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Error registering folder: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/session/register-files', methods=['POST'])
def register_files():
    """Pre-register files before upload."""
    try:
        data = request.get_json()
        file_metadata_list = data.get('files', [])
        if not file_metadata_list:
            return jsonify({"error": "No files provided"}), 400

        image_ids = session_manager.register_files(file_metadata_list)
        return jsonify({
            "success": True,
            "image_ids": image_ids,
            "total": len(image_ids)
        })
    except Exception as e:
        logger.exception("Error registering files: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/upload/batch', methods=['POST'])
def upload_batch():
    """Upload a batch of files."""
    try:
        if 'files' not in request.files:
            return jsonify({"error": "No files provided"}), 400

        files = request.files.getlist('files')
        image_ids = request.form.getlist('image_ids')

        if len(files) != len(image_ids):
            return jsonify({"error": "Files and image_ids count mismatch"}), 400

        uploaded = 0
        failed = 0

        for file, image_id in zip(files, image_ids):
            if session_manager.save_uploaded_file(image_id, file, file.filename):
                uploaded += 1
            else:
                failed += 1

        return jsonify({
            "success": True,
            "uploaded": uploaded,
            "failed": failed,
            "image_ids": image_ids
        })
    except Exception as e:
        logger.exception("Error uploading batch: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/images', methods=['GET'])
def list_images():
    """Get paginated list of images."""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))
        search = request.args.get('search', '')

        result = session_manager.list_images(page, per_page, search)
        return jsonify(result)
    except Exception as e:
        logger.exception("Error listing images: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/image/<image_id>', methods=['GET'])
def get_image_by_id(image_id):
    """Get full-resolution image by ID."""
    try:
        image_path = session_manager.get_image_path(image_id)
        if not image_path:
            return jsonify({"error": "Image not found"}), 404

        image = load_image(image_path)
        return jsonify({"success": True, "image": image_to_base64(image)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.exception("Error loading image: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/image/<image_id>/info', methods=['GET'])
def get_image_info(image_id):
    """Get image metadata by ID."""
    try:
        metadata = session_manager.get_image_metadata(image_id)
        if not metadata:
            return jsonify({"error": "Image not found"}), 404
        return jsonify(metadata)
    except Exception as e:
        logger.exception("Error getting image info: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/session/clear', methods=['DELETE'])
def clear_session():
    """Clear all images and temporary files."""
    try:
        count = session_manager.clear_all()
        return jsonify({
            "success": True,
            "message": "Session cleared",
            "deleted_count": count
        })
    except Exception as e:
        logger.exception("Error clearing session: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/session/remove/<image_id>', methods=['DELETE'])
def remove_image(image_id):
    """Remove a single image from the session."""
    try:
        success = session_manager.delete_image(image_id)
        if success:
            return jsonify({
                "success": True,
                "message": "Image removed",
                "image_id": image_id
            })
        else:
            return jsonify({
                "success": False,
                "error": "Image not found"
            }), 404
    except Exception as e:
        logger.exception("Error removing image %s: %s", image_id, e)
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
async def generate_caption():
    """
    Generate caption for a single image (async, non-blocking).

    Database operations run in thread pool to avoid blocking AI inference.
    Caption saving happens in background (fire-and-forget).
    """
    try:
        image_id = request.form.get('image_id', '')
        logger.info("Single generate request - image_id: %s", image_id)

        # Async DB lookup for image path
        if image_id:
            image_path = await async_session_manager.get_image_path(image_id)
            if not image_path:
                logger.error("Image not found for image_id: %s", image_id)
                return jsonify({"error": "Image not found"}), 404
            image_source, filename = image_path, Path(image_path).name
        elif request.form.get('image_path', ''):
            image_path = request.form.get('image_path')
            image_source, filename = image_path, Path(image_path).name
        elif 'image' in request.files:
            image_file = request.files['image']
            if not image_file.filename:
                return jsonify({"error": "No image file selected"}), 400
            image_source, filename = image_file.read(), image_file.filename
        else:
            logger.error("No valid image source provided. Form: %s", request.form)
            return jsonify({"error": "No image_id, image file, or path provided"}), 400

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

        # AI inference (GPU, potentially slow)
        caption = model_adapter.generate_caption(image, prompt, parameters)

        # Save caption in background (non-blocking, fire-and-forget)
        if image_id:
            asyncio.create_task(async_session_manager.save_caption(image_id, caption))

        return jsonify({
            "caption": caption,
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

@app.route('/generate/batch', methods=['POST'])
async def generate_captions_batch():
    """
    Generate captions for multiple images (async, optimized for concurrency).

    Database operations (path lookups and caption saves) run concurrently
    in thread pool while AI inference runs on GPU.
    """
    try:
        import time
        start_time = time.time()

        data = request.get_json()
        image_ids = data.get('image_ids', [])
        model_name = data.get('model', 'blip')
        prompt = data.get('prompt', '') or None
        parameters = data.get('parameters', {})

        if not image_ids:
            return jsonify({"error": "No image_ids provided"}), 400

        # Load all image paths concurrently (async batch operation)
        image_paths_dict = await async_session_manager.get_image_paths_batch(image_ids)

        # Load images and prepare data
        images = []
        filenames = []
        valid_image_ids = []
        for image_id in image_ids:
            image_path = image_paths_dict.get(image_id)
            if image_path:
                images.append(load_image(image_path))
                filenames.append(Path(image_path).name)
                valid_image_ids.append(image_id)

        if not images:
            return jsonify({"error": "No valid images found"}), 404

        # Get model
        precision_params = _extract_precision_params(model_name, parameters)
        model_adapter = get_model(model_name, precision_params)

        if not model_adapter or not model_adapter.is_loaded():
            return jsonify({"error": f"Model {model_name} not available"}), 500

        # Generate captions for batch (GPU inference)
        prompts = [prompt] * len(images) if prompt else None
        captions = model_adapter.generate_captions_batch(images, prompts, parameters)

        elapsed = time.time() - start_time
        logger.info("Batch: %d images with %s → %d captions (%.1fs)",
                   len(images), model_name, len(captions), elapsed)

        # Save all captions concurrently (async batch operation)
        captions_data = [
            {"image_id": img_id, "caption": cap}
            for img_id, cap in zip(valid_image_ids, captions)
        ]
        asyncio.create_task(async_session_manager.save_captions_batch(captions_data))

        # Build results
        results = []
        for image_id, caption, filename in zip(valid_image_ids, captions, filenames):
            results.append({
                "image_id": image_id,
                "caption": caption,
                "image_filename": filename
            })

        return jsonify({
            "results": results,
            "model": model_adapter.model_name,
            "parameters_used": parameters
        })
    except Exception as e:
        logger.exception("Error generating batch captions: %s", e)
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

@app.route('/graph/execute', methods=['POST'])
def execute_graph():
    """
    Submit a graph for execution.

    Graph execution runs independently in background thread,
    surviving page refreshes and browser sessions.
    """
    try:
        data = request.get_json()
        graph = data.get('graph', {})
        image_ids = data.get('image_ids', [])
        clear_previous = data.get('clear_previous', True)

        if not graph or not image_ids:
            return jsonify({"error": "Missing graph or image_ids"}), 400

        # Clear previous captions if requested
        if clear_previous:
            session_manager.clear_all_captions()
            logger.info("Cleared previous captions before execution")

        # Create job
        job_id = execution_manager.create_job(graph, image_ids)

        # Start execution in background thread
        executor = GraphExecutor(execution_manager, async_session_manager)
        active_executors[job_id] = executor

        def run_executor():
            try:
                asyncio.run(executor.execute(job_id, get_model))
            finally:
                # Cleanup
                active_executors.pop(job_id, None)

        thread = threading.Thread(target=run_executor, daemon=True)
        thread.start()

        logger.info(f"Started execution job {job_id} in background")

        return jsonify({
            "success": True,
            "job_id": job_id,
            "message": "Graph execution started"
        })

    except Exception as e:
        logger.exception("Error starting graph execution: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/graph/status/<job_id>', methods=['GET'])
def get_graph_status_sse(job_id):
    """
    Get real-time status updates for a job via Server-Sent Events.

    Client connects once and receives updates as they happen.
    Connection stays open until job completes/fails/cancelled.
    """
    def generate():
        import time
        last_status = None

        while True:
            status = execution_manager.get_status(job_id)

            if not status:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                break

            # Only send if status changed
            if status != last_status:
                yield f"data: {json.dumps(status)}\n\n"
                last_status = status

            # If terminal state, close connection
            if status['status'] in ('completed', 'failed', 'cancelled'):
                break

            time.sleep(0.5)  # Poll every 500ms

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/graph/cancel/<job_id>', methods=['POST'])
def cancel_graph_execution(job_id):
    """Cancel a running graph execution."""
    try:
        # Signal executor to stop
        executor = active_executors.get(job_id)
        if executor:
            executor.cancel()

        # Update database status
        execution_manager.cancel_job(job_id)

        logger.info(f"Cancelled job {job_id}")

        return jsonify({
            "success": True,
            "message": "Job cancellation requested"
        })

    except Exception as e:
        logger.exception("Error cancelling job: %s", e)
        return jsonify({"error": str(e)}), 500


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "File too large"}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.exception("Internal server error: %s", e)
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Determine frontend path - works in both development and PyInstaller bundle
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        base_path = sys._MEIPASS
        frontend_path = os.path.join(base_path, "frontend", "index.html")
    else:
        # Running as normal Python script
        frontend_path = os.path.join(os.path.dirname(__file__), "../frontend/index.html")
        frontend_path = os.path.abspath(frontend_path)
    
    # Open browser only if not running in Docker (where DISPLAY isn't available)
    in_docker = os.path.exists('/.dockerenv') or os.environ.get('RUNNING_IN_DOCKER') == '1'
    
    if not in_docker:
        webbrowser.open(f"file://{frontend_path}")
    else:
        # In Docker, print URL for user to open manually
        logger.info("=" * 60)
        logger.info("AI Image Captioner is running!")
        logger.info("Open in your browser: http://localhost:5000")
        logger.info("=" * 60)
    
    app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE  # None = no limit
    init_models()
    flask_debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=flask_debug, host='0.0.0.0', port=5000)