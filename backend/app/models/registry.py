"""
Model Registry Module
Centralized model metadata, categories, and validation.
"""
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


# Category definitions for organizing models in the UI
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
    'classification': {
        'name': 'Classification',
        'icon': '🏷️',
        'color': '#f59e0b',
        'description': 'Image classification and object recognition'
    },
    'ocr': {
        'name': 'OCR & Text',
        'icon': '📝',
        'color': '#10b981',
        'description': 'Text extraction and analysis'
    }
}

# Comprehensive model metadata registry
MODEL_METADATA = {
    'blip': {
        'category': 'general',
        'description': "Fast, basic image captioning",
        'adapter': BlipAdapter,
        'adapter_args': {},
        'vlm_capable': True
    },
    'blip2': {
        'category': 'general',
        'description': "BLIP2-OPT-2.7B - Enhanced captioning",
        'adapter': Blip2Adapter,
        'adapter_args': {'model_id': "Salesforce/blip2-opt-2.7b"},
        'vlm_capable': True
    },
    'r4b': {
        'category': 'general',
        'description': "Advanced reasoning model with configurable parameters",
        'adapter': R4BAdapter,
        'adapter_args': {},
        'vlm_capable': True
    },
    'wdvit': {
        'category': 'anime',
        'description': "WD-ViT Large Tagger v3 - Anime-style image tagging model with ViT backbone",
        'adapter': WdVitAdapter,
        'adapter_args': {'model_id': "SmilingWolf/wd-vit-large-tagger-v3"},
        'vlm_capable': False
    },
    'wdeva02': {
        'category': 'anime',
        'description': "WD-EVA02 Large Tagger v3 - Anime-style image tagging model with EVA02 backbone (improved accuracy)",
        'adapter': WdVitAdapter,
        'adapter_args': {'model_id': "SmilingWolf/wd-eva02-large-tagger-v3"},
        'vlm_capable': False
    },
    'wd14-convnext': {
        'category': 'anime',
        'description': "WD v1.4 ConvNext Tagger v2 - Fast ONNX-based anime tagging (optimized inference)",
        'adapter': Wd14ConvNextAdapter,
        'adapter_args': {'model_id': "SmilingWolf/wd-v1-4-convnext-tagger-v2"},
        'vlm_capable': False
    },
    'janus-1.3b': {
        'category': 'multimodal',
        'description': "Janus 1.3B - Multimodal vision-language model with efficient architecture",
        'adapter': JanusAdapter,
        'adapter_args': {'model_id': "deepseek-ai/Janus-1.3B"},
        'vlm_capable': True
    },
    'janus-pro-1b': {
        'category': 'multimodal',
        'description': "Janus Pro 1B - Compact professional-grade vision model",
        'adapter': JanusAdapter,
        'adapter_args': {'model_id': "deepseek-ai/Janus-Pro-1B"},
        'vlm_capable': True
    },
    'janus-pro-7b': {
        'category': 'multimodal',
        'description': "Janus Pro 7B - Advanced multimodal model with superior reasoning capabilities",
        'adapter': JanusAdapter,
        'adapter_args': {'model_id': "deepseek-ai/Janus-Pro-7B"},
        'vlm_capable': True
    },
    'lfm2-vl-3b': {
        'category': 'multimodal',
        'description': "LFM2-VL-3B - LiquidAI's vision-language model with chat capabilities",
        'adapter': LFM2Adapter,
        'adapter_args': {'model_id': "LiquidAI/LFM2-VL-3B"},
        'vlm_capable': True
    },
    'nanonets-ocr-s': {
        'category': 'ocr',
        'description': "Nanonets OCR-S - Lightweight OCR (tables/equations/HTML) via Transformers",
        'adapter': NanonetsOCRAdapter,
        'adapter_args': {'model_id': "nanonets/Nanonets-OCR-s"},
        'vlm_capable': False
    },
    'chandra-ocr': {
        'category': 'ocr',
        'description': "Chandra OCR - Advanced layout-aware text extraction with table/equation support",
        'adapter': ChandraAdapter,
        'adapter_args': {'model_id': "datalab-to/chandra"},
        'vlm_capable': False
    },
    'trocr-large-printed': {
        'category': 'ocr',
        'description': "TrOCR Large Printed - Microsoft's transformer-based OCR for printed text",
        'adapter': TrOCRAdapter,
        'adapter_args': {'model_id': "microsoft/trocr-large-printed"},
        'vlm_capable': False
    },
    'llava-phi3': {
        'category': 'multimodal',
        'description': "LLaVA-Phi-3-Mini - Compact and efficient vision-language model",
        'adapter': LlavaPhiAdapter,
        'adapter_args': {'model_id': "xtuner/llava-phi-3-mini-hf"},
        'vlm_capable': True
    },
    'vit-classifier': {
        'category': 'classification',
        'description': "Google ViT Base - ImageNet classification model (1000 object classes)",
        'adapter': VitClassifierAdapter,
        'adapter_args': {'model_id': "google/vit-base-patch16-224"},
        'vlm_capable': False
    }
}


def validate_model_name(model_name: str) -> bool:
    """
    Validate that a model name exists in the registry.

    Args:
        model_name: The model name to validate

    Returns:
        True if the model exists in MODEL_METADATA, False otherwise
    """
    if not model_name:
        return False
    return model_name in MODEL_METADATA


def get_available_models():
    """
    Get list of all available model names.

    Returns:
        List of model names from the registry
    """
    return list(MODEL_METADATA.keys())


def get_model_metadata(model_name: str):
    """
    Get metadata for a specific model.

    Args:
        model_name: The model name to retrieve metadata for

    Returns:
        Model metadata dict if model exists, None otherwise
    """
    return MODEL_METADATA.get(model_name)


def get_models_by_category(category: str):
    """
    Get all models belonging to a specific category.

    Args:
        category: The category name (e.g., 'general', 'anime', 'multimodal')

    Returns:
        Dict of model names to metadata for models in the category
    """
    if category not in CATEGORIES:
        return {}

    return {
        name: meta
        for name, meta in MODEL_METADATA.items()
        if meta['category'] == category
    }
