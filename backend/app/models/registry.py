"""
Model Registry Module
Centralized model metadata, categories, and validation.
Now uses adapter factory for automatic adapter creation.
"""
from models.adapter_factory import ModelAdapterFactory
import logging

logger = logging.getLogger(__name__)

# Initialize the global factory instance
_factory = ModelAdapterFactory()


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


def _build_model_metadata():
    """
    Build MODEL_METADATA from factory configs.
    This provides backward compatibility with existing code.
    """
    metadata = {}
    
    for model_key in _factory.get_available_models():
        config = _factory.get_model_config(model_key)
        if config:
            metadata[model_key] = {
                'category': config.get('category', 'general'),
                'description': config.get('description', ''),
                'adapter': lambda key=model_key: _factory.create_adapter(key),
                'adapter_args': {},  # Keep for compatibility but not used
                'vlm_capable': config.get('vlm_capable', False)
            }
    
    return metadata


# Build MODEL_METADATA dynamically from factory
MODEL_METADATA = _build_model_metadata()


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


def get_factory() -> ModelAdapterFactory:
    """
    Get the global factory instance.
    
    Returns:
        ModelAdapterFactory instance
    """
    return _factory


def reload_model_configs():
    """
    Reload model configurations from file.
    Useful for hot-reloading new models without restart.
    """
    global MODEL_METADATA, _factory
    
    _factory.reload_configs()
    MODEL_METADATA = _build_model_metadata()
    
    logger.info("Reloaded model configurations: %d models available", len(MODEL_METADATA))



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
