"""
Model registry and metadata management.
"""
from .registry import (
    CATEGORIES,
    MODEL_METADATA,
    validate_model_name,
    get_available_models,
    get_model_metadata,
    get_models_by_category,
    get_factory,
)

__all__ = [
    'CATEGORIES',
    'MODEL_METADATA',
    'validate_model_name',
    'get_available_models',
    'get_model_metadata',
    'get_models_by_category',
    'get_factory',
]
