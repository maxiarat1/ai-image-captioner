"""
Model Adapter Factory
Automatically creates adapters from configuration and handlers.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from models.unified_adapter import UnifiedModelAdapter
from models.handlers import (
    HuggingFaceVLMHandler,
    HuggingFaceTaggerHandler,
    HuggingFaceOCRHandler,
    HuggingFaceClassifierHandler,
    ONNXTaggerHandler,
    JanusHandler,
    R4BHandler,
    TrOCRHandler
)

logger = logging.getLogger(__name__)


class ModelAdapterFactory:
    """
    Factory for creating model adapters from configuration.
    
    This eliminates the need for individual adapter files by automatically
    creating adapters based on model type and configuration.
    """
    
    # Map model types to handler classes
    HANDLER_MAP = {
        'hf_vlm': HuggingFaceVLMHandler,
        'hf_tagger': HuggingFaceTaggerHandler,
        'hf_ocr': HuggingFaceOCRHandler,
        'hf_classifier': HuggingFaceClassifierHandler,
        'onnx_tagger': ONNXTaggerHandler,
        'hf_ocr_trocr': TrOCRHandler,
        'hf_vlm_custom': None,  # Requires custom_handler lookup
    }
    
    # Map custom handler names to classes
    CUSTOM_HANDLER_MAP = {
        'janus': JanusHandler,
        'r4b': R4BHandler,
        'trocr': TrOCRHandler,
    }
    
    def __init__(self, config_path: str = None):
        """
        Initialize factory with model configurations.
        
        Args:
            config_path: Path to models_config.jsonl file
        """
        if config_path is None:
            # Default to backend/models_config.jsonl
            config_path = Path(__file__).parent.parent / "models_config.jsonl"
        
        self.config_path = Path(config_path)
        self.configs: Dict[str, Dict[str, Any]] = {}
        self._load_configs()
    
    def _load_configs(self):
        """Load model configurations from JSONL file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    config = json.loads(line)
                    model_key = config.get('model_key')
                    if model_key:
                        self.configs[model_key] = config
            
            logger.info("Loaded %d model configurations from %s", 
                       len(self.configs), self.config_path)
        except FileNotFoundError:
            logger.error("Model config file not found: %s", self.config_path)
            raise
        except json.JSONDecodeError as e:
            logger.error("Error parsing model config: %s", e)
            raise
    
    def create_adapter(self, model_key: str, **adapter_args) -> UnifiedModelAdapter:
        """
        Create an adapter for the specified model.
        
        Args:
            model_key: Model identifier (e.g., 'blip', 'r4b', 'wdvit')
            **adapter_args: Additional arguments (currently unused, for compatibility)
            
        Returns:
            UnifiedModelAdapter instance
            
        Raises:
            ValueError: If model_key is not found in configs
            RuntimeError: If handler creation fails
        """
        if model_key not in self.configs:
            raise ValueError(f"Unknown model: {model_key}")
        
        config = self.configs[model_key].copy()
        
        # Override model_id if provided in adapter_args
        if 'model_id' in adapter_args:
            config['model_id'] = adapter_args['model_id']
        
        try:
            # Determine handler class
            handler_class = self._get_handler_class(config)
            
            # Create handler instance
            handler = handler_class(config)
            
            # Create and return unified adapter
            adapter = UnifiedModelAdapter(handler)
            
            logger.debug("Created adapter for %s (type: %s)", model_key, config.get('type'))
            return adapter
            
        except Exception as e:
            logger.exception("Error creating adapter for %s: %s", model_key, e)
            raise RuntimeError(f"Failed to create adapter for {model_key}: {e}")
    
    def _get_handler_class(self, config: Dict[str, Any]):
        """
        Determine the appropriate handler class for a model config.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            Handler class
            
        Raises:
            ValueError: If handler type is unknown
        """
        model_type = config.get('type')
        
        # Check for custom handler override
        if 'custom_handler' in config:
            custom_name = config['custom_handler']
            if custom_name in self.CUSTOM_HANDLER_MAP:
                return self.CUSTOM_HANDLER_MAP[custom_name]
            else:
                raise ValueError(f"Unknown custom handler: {custom_name}")
        
        # Check if it's R4B (special case detection)
        if config.get('model_key') == 'r4b':
            return R4BHandler
        
        # Use type-based mapping
        if model_type in self.HANDLER_MAP:
            handler_class = self.HANDLER_MAP[model_type]
            if handler_class is None:
                raise ValueError(f"Model type {model_type} requires custom_handler specification")
            return handler_class
        
        raise ValueError(f"Unknown model type: {model_type}")
    
    def get_available_models(self) -> list:
        """
        Get list of all available model keys.
        
        Returns:
            List of model keys
        """
        return list(self.configs.keys())
    
    def get_model_config(self, model_key: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific model.
        
        Args:
            model_key: Model identifier
            
        Returns:
            Model configuration dict or None if not found
        """
        return self.configs.get(model_key)
    
    def get_models_by_category(self, category: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all models in a specific category.
        
        Args:
            category: Category name (e.g., 'general', 'anime', 'multimodal')
            
        Returns:
            Dictionary mapping model keys to their configs
        """
        return {
            key: config
            for key, config in self.configs.items()
            if config.get('category') == category
        }
    
    def get_categories(self) -> set:
        """
        Get set of all categories.
        
        Returns:
            Set of category names
        """
        return {config.get('category') for config in self.configs.values() if config.get('category')}
    
    def reload_configs(self):
        """Reload configurations from file."""
        self.configs.clear()
        self._load_configs()
