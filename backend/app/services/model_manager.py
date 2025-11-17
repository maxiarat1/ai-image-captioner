"""
Model Manager Service
Handles model lifecycle: loading, unloading, caching, and precision parameter management.
"""
import logging
from typing import Optional, Dict, Any

from app.models import MODEL_METADATA, validate_model_name, get_available_models, get_factory

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages the lifecycle of AI models including loading, unloading, and caching.
    Implements single-model-at-a-time loading strategy to manage memory.
    """

    def __init__(self):
        """Initialize the model manager with empty caches."""
        self.models: Dict[str, Any] = {name: None for name in MODEL_METADATA.keys()}
        self._model_info_cache: Dict[str, Dict] = {}
        logger.debug("ModelManager initialized")

    def initialize(self):
        """Log initialization message about model registry."""
        logger.info("Model registry ready. Models load on first use.")
        logger.info("Available models: %s", ", ".join(get_available_models()))

    def get_model(self, model_name: str, precision_params: Optional[Dict] = None,
                  force_reload: bool = False):
        """
        Get a model instance, loading it if necessary.
        Automatically unloads other models to conserve memory.

        Args:
            model_name: Name of the model to get
            precision_params: Optional precision parameters (precision, use_flash_attention, etc.)
            force_reload: If True, reload the model even if already loaded

        Returns:
            The loaded model adapter instance

        Raises:
            ValueError: If model_name is not in the registry
            Exception: If model loading fails
        """
        if not validate_model_name(model_name):
            raise ValueError(f"Unknown model: {model_name}")

        # Unload other models to save memory (single model policy)
        self._unload_other_models(model_name)

        # Check if reload is needed due to precision parameter changes
        should_reload = force_reload or self._needs_precision_reload(model_name, precision_params)

        # Load or reload if necessary
        if self.models[model_name] is None or should_reload:
            self._load_model(model_name, precision_params, should_reload)

        return self.models[model_name]

    def unload_model(self, model_name: str):
        """
        Unload a specific model from memory.

        Args:
            model_name: Name of the model to unload

        Raises:
            ValueError: If model_name is not in the registry
        """
        if not validate_model_name(model_name):
            raise ValueError(f"Unknown model: {model_name}")

        if self.models[model_name] is not None:
            logger.info("Unloading model: %s", model_name)
            self.models[model_name].unload()
            self.models[model_name] = None

    def is_loaded(self, model_name: str) -> bool:
        """
        Check if a model is currently loaded.

        Args:
            model_name: Name of the model to check

        Returns:
            True if model is loaded, False otherwise
        """
        if not validate_model_name(model_name):
            return False
        return self.models[model_name] is not None and self.models[model_name].is_loaded()

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get model parameter information (cached).

        Args:
            model_name: Name of the model

        Returns:
            Dict with model name and available parameters

        Raises:
            ValueError: If model_name is not in the registry
            Exception: If getting model info fails
        """
        if not validate_model_name(model_name):
            raise ValueError(f"Unknown model: {model_name}")

        if model_name not in self._model_info_cache:
            try:
                metadata = MODEL_METADATA[model_name]
                # The adapter is now a callable that returns the adapter instance
                adapter = metadata['adapter']()
                self._model_info_cache[model_name] = {
                    "name": model_name,
                    "parameters": adapter.get_available_parameters()
                }
            except Exception as e:
                logger.exception("Error getting model info for %s: %s", model_name, e)
                raise

        return self._model_info_cache[model_name]

    def get_all_models_status(self) -> Dict[str, bool]:
        """
        Get load status for all models.

        Returns:
            Dict mapping model names to their loaded status
        """
        return {
            name: model is not None and model.is_loaded()
            for name, model in self.models.items()
        }

    def _unload_other_models(self, current_model_name: str):
        """
        Unload all models except the specified one.

        Args:
            current_model_name: Name of the model to keep loaded
        """
        for other_name, other_model in self.models.items():
            if other_name != current_model_name and other_model is not None:
                logger.info("Switching model: unloading %s", other_name)
                other_model.unload()
                self.models[other_name] = None

    def _needs_precision_reload(self, model_name: str, precision_params: Optional[Dict]) -> bool:
        """
        Check if model needs to be reloaded due to precision parameter changes.

        Args:
            model_name: Name of the model
            precision_params: New precision parameters to apply

        Returns:
            True if reload is needed, False otherwise
        """
        # Get factory to check if model has precision defaults
        factory = get_factory()
        defaults = factory.get_precision_defaults(model_name)

        if not precision_params or not defaults:
            return False

        current_model = self.models[model_name]
        if not current_model:
            return False

        if not hasattr(current_model, 'current_precision_params'):
            return False

        return current_model.current_precision_params != precision_params

    def _load_model(self, model_name: str, precision_params: Optional[Dict], is_reload: bool):
        """
        Load or reload a model with the specified precision parameters.

        Args:
            model_name: Name of the model to load
            precision_params: Optional precision parameters
            is_reload: Whether this is a reload operation

        Raises:
            Exception: If model loading fails
        """
        # Unload existing model if reloading
        if is_reload and self.models[model_name]:
            self.models[model_name].unload()
            self.models[model_name] = None

        try:
            metadata = MODEL_METADATA[model_name]
            action = "Reloading" if is_reload else "Loading"
            logger.info("%s %s model on-demand…", action, model_name)

            # Create adapter instance using factory
            # The adapter is now a callable that returns the adapter
            adapter_factory = metadata['adapter']
            adapter = adapter_factory()

            # Load model with precision parameters
            if precision_params:
                adapter.load_model(**precision_params)
                adapter.current_precision_params = precision_params.copy()
            else:
                adapter.load_model()
                # Set default precision params if available
                factory = get_factory()
                defaults = factory.get_precision_defaults(model_name)
                if defaults:
                    adapter.current_precision_params = defaults

            self.models[model_name] = adapter
            logger.info("Successfully loaded model: %s", model_name)

        except Exception as e:
            logger.exception("Failed to load %s model: %s", model_name, e)
            self.models[model_name] = None
            raise
