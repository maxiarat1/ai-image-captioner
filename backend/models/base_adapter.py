from abc import ABC, abstractmethod
from typing import Any, Dict
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class BaseModelAdapter(ABC):
    """Base class for all model adapters"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.processor = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the model and processor"""
        pass

    @abstractmethod
    def generate_caption(self, image: Image.Image, prompt: str = None) -> str:
        """Generate caption for the given image"""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": self.model_name,
            "loaded": self.is_loaded(),
            "parameters": self.get_available_parameters()
        }

    def get_available_parameters(self) -> list:
        """
        Get list of available parameters for this model.
        Each parameter is a dict with: name, type, min, max, step, default_description, param_key

        Should be overridden by subclasses to define model-specific parameters.
        """
        return []

    def unload(self) -> None:
        """
        Unload the model and free memory.
        This method should be called before reloading with different parameters.
        """
        if hasattr(self, 'model') and self.model is not None:
            logger.info("Unloading %s model…", self.model_name)
            # Delete model to free GPU memory
            del self.model
            self.model = None
            
        if hasattr(self, 'processor') and self.processor is not None:
            # Delete processor to free memory  
            del self.processor
            self.processor = None
            
        # Force garbage collection and clear GPU cache
        import gc
        import torch
        from utils.torch_utils import maybe_empty_cuda_cache
        gc.collect()
        # Safely clear CUDA cache if applicable
        maybe_empty_cuda_cache(torch)