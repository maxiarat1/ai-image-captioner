from abc import ABC, abstractmethod
from typing import Any, Dict, List
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class BaseModelAdapter(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.processor = None

    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def generate_caption(self, image: Image.Image, prompt: str = None) -> str:
        pass

    def generate_captions_batch(self, images: List[Image.Image], prompts: List[str] = None, parameters: dict = None) -> List[str]:
        """
        Generate captions for multiple images. Default implementation processes sequentially.
        Subclasses can override for optimized batch processing.
        """
        if prompts is None:
            prompts = [None] * len(images)
        return [self.generate_caption(img, prompt, parameters) for img, prompt in zip(images, prompts)]

    @abstractmethod
    def is_loaded(self) -> bool:
        pass

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.model_name,
            "loaded": self.is_loaded(),
            "parameters": self.get_available_parameters()
        }

    def get_available_parameters(self) -> list:
        return []

    def unload(self) -> None:
        if hasattr(self, 'model') and self.model is not None:
            logger.info("Unloading %s model…", self.model_name)
            del self.model
            self.model = None

        if hasattr(self, 'processor') and self.processor is not None:
            del self.processor
            self.processor = None

        import gc
        import torch
        from utils.torch_utils import maybe_empty_cuda_cache
        gc.collect()
        maybe_empty_cuda_cache(torch)