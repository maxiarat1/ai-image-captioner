"""
Base Model Handler
Abstract base class that defines the interface for all model handlers.
Handlers contain the actual loading and inference logic, decoupled from adapters.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class BaseModelHandler(ABC):
    """
    Base handler for model operations.
    
    Handlers are responsible for:
    - Loading models with specific configurations
    - Running inference (single and batch)
    - Preprocessing/postprocessing
    - Device and precision management
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize handler with model configuration.
        
        Args:
            config: Dictionary containing model configuration from JSONL
        """
        self.config = config
        self.model_id = config['model_id']
        self.model_key = config['model_key']
        self.model = None
        self.processor = None
        self.device = None
        
    @abstractmethod
    def load(self, precision: str = None, use_flash_attention: bool = False, **kwargs) -> None:
        """
        Load the model with specified configuration.
        
        Args:
            precision: Precision mode (float32, float16, bfloat16, 4bit, 8bit, auto)
            use_flash_attention: Whether to use Flash Attention 2
            **kwargs: Additional model-specific parameters
        """
        pass
    
    @abstractmethod
    def infer_single(self, image: Image.Image, prompt: Optional[str] = None, 
                     parameters: Optional[Dict] = None) -> str:
        """
        Run inference on a single image.
        
        Args:
            image: PIL Image to process
            prompt: Optional text prompt
            parameters: Optional generation parameters
            
        Returns:
            Generated text/caption/tags
        """
        pass
    
    def infer_batch(self, images: List[Image.Image], prompts: Optional[List[str]] = None,
                   parameters: Optional[Dict] = None) -> List[str]:
        """
        Run inference on multiple images. Default: sequential processing.
        Override for optimized batch processing.
        
        Args:
            images: List of PIL Images
            prompts: Optional list of prompts (one per image)
            parameters: Optional generation parameters
            
        Returns:
            List of generated texts
        """
        if prompts is None:
            prompts = [None] * len(images)
        return [self.infer_single(img, prompt, parameters) 
                for img, prompt in zip(images, prompts)]
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass
    
    def unload(self) -> None:
        """Unload model from memory."""
        if self.model is not None:
            logger.info("Unloading %s model…", self.model_key)
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        import gc
        import torch
        from utils.torch_utils import maybe_empty_cuda_cache
        gc.collect()
        maybe_empty_cuda_cache(torch)
    
    def get_special_params(self) -> set:
        """Get set of special parameters that shouldn't be passed to model.generate()."""
        base_params = {'precision', 'use_flash_attention', 'batch_size'}
        special = set(self.config.get('special_params', []))
        return base_params | special
    
    def supports_prompts(self) -> bool:
        """Check if model supports text prompts."""
        return self.config.get('supports_prompts', False)
    
    def supports_batch(self) -> bool:
        """Check if model supports batch processing."""
        return self.config.get('supports_batch', False)
