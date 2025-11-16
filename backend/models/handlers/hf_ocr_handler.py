"""
HuggingFace OCR Handler
Handles OCR models (Nanonets, Chandra, etc.)
"""
import torch
from PIL import Image
from typing import Any, Dict, List, Optional
import logging
from transformers import AutoProcessor, AutoModelForImageTextToText
from .base_handler import BaseModelHandler

logger = logging.getLogger(__name__)


class HuggingFaceOCRHandler(BaseModelHandler):
    """Handler for HuggingFace OCR models."""
    
    def load(self, precision: str = None, use_flash_attention: bool = False, **kwargs) -> None:
        """
        Template method for loading HuggingFace OCR models.
        Subclasses can override hooks to customize specific steps.
        """
        try:
            from utils.torch_utils import pick_device
            self.device = pick_device(torch)

            # Hook: Pre-load validation and setup
            precision = self._pre_load_hook(precision, **kwargs)

            logger.info("Loading %s on %s with %s precision…", self.model_key, self.device, precision)

            # Hook: Load processor (can be customized)
            self._load_processor(**kwargs)

            # Setup pad token
            self._setup_pad_token()

            # Prepare model loading kwargs
            model_kwargs = self._prepare_model_kwargs(precision, use_flash_attention)

            # Hook: Load model (can be customized)
            self._load_model(model_kwargs, precision)

            # Move to device if not quantized
            if precision not in ["4bit", "8bit"]:
                self.model.to(self.device)

            # Hook: Post-load setup
            self._post_load_hook()

            self.model.eval()
            logger.info("%s loaded successfully", self.model_key)

        except Exception as e:
            logger.exception("Error loading %s: %s", self.model_key, e)
            raise

    def _pre_load_hook(self, precision: str = None, **kwargs) -> str:
        """
        Hook for pre-load validation and setup.
        Override to add custom validation logic.

        Returns:
            Validated precision string
        """
        return precision or self.config.get('default_precision', 'bfloat16')

    def _load_processor(self, **kwargs):
        """
        Hook for loading processor.
        Override to use custom processor class or loading logic.
        """
        processor_config = self.config.get('processor_config', {})
        self.processor = AutoProcessor.from_pretrained(self.model_id, **processor_config)

    def _prepare_model_kwargs(self, precision: str, use_flash_attention: bool) -> dict:
        """
        Prepare model loading kwargs.
        Override to customize model loading arguments.
        """
        model_kwargs = self.config.get('model_config', {}).copy()

        # Handle quantization
        quantization_config = self._create_quantization_config(precision)
        if quantization_config:
            model_kwargs['quantization_config'] = quantization_config
        else:
            model_kwargs['torch_dtype'] = self._get_dtype(precision)

        # Setup flash attention if requested
        if use_flash_attention:
            self._setup_flash_attention(model_kwargs, precision)

        return model_kwargs

    def _load_model(self, model_kwargs: dict, precision: str):
        """
        Hook for loading model.
        Override to use custom model class or loading logic.
        """
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **model_kwargs)

    def _post_load_hook(self):
        """
        Hook for post-load setup.
        Override to add custom post-load configuration.
        """
        pass
    
    def infer_single(self, image: Image.Image, prompt: Optional[str] = None, 
                    parameters: Optional[Dict] = None) -> str:
        """Extract text from a single image."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")
        
        try:
            image = self._ensure_rgb(image)
            
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Prepare generation parameters
            gen_params = self._filter_generation_params(parameters)
            gen_params = self._sanitize_generation_params(gen_params)
            
            # Apply preset if configured (for Chandra)
            if 'chandra_preset' in parameters:
                gen_params = self._apply_chandra_preset(parameters['chandra_preset'], gen_params)
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_params)
            
            # Decode
            result = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            return result.strip()
            
        except Exception as e:
            logger.exception("Error extracting text: %s", e)
            return f"Error: {str(e)}"
    
    def infer_batch(self, images: List[Image.Image], prompts: Optional[List[str]] = None,
                   parameters: Optional[Dict] = None) -> List[str]:
        """Extract text from multiple images."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")
        
        try:
            images = [self._ensure_rgb(img) for img in images]
            
            # Process batch
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Prepare generation parameters
            gen_params = self._filter_generation_params(parameters)
            gen_params = self._sanitize_generation_params(gen_params)
            
            if 'chandra_preset' in parameters:
                gen_params = self._apply_chandra_preset(parameters['chandra_preset'], gen_params)
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_params)
            
            # Decode
            results = self.processor.batch_decode(output_ids, skip_special_tokens=True)
            return [r.strip() for r in results]
            
        except Exception as e:
            logger.exception("Error in batch OCR: %s", e)
            error_msg = f"Error: {str(e)}"
            return [error_msg] * len(images)
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None and self.processor is not None

    def _apply_chandra_preset(self, preset: str, gen_params: Dict) -> Dict:
        """Apply Chandra OCR preset configurations."""
        presets = {
            'fast': {'max_new_tokens': 512},
            'balanced': {'max_new_tokens': 1024},
            'detailed': {'max_new_tokens': 2048}
        }

        if preset in presets:
            gen_params.update(presets[preset])

        return gen_params
