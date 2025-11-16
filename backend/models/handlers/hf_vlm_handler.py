"""
HuggingFace Vision-Language Model Handler
Handles standard HF VLM models (BLIP, BLIP2, LFM2, LLaVA, etc.)
"""
import torch
from PIL import Image
from typing import Any, Dict, List, Optional
import logging
from transformers import AutoProcessor, AutoModel
from .base_handler import BaseModelHandler
from .inference_strategies import (
    PromptStrategy, ResponseStrategy,
    DefaultPromptStrategy, DefaultResponseStrategy
)

logger = logging.getLogger(__name__)


class HuggingFaceVLMHandler(BaseModelHandler):
    """Handler for HuggingFace vision-language models."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Initialize strategies (can be overridden by subclasses)
        self.prompt_strategy: PromptStrategy = self._get_prompt_strategy()
        self.response_strategy: ResponseStrategy = self._get_response_strategy()

    def _get_prompt_strategy(self) -> PromptStrategy:
        """Get prompt formatting strategy. Override for custom strategies."""
        return DefaultPromptStrategy()

    def _get_response_strategy(self) -> ResponseStrategy:
        """Get response extraction strategy. Override for custom strategies."""
        return DefaultResponseStrategy()

    def load(self, precision: str = None, use_flash_attention: bool = False, **kwargs) -> None:
        """
        Template method for loading HuggingFace VLM models.
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

            # Setup pad token for batch processing
            self._setup_pad_token()

            # Prepare model loading kwargs
            model_kwargs = self._prepare_model_kwargs(precision, use_flash_attention)

            # Hook: Load model (can be customized)
            self._load_model(model_kwargs)

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
        precision = precision or self.config.get('default_precision', 'float32')

        # Validate precision against supported_precisions if specified
        supported_precisions = self.config.get('supported_precisions')
        if supported_precisions and precision not in supported_precisions:
            logger.warning(
                "%s does not support %s precision. Falling back to %s. "
                "Supported precisions: %s",
                self.model_key, precision, self.config.get('default_precision', 'float32'),
                ', '.join(supported_precisions)
            )
            precision = self.config.get('default_precision', 'float32')

        return precision

    def _load_processor(self, **kwargs):
        """
        Hook for loading processor.
        Override to use custom processor class or loading logic.
        """
        processor_class = self._get_processor_class()
        processor_config = self.config.get('processor_config', {})
        self.processor = processor_class.from_pretrained(self.model_id, **processor_config)

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

    def _load_model(self, model_kwargs: dict):
        """
        Hook for loading model.
        Override to use custom model class or loading logic.
        """
        model_class = self._get_model_class()
        self.model = model_class.from_pretrained(self.model_id, **model_kwargs)

    def _post_load_hook(self):
        """
        Hook for post-load setup.
        Override to add custom post-load configuration.
        """
        # Setup generation config pad token
        self._setup_generation_config_pad_token()
    
    def infer_single(self, image: Image.Image, prompt: Optional[str] = None,
                    parameters: Optional[Dict] = None) -> str:
        """Generate caption for a single image using prompt and response strategies."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")

        try:
            image = self._ensure_rgb(image)

            # Use prompt strategy to format inputs
            inputs = self.prompt_strategy.format_single(
                self.processor, image, prompt, self.device, self.supports_prompts()
            )

            # Move inputs to device if not already done by strategy
            inputs = self._move_inputs_to_device(inputs, match_model_dtype=False)

            # Prepare generation parameters
            gen_params = self._filter_generation_params(parameters)
            gen_params = self._sanitize_generation_params(gen_params)

            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_params)

            # Use response strategy to extract result
            input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else None
            result = self.response_strategy.extract_single(self.processor, output_ids, input_ids)

            # Apply postprocessing if configured
            return self._postprocess(result)

        except Exception as e:
            logger.exception("Error generating caption: %s", e)
            return f"Error: {str(e)}"
    
    def infer_batch(self, images: List[Image.Image], prompts: Optional[List[str]] = None,
                   parameters: Optional[Dict] = None) -> List[str]:
        """Generate captions for multiple images using batch processing with strategies."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")

        try:
            if prompts is None:
                prompts = [None] * len(images)

            # Ensure RGB
            images = [self._ensure_rgb(img) for img in images]

            # Use prompt strategy to format inputs
            inputs = self.prompt_strategy.format_batch(
                self.processor, images, prompts, self.device, self.supports_prompts()
            )

            # If strategy returns None, fall back to sequential processing
            if inputs is None:
                return [self.infer_single(img, prompt, parameters)
                       for img, prompt in zip(images, prompts)]

            # Move inputs to device if not already done by strategy
            inputs = self._move_inputs_to_device(inputs, match_model_dtype=False)

            # Prepare generation parameters
            gen_params = self._filter_generation_params(parameters)
            gen_params = self._sanitize_generation_params(gen_params)

            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_params)

            # Use response strategy to extract results
            input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else None
            results = self.response_strategy.extract_batch(self.processor, output_ids, input_ids)

            # Apply postprocessing
            return [self._postprocess(r) for r in results]

        except Exception as e:
            logger.exception("Error in batch generation: %s", e)
            error_msg = f"Error: {str(e)}"
            return [error_msg] * len(images)
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None and self.processor is not None

    def _get_processor_class(self):
        """Dynamically import and return processor class."""
        processor_name = self.config.get('processor_class', 'AutoProcessor')
        if processor_name == 'AutoProcessor':
            from transformers import AutoProcessor
            return AutoProcessor
        elif processor_name == 'BlipProcessor':
            from transformers import BlipProcessor
            return BlipProcessor
        elif processor_name == 'Blip2Processor':
            from transformers import Blip2Processor
            return Blip2Processor
        else:
            # Fallback to AutoProcessor
            from transformers import AutoProcessor
            return AutoProcessor
    
    def _get_model_class(self):
        """Dynamically import and return model class."""
        model_class_name = self.config.get('model_class', 'AutoModel')
        if model_class_name == 'AutoModel':
            from transformers import AutoModel
            return AutoModel
        elif model_class_name == 'BlipForConditionalGeneration':
            from transformers import BlipForConditionalGeneration
            return BlipForConditionalGeneration
        elif model_class_name == 'Blip2ForConditionalGeneration':
            from transformers import Blip2ForConditionalGeneration
            return Blip2ForConditionalGeneration
        elif model_class_name == 'LlavaForConditionalGeneration':
            from transformers import LlavaForConditionalGeneration
            return LlavaForConditionalGeneration
        elif model_class_name == 'AutoModelForImageTextToText':
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText
        else:
            from transformers import AutoModel
            return AutoModel
    
    def _postprocess(self, text: str) -> str:
        """Apply any postprocessing configured for this model."""
        postprocess_type = self.config.get('postprocess')
        if not postprocess_type:
            return text
        
        if postprocess_type == 'extract_thinking_tags':
            # Remove thinking tags for R4B-style models
            if "</think>" in text:
                parts = text.split("</think>")
                if len(parts) > 1:
                    return parts[-1].replace('\n', ' ').strip()
            if text.startswith("Auto-Thinking Output: "):
                return text[len("Auto-Thinking Output: "):].strip()
        
        return text.strip()
