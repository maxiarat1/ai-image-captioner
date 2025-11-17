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
    
    def _move_inputs_to_device(self, inputs, match_model_dtype: bool = True):
        """
        Move input tensors to device, optionally matching model dtype.

        Args:
            inputs: Dictionary of input tensors or object with .to() method
            match_model_dtype: If True, convert floating point tensors to model's dtype

        Returns:
            Inputs moved to device (and optionally converted to model dtype)
        """
        import torch

        # Determine target dtype if needed
        target_dtype = None
        if match_model_dtype and self.model is not None:
            target_dtype = next(self.model.parameters()).dtype

        # Handle objects with .to() method (like BatchedVLChatProcessorOutput)
        if hasattr(inputs, 'to') and not isinstance(inputs, dict):
            if target_dtype is not None:
                return inputs.to(self.device, dtype=target_dtype)
            else:
                return inputs.to(self.device)

        # Handle dictionary inputs
        if not isinstance(inputs, dict):
            return inputs

        # Convert dictionary inputs
        if target_dtype is not None:
            # Convert floating point tensors to target dtype
            return {
                k: (v.to(self.device, dtype=target_dtype) if torch.is_floating_point(v)
                    else v.to(self.device))
                for k, v in inputs.items()
            }
        else:
            # Just move to device without dtype conversion
            return {k: v.to(self.device) if hasattr(v, 'to') else v
                   for k, v in inputs.items()}

    def _ensure_rgb(self, image):
        """Convert image to RGB if needed."""
        if isinstance(image, list):
            return [img.convert('RGB') if img.mode != 'RGB' else img for img in image]
        return image.convert('RGB') if image.mode != 'RGB' else image

    def _setup_pad_token(self):
        """Ensure tokenizer has a pad token."""
        if hasattr(self.processor, 'tokenizer') and self.processor.tokenizer:
            if not self.processor.tokenizer.pad_token:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

    def _setup_generation_config_pad_token(self):
        """Setup pad_token_id in generation config."""
        try:
            if hasattr(self.processor, 'tokenizer'):
                tok = self.processor.tokenizer
                pad_id = getattr(tok, 'pad_token_id', None) or getattr(tok, 'eos_token_id', None)
                if pad_id is not None and hasattr(self.model, 'generation_config'):
                    self.model.generation_config.pad_token_id = pad_id
        except Exception:
            pass

    def _create_quantization_config(self, precision: str):
        """Create quantization configuration."""
        import torch

        if precision not in ["4bit", "8bit"]:
            return None

        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            logger.warning("BitsAndBytesConfig not available")
            return None

        if precision == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif precision == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )

    def _get_dtype(self, precision: str):
        """Map precision string to torch dtype."""
        import torch

        if precision == "auto":
            return "auto"
        precision_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }
        return precision_map.get(precision, torch.float32)

    def _setup_flash_attention(self, model_kwargs: dict, precision: str):
        """Setup Flash Attention 2 if available."""
        import torch
        from utils.torch_utils import force_cpu_mode

        if force_cpu_mode() or not torch.cuda.is_available():
            return False

        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
            if precision not in ["4bit", "8bit"]:
                model_kwargs["torch_dtype"] = torch.bfloat16
            logger.info("Using Flash Attention 2")
            return True
        except ImportError:
            logger.debug("Flash Attention not available")
            return False

    def _filter_generation_params(self, parameters: Optional[Dict]) -> Dict:
        """Filter parameters to only include valid generation params."""
        if not parameters:
            return {}

        special_params = self.get_special_params()
        # Get valid params from config if available
        available_params = self.config.get('available_parameters', [])
        if available_params:
            valid_keys = {p['param_key'] for p in available_params if 'param_key' in p}
            return {k: v for k, v in parameters.items()
                   if k not in special_params and k in valid_keys}
        else:
            # If no available_parameters defined, just exclude special params
            return {k: v for k, v in parameters.items() if k not in special_params}

    def _sanitize_generation_params(self, parameters: Dict) -> Dict:
        """Sanitize generation parameters to prevent conflicts."""
        if not parameters:
            return {}

        params = parameters.copy()
        do_sample = params.get('do_sample', params.get('sample', False))
        num_beams = params.get('num_beams', 1)

        # Can't combine sampling with beam search
        if do_sample and num_beams > 1:
            logger.warning("Conflicting params: do_sample with num_beams>1")
            do_sample = False
            params['do_sample'] = False

        # Remove sampling params if not sampling
        if not do_sample or num_beams > 1:
            for param in ['temperature', 'top_p', 'top_k']:
                params.pop(param, None)

        # Remove beam search params if num_beams <= 1
        if num_beams <= 1:
            for param in ['length_penalty', 'early_stopping', 'num_beam_groups', 'diversity_penalty']:
                params.pop(param, None)

        return params

    def _format_tags(self, probs, parameters: Optional[Dict] = None) -> str:
        """
        Format prediction probabilities into tag string.
        Shared by HuggingFaceTaggerHandler and ONNXTaggerHandler.

        Args:
            probs: Probability array/tensor for each tag
            parameters: Optional formatting parameters

        Returns:
            Comma-separated string of formatted tags
        """
        params = parameters or {}
        threshold = params.get('threshold', 0.35)
        top_n = params.get('top_n', 50)
        replace_underscores = params.get('replace_underscores', True)
        add_confidence = params.get('add_confidence', False)

        # Get indices of tags above threshold
        indices = [i for i, p in enumerate(probs) if p >= threshold]

        # Sort by probability (descending)
        indices = sorted(indices, key=lambda i: probs[i], reverse=True)

        # Limit to top_n
        indices = indices[:top_n]

        # Format tags
        tags = []
        for idx in indices:
            tag = self.tag_names[idx]
            if replace_underscores:
                tag = tag.replace('_', ' ')

            if add_confidence:
                tag = f"{tag} ({probs[idx]:.2f})"

            tags.append(tag)

        return ', '.join(tags)

    def _check_loaded(self):
        """Check if model is loaded and raise error if not."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")

    def _validate_batch_inputs(self, images: List, prompts: Optional[List[str]] = None) -> List[Optional[str]]:
        """
        Validate and prepare prompts for batch processing.

        Args:
            images: List of images
            prompts: Optional list of prompts

        Returns:
            List of prompts (None for each image if prompts not provided)

        Raises:
            ValueError: If number of prompts doesn't match number of images
        """
        if prompts is None:
            return [None] * len(images)
        elif len(prompts) != len(images):
            raise ValueError(
                f"Number of prompts ({len(prompts)}) must match number of images ({len(images)})"
            )
        return prompts
