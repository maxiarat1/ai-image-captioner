from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from PIL import Image
import logging
import torch

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

    @staticmethod
    def _ensure_rgb(images):
        """Convert image(s) to RGB mode if needed"""
        if isinstance(images, list):
            return [img.convert('RGB') if img.mode != 'RGB' else img for img in images]
        return images.convert('RGB') if images.mode != 'RGB' else images

    def _filter_generation_params(self, parameters: dict, exclude_keys: set = None) -> dict:
        """
        Filter parameters to only include valid generation params for this model.
        
        This method filters in two ways:
        1. Excludes keys in exclude_keys (model-specific special params like 'precision')
        2. Only includes params that are declared in get_available_parameters()
        
        This prevents parameters from other models from being passed through.
        """
        if not parameters:
            return {}
        
        # Get valid parameter keys for this model
        available_params = self.get_available_parameters()
        valid_param_keys = {param['param_key'] for param in available_params if 'param_key' in param}
        
        # Filter: exclude special keys AND only include valid params for this model
        exclude_keys = exclude_keys or set()
        return {
            k: v for k, v in parameters.items() 
            if k not in exclude_keys and k in valid_param_keys
        }

    def _setup_pad_token(self):
        """Ensure tokenizer has a pad token set (use EOS token if not set)"""
        if hasattr(self, 'processor') and self.processor and hasattr(self.processor, 'tokenizer'):
            if not self.processor.tokenizer.pad_token:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

    def _init_device(self, torch_module) -> str:
        """Initialize and return the appropriate device (cuda/mps/cpu)"""
        from utils.torch_utils import pick_device
        return pick_device(torch_module)

    def _create_quantization_config(self, precision: str):
        """Create quantization configuration for 4bit/8bit precision modes"""
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            logger.warning("BitsAndBytesConfig not available, quantization disabled")
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
        return None

    def _get_dtype(self, precision: str):
        """Map precision string to torch dtype"""
        if precision == "auto":
            return "auto"

        precision_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }
        return precision_map.get(precision, torch.float32)

    def _setup_flash_attention(self, model_kwargs: dict, precision: str, force_bfloat16: bool = False) -> bool:
        """
        Setup Flash Attention 2 if available and not in CPU mode.

        Args:
            model_kwargs: Dictionary to add flash attention config to
            precision: Current precision mode (skips if 4bit/8bit)
            force_bfloat16: If True, forces bfloat16 dtype when using flash attention

        Returns:
            True if flash attention was enabled, False otherwise
        """
        from utils.torch_utils import force_cpu_mode

        if force_cpu_mode() or not torch.cuda.is_available():
            return False

        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"

            # Optionally force bfloat16 dtype 
            if force_bfloat16 and precision not in ["4bit", "8bit"]:
                model_kwargs["torch_dtype"] = torch.bfloat16
                logger.info("Using Flash Attention 2 (dtype=bfloat16)")
            else:
                logger.info("Using Flash Attention 2")
            return True
        except ImportError:
            logger.debug("Flash Attention not available; using default attention. Install via: pip install flash-attn --no-build-isolation")
            return False

    def _move_inputs_to_device(self, inputs: dict, device: str, model_dtype: Optional[torch.dtype] = None) -> dict:
        """
        Move input tensors to the specified device with optional dtype conversion.

        Args:
            inputs: Dictionary of input tensors
            device: Target device (cuda/cpu/mps)
            model_dtype: Optional dtype for floating point tensors

        Returns:
            Dictionary with moved tensors
        """
        if model_dtype is not None:
            return {
                k: (v.to(device, dtype=model_dtype) if torch.is_floating_point(v) else v.to(device))
                for k, v in inputs.items()
            }
        else:
            return {k: v.to(device) for k, v in inputs.items()}

    def _format_batch_error(self, error: Exception, batch_size: int) -> list:
        """Format an error for batch processing - returns list of error messages"""
        error_msg = f"Error: {str(error)}"
        return [error_msg] * batch_size

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