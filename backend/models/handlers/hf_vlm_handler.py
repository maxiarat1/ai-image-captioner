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

logger = logging.getLogger(__name__)


class HuggingFaceVLMHandler(BaseModelHandler):
    """Handler for HuggingFace vision-language models."""
    
    def load(self, precision: str = None, use_flash_attention: bool = False, **kwargs) -> None:
        """Load HuggingFace VLM model."""
        try:
            from utils.torch_utils import pick_device
            self.device = pick_device(torch)
            
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
            
            logger.info("Loading %s on %s with %s precision…", self.model_key, self.device, precision)
            
            # Load processor
            processor_class = self._get_processor_class()
            processor_config = self.config.get('processor_config', {})
            self.processor = processor_class.from_pretrained(self.model_id, **processor_config)
            
            # Setup pad token for batch processing
            self._setup_pad_token()
            
            # Prepare model loading kwargs
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
            
            # Load model
            model_class = self._get_model_class()
            self.model = model_class.from_pretrained(self.model_id, **model_kwargs)
            
            # Move to device if not quantized
            if precision not in ["4bit", "8bit"]:
                self.model.to(self.device)
            
            # Setup generation config pad token
            self._setup_generation_config_pad_token()
            
            self.model.eval()
            logger.info("%s loaded successfully", self.model_key)
            
        except Exception as e:
            logger.exception("Error loading %s: %s", self.model_key, e)
            raise
    
    def infer_single(self, image: Image.Image, prompt: Optional[str] = None, 
                    parameters: Optional[Dict] = None) -> str:
        """Generate caption for a single image."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")
        
        try:
            image = self._ensure_rgb(image)
            
            # Prepare inputs
            if prompt and prompt.strip() and self.supports_prompts():
                inputs = self.processor(image, prompt.strip(), return_tensors="pt")
            else:
                inputs = self.processor(image, return_tensors="pt")
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Prepare generation parameters
            gen_params = self._filter_generation_params(parameters)
            gen_params = self._sanitize_generation_params(gen_params)
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_params)
            
            # Decode
            result = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            
            # Apply postprocessing if configured
            return self._postprocess(result)
            
        except Exception as e:
            logger.exception("Error generating caption: %s", e)
            return f"Error: {str(e)}"
    
    def infer_batch(self, images: List[Image.Image], prompts: Optional[List[str]] = None,
                   parameters: Optional[Dict] = None) -> List[str]:
        """Generate captions for multiple images using batch processing."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")
        
        try:
            if prompts is None:
                prompts = [None] * len(images)
            
            # Ensure RGB
            images = [self._ensure_rgb(img) for img in images]
            
            # Prepare inputs
            if prompts[0] and self.supports_prompts():
                text_prompts = [p.strip() if p else "" for p in prompts]
                inputs = self.processor(images=images, text=text_prompts, 
                                       return_tensors="pt", padding=True)
            else:
                inputs = self.processor(images=images, return_tensors="pt", padding=True)
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Prepare generation parameters
            gen_params = self._filter_generation_params(parameters)
            gen_params = self._sanitize_generation_params(gen_params)
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_params)
            
            # Decode
            results = self.processor.batch_decode(output_ids, skip_special_tokens=True)
            
            # Apply postprocessing
            return [self._postprocess(r) for r in results]
            
        except Exception as e:
            logger.exception("Error in batch generation: %s", e)
            error_msg = f"Error: {str(e)}"
            return [error_msg] * len(images)
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None and self.processor is not None
    
    # Helper methods
    
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
