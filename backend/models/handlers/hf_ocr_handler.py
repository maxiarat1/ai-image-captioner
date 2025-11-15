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
        """Load HuggingFace OCR model."""
        try:
            from utils.torch_utils import pick_device
            self.device = pick_device(torch)
            
            precision = precision or self.config.get('default_precision', 'bfloat16')
            logger.info("Loading %s on %s with %s precision…", self.model_key, self.device, precision)
            
            # Load processor
            processor_config = self.config.get('processor_config', {})
            self.processor = AutoProcessor.from_pretrained(self.model_id, **processor_config)
            
            # Setup pad token
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
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **model_kwargs)
            
            if precision not in ["4bit", "8bit"]:
                self.model.to(self.device)
            
            self.model.eval()
            logger.info("%s loaded successfully", self.model_key)
            
        except Exception as e:
            logger.exception("Error loading %s: %s", self.model_key, e)
            raise
    
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
    
    # Helper methods
    
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
    
    def _create_quantization_config(self, precision: str):
        """Create quantization configuration."""
        if precision not in ["4bit", "8bit"]:
            return None
        
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            return None
        
        if precision == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif precision == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
    
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
            return False
    
    def _filter_generation_params(self, parameters: Optional[Dict]) -> Dict:
        """Filter parameters to only include valid generation params."""
        if not parameters:
            return {}
        
        special_params = self.get_special_params()
        return {k: v for k, v in parameters.items() if k not in special_params}
    
    def _sanitize_generation_params(self, parameters: Dict) -> Dict:
        """Sanitize generation parameters."""
        if not parameters:
            return {}
        
        params = parameters.copy()
        do_sample = params.get('do_sample', False)
        num_beams = params.get('num_beams', 1)
        
        if do_sample and num_beams > 1:
            do_sample = False
            params['do_sample'] = False
        
        if not do_sample or num_beams > 1:
            for param in ['temperature', 'top_p', 'top_k']:
                params.pop(param, None)
        
        if num_beams <= 1:
            for param in ['length_penalty', 'early_stopping']:
                params.pop(param, None)
        
        return params
