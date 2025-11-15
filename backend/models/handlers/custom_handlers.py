"""
Custom Model Handlers
Specialized handlers for models that need unique logic (Janus, R4B, TrOCR)
"""
import torch
from PIL import Image
from typing import Any, Dict, List, Optional
import logging
from .hf_vlm_handler import HuggingFaceVLMHandler
from .hf_ocr_handler import HuggingFaceOCRHandler

logger = logging.getLogger(__name__)


class JanusHandler(HuggingFaceVLMHandler):
    """
    Custom handler for Janus models (requires VLChatProcessor).
    Inherits from HF VLM handler with specialized processor handling.
    """
    
    def _get_processor_class(self):
        """Import VLChatProcessor for Janus models."""
        try:
            from transformers import VLChatProcessor
            return VLChatProcessor
        except ImportError:
            logger.error("VLChatProcessor not available for Janus model")
            raise
    
    def _get_model_class(self):
        """Use AutoModelForCausalLM for Janus."""
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM
    
    def infer_single(self, image: Image.Image, prompt: Optional[str] = None, 
                    parameters: Optional[Dict] = None) -> str:
        """Generate caption using Janus-specific conversation format."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")
        
        try:
            image = self._ensure_rgb(image)
            
            # Janus uses conversation format
            conversation = [
                {
                    "role": "User",
                    "content": prompt.strip() if prompt and prompt.strip() else "Describe this image in detail.",
                    "images": [image],
                },
                {"role": "Assistant", "content": ""},
            ]
            
            # Prepare inputs using VLChatProcessor
            pil_images = [image]
            prepare_inputs = self.processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True
            ).to(self.device)
            
            # Prepare generation parameters
            gen_params = self._filter_generation_params(parameters)
            gen_params = self._sanitize_generation_params(gen_params)
            
            # Generate
            with torch.no_grad():
                inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
                outputs = self.model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    bos_token_id=self.processor.tokenizer.bos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    **gen_params
                )
            
            # Decode
            result = self.processor.tokenizer.decode(
                outputs[0].cpu().tolist(),
                skip_special_tokens=True
            )
            
            return result.strip()
            
        except Exception as e:
            logger.exception("Error with Janus inference: %s", e)
            return f"Error: {str(e)}"


class R4BHandler(HuggingFaceVLMHandler):
    """
    Custom handler for R4B model with thinking mode support.
    R4B only supports float32, 4bit, and 8bit precision modes.
    """
    
    def infer_single(self, image: Image.Image, prompt: Optional[str] = None, 
                    parameters: Optional[Dict] = None) -> str:
        """Generate caption with R4B thinking mode support."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")
        
        try:
            image = self._ensure_rgb(image)
            
            # Extract thinking mode
            thinking_mode = self._extract_thinking_mode(parameters)
            
            # Build prompt
            user_prompt = prompt.strip() if prompt and prompt.strip() else "Describe this image."
            
            # R4B requires structured message format with image object
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt}
                ]
            }]
            
            # Apply chat template with thinking_mode parameter
            text_prompt = self.processor.apply_chat_template(
                messages, 
                tokenize=False,
                add_generation_prompt=True, 
                thinking_mode=thinking_mode
            )
            
            # Process inputs with the formatted text
            inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")
            
            # Move inputs to device with proper dtype
            model_dtype = next(self.model.parameters()).dtype
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            if model_dtype != torch.float32:
                inputs = {k: v.to(model_dtype) if hasattr(v, 'to') and v.dtype in [torch.float16, torch.float32, torch.bfloat16] else v 
                         for k, v in inputs.items()}
            
            # Prepare generation parameters
            gen_params = self._filter_generation_params(parameters)
            gen_params = self._sanitize_generation_params(gen_params)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **gen_params)
            
            # Decode only the generated portion (skip input tokens)
            output_ids = generated_ids[0][len(inputs["input_ids"][0]):]
            result = self.processor.decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            # Extract final result (remove thinking tags)
            return self._extract_final_result(result)
            
        except Exception as e:
            logger.exception("Error with R4B inference: %s", e)
            return f"Error: {str(e)}"
    
    def _extract_thinking_mode(self, parameters: Optional[Dict]) -> str:
        """Extract and validate thinking_mode parameter."""
        if not parameters:
            return 'auto'
        thinking_mode = parameters.get('thinking_mode', 'auto')
        return thinking_mode if thinking_mode in ['auto', 'short', 'long'] else 'auto'
    
    def _extract_final_result(self, caption: str) -> str:
        """Extract final caption from R4B output (removes thinking tags)."""
        if not caption:
            return caption
        if "</think>" in caption:
            parts = caption.split("</think>")
            if len(parts) > 1:
                return parts[-1].replace('\n', ' ').strip()
        if caption.startswith("Auto-Thinking Output: "):
            return caption[len("Auto-Thinking Output: "):].strip()
        return caption.strip()


class TrOCRHandler(HuggingFaceOCRHandler):
    """
    Custom handler for TrOCR with text detection support.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.detection_model = None
        self.detection_processor = None
    
    def load(self, precision: str = None, use_flash_attention: bool = False, **kwargs) -> None:
        """Load TrOCR model with optional text detection."""
        try:
            from utils.torch_utils import pick_device
            self.device = pick_device(torch)
            
            precision = precision or self.config.get('default_precision', 'float32')
            use_fast = kwargs.get('use_fast', True)
            
            logger.info("Loading %s on %s with %s precision…", self.model_key, self.device, precision)
            
            # Load TrOCR processor and model
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            processor_config = self.config.get('processor_config', {})
            processor_config['use_fast'] = use_fast
            self.processor = TrOCRProcessor.from_pretrained(self.model_id, **processor_config)
            
            # Prepare model loading
            model_kwargs = {}
            quantization_config = self._create_quantization_config(precision)
            if quantization_config:
                model_kwargs['quantization_config'] = quantization_config
            else:
                model_kwargs['torch_dtype'] = self._get_dtype(precision)
            
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_id, **model_kwargs)
            
            if precision not in ["4bit", "8bit"]:
                self.model.to(self.device)
            
            self.model.eval()
            
            # Load detection model if needed
            if self.config.get('requires_text_detection'):
                self._load_detection_model()
            
            logger.info("%s loaded successfully", self.model_key)
            
        except Exception as e:
            logger.exception("Error loading %s: %s", self.model_key, e)
            raise
    
    def _load_detection_model(self):
        """Load text detection model (CRAFT)."""
        try:
            from transformers import AutoProcessor, AutoModelForObjectDetection
            
            detection_model_id = "microsoft/table-transformer-detection"
            self.detection_processor = AutoProcessor.from_pretrained(detection_model_id)
            self.detection_model = AutoModelForObjectDetection.from_pretrained(detection_model_id)
            self.detection_model.to(self.device)
            self.detection_model.eval()
            
            logger.info("Text detection model loaded")
        except Exception as e:
            logger.warning("Could not load text detection model: %s", e)
    
    def infer_single(self, image: Image.Image, prompt: Optional[str] = None, 
                    parameters: Optional[Dict] = None) -> str:
        """Extract text using TrOCR with optional detection."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")
        
        try:
            image = self._ensure_rgb(image)
            
            # If detection is available, detect text regions first
            if self.detection_model:
                # For simplicity, process whole image
                # In production, you'd crop detected regions
                pass
            
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            gen_params = self._filter_generation_params(parameters)
            
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_params)
            
            # Decode
            result = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            return result.strip()
            
        except Exception as e:
            logger.exception("Error with TrOCR: %s", e)
            return f"Error: {str(e)}"
