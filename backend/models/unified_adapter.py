"""
Unified Model Adapter
Simplified adapter that delegates all work to handlers.
This replaces the need for individual model adapter files.
"""
from typing import Any, Dict, List
from PIL import Image
import logging
from models.handlers.base_handler import BaseModelHandler

logger = logging.getLogger(__name__)


class UnifiedModelAdapter:
    """
    Universal adapter that works with any model handler.
    
    This adapter is a thin wrapper around handlers, providing a consistent
    interface while delegating all actual work to the appropriate handler.
    """
    
    def __init__(self, handler: BaseModelHandler):
        """
        Initialize adapter with a model handler.
        
        Args:
            handler: The handler instance for this model
        """
        self.handler = handler
        self.model_name = handler.model_key
        self.config = handler.config
        self.current_precision_params = None
    
    def load_model(self, precision: str = None, use_flash_attention: bool = False, **kwargs) -> None:
        """
        Load the model using the handler.
        
        Args:
            precision: Precision mode (float32, float16, bfloat16, 4bit, 8bit, auto)
            use_flash_attention: Whether to use Flash Attention 2
            **kwargs: Additional model-specific parameters
        """
        self.handler.load(precision=precision, use_flash_attention=use_flash_attention, **kwargs)
        
        # Track precision params for reload detection
        self.current_precision_params = {
            'precision': precision,
            'use_flash_attention': use_flash_attention,
            **kwargs
        }
    
    def generate_caption(self, image: Image.Image, prompt: str = None, parameters: dict = None) -> str:
        """
        Generate caption/text for a single image.
        
        Args:
            image: PIL Image to process
            prompt: Optional text prompt
            parameters: Optional generation parameters
            
        Returns:
            Generated text/caption/tags
        """
        return self.handler.infer_single(image, prompt, parameters)
    
    def generate_captions_batch(self, images: List[Image.Image], prompts: List[str] = None, 
                               parameters: dict = None) -> List[str]:
        """
        Generate captions/text for multiple images.
        
        Args:
            images: List of PIL Images
            prompts: Optional list of prompts (one per image)
            parameters: Optional generation parameters
            
        Returns:
            List of generated texts
        """
        return self.handler.infer_batch(images, prompts, parameters)
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.handler.is_loaded()
    
    def unload(self) -> None:
        """Unload model from memory."""
        self.handler.unload()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            "name": self.model_name,
            "loaded": self.is_loaded(),
            "parameters": self.get_available_parameters(),
            "category": self.config.get('category'),
            "description": self.config.get('description'),
            "vlm_capable": self.config.get('vlm_capable', False),
            "supports_prompts": self.handler.supports_prompts(),
            "supports_batch": self.handler.supports_batch()
        }
    
    def get_available_parameters(self) -> list:
        """
        Get list of available parameters for this model.
        
        Returns:
            List of parameter specifications
        """
        # Return model-specific parameters if defined in config
        if 'available_parameters' in self.config:
            return self.config['available_parameters']
        
        # Otherwise return common parameters based on model type
        model_type = self.config.get('type')
        
        if model_type in ['hf_vlm', 'hf_vlm_custom', 'hf_ocr', 'hf_ocr_trocr', 'hf_ocr_custom']:
            return self._get_vlm_parameters()
        elif model_type in ['hf_tagger', 'onnx_tagger']:
            return self._get_tagger_parameters()
        elif model_type == 'hf_classifier':
            return self._get_classifier_parameters()
        
        return []
    
    def _get_vlm_parameters(self) -> list:
        """Get standard VLM generation parameters."""
        # Get precision options based on model type and config
        precision_options = self._get_precision_options()
        
        params = [
            {
                "name": "Precision",
                "param_key": "precision",
                "type": "select",
                "options": precision_options,
                "description": "Model precision/quantization mode (requires model reload)",
                "group": "model_loading"
            },
            {
                "name": "Flash Attention",
                "param_key": "use_flash_attention",
                "type": "checkbox",
                "description": "Use Flash Attention 2 for faster inference (requires compatible GPU, model reload)",
                "group": "model_loading"
            },
            {
                "name": "Max Tokens",
                "param_key": "max_new_tokens",
                "type": "number",
                "min": 10,
                "max": 500,
                "step": 1,
                "description": "Maximum number of tokens to generate",
                "group": "general"
            },
            {
                "name": "Do Sample",
                "param_key": "do_sample",
                "type": "checkbox",
                "description": "Enable sampling mode (required for temperature, top_p, top_k)",
                "group": "sampling"
            },
            {
                "name": "Temperature",
                "param_key": "temperature",
                "type": "number",
                "min": 0.1,
                "max": 2.0,
                "step": 0.1,
                "description": "Sampling temperature (requires do_sample=true)",
                "depends_on": "do_sample",
                "group": "sampling"
            },
            {
                "name": "Top P",
                "param_key": "top_p",
                "type": "number",
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "description": "Nucleus sampling threshold (requires do_sample=true)",
                "depends_on": "do_sample",
                "group": "sampling"
            },
            {
                "name": "Top K",
                "param_key": "top_k",
                "type": "number",
                "min": 0,
                "max": 100,
                "step": 1,
                "description": "Top-k sampling parameter (requires do_sample=true)",
                "depends_on": "do_sample",
                "group": "sampling"
            },
            {
                "name": "Num Beams",
                "param_key": "num_beams",
                "type": "number",
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Number of beams for beam search (conflicts with do_sample)",
                "group": "beam_search"
            },
            {
                "name": "Batch Size",
                "param_key": "batch_size",
                "type": "number",
                "min": 1,
                "max": 16,
                "step": 1,
                "description": "Number of images to process simultaneously (higher = faster but more VRAM)",
                "group": "advanced"
            }
        ]
        
        # Add special parameters from config
        special_params = self.config.get('special_params', [])
        if 'thinking_mode' in special_params:
            params.append({
                "name": "Thinking Mode",
                "param_key": "thinking_mode",
                "type": "select",
                "options": [
                    {"value": "auto", "label": "Auto"},
                    {"value": "short", "label": "Short"},
                    {"value": "long", "label": "Long"}
                ],
                "description": "Control reasoning depth",
                "group": "general"
            })
        
        if 'chandra_preset' in special_params:
            params.append({
                "name": "OCR Preset",
                "param_key": "chandra_preset",
                "type": "select",
                "options": [
                    {"value": "fast", "label": "Fast"},
                    {"value": "balanced", "label": "Balanced"},
                    {"value": "detailed", "label": "Detailed"}
                ],
                "description": "OCR quality preset",
                "group": "general"
            })
        
        return params
    
    def _get_tagger_parameters(self) -> list:
        """Get standard tagging parameters."""
        # Get precision options
        precision_options = self._get_precision_options()
        
        params = [
            {
                "name": "Precision",
                "param_key": "precision",
                "type": "select",
                "options": precision_options,
                "description": "Model precision mode (requires model reload)",
                "group": "model_loading"
            },
            {
                "name": "Threshold",
                "param_key": "threshold",
                "type": "number",
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "description": "Minimum confidence threshold for tags",
                "group": "general"
            },
            {
                "name": "Max Tags",
                "param_key": "top_n",
                "type": "number",
                "min": 1,
                "max": 200,
                "step": 1,
                "description": "Maximum number of tags to return",
                "group": "general"
            },
            {
                "name": "Replace Underscores",
                "param_key": "replace_underscores",
                "type": "checkbox",
                "description": "Replace underscores with spaces in tags",
                "group": "general"
            },
            {
                "name": "Show Confidence",
                "param_key": "add_confidence",
                "type": "checkbox",
                "description": "Include confidence scores in output",
                "group": "general"
            },
            {
                "name": "Batch Size",
                "param_key": "batch_size",
                "type": "number",
                "min": 1,
                "max": 32,
                "step": 1,
                "description": "Number of images to process simultaneously (higher = faster but more VRAM)",
                "group": "advanced"
            }
        ]
        
        return params
    
    def _get_classifier_parameters(self) -> list:
        """Get standard classification parameters."""
        # Get precision options
        precision_options = self._get_precision_options()
        
        return [
            {
                "name": "Precision",
                "param_key": "precision",
                "type": "select",
                "options": precision_options,
                "description": "Model precision mode (requires model reload)",
                "group": "model_loading"
            },
            {
                "name": "Top K Classes",
                "param_key": "top_k",
                "type": "number",
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Number of top predictions to return",
                "group": "general"
            },
            {
                "name": "Show Confidence",
                "param_key": "add_confidence",
                "type": "checkbox",
                "description": "Include confidence percentages",
                "group": "general"
            },
            {
                "name": "Batch Size",
                "param_key": "batch_size",
                "type": "number",
                "min": 1,
                "max": 32,
                "step": 1,
                "description": "Number of images to process simultaneously (higher = faster but more VRAM)",
                "group": "advanced"
            }
        ]
    
    def _get_precision_options(self) -> list:
        """
        Get precision/quantization options based on model capabilities.
        
        Returns:
            List of precision options for the model
        """
        model_type = self.config.get('type')
        default_precision = self.config.get('default_precision', 'float32')
        
        # Check if model explicitly defines supported precisions
        supported_precisions = self.config.get('supported_precisions')
        if supported_precisions:
            # Use the explicit list from config
            precision_map = {
                "float32": {"label": "FP32 (Float32)", "value": "float32"},
                "float16": {"label": "FP16 (Float16)", "value": "float16"},
                "bfloat16": {"label": "BF16 (BFloat16)", "value": "bfloat16"},
                "4bit": {"label": "4-bit Quantization", "value": "4bit"},
                "8bit": {"label": "8-bit Quantization", "value": "8bit"},
                "onnx": {"label": "ONNX Native", "value": "onnx"},
            }
            return [precision_map[p] for p in supported_precisions if p in precision_map]
        
        # ONNX models only support their native precision
        if model_type == 'onnx_tagger':
            return [{"label": "ONNX Native", "value": "onnx"}]
        
        # Most HuggingFace models support these precisions
        options = [
            {"label": "FP32 (Float32)", "value": "float32"},
            {"label": "FP16 (Float16)", "value": "float16"},
            {"label": "BF16 (BFloat16)", "value": "bfloat16"},
        ]
        
        # VLM and OCR models typically support quantization
        if model_type in ['hf_vlm', 'hf_vlm_custom', 'hf_ocr', 'hf_ocr_trocr', 'hf_ocr_custom']:
            options.extend([
                {"label": "4-bit Quantization", "value": "4bit"},
                {"label": "8-bit Quantization", "value": "8bit"},
            ])
        
        # Tagger models might have more limited precision support
        # Some only work well with float32
        if model_type == 'hf_tagger':
            options.extend([
                {"label": "4-bit Quantization", "value": "4bit"},
                {"label": "8-bit Quantization", "value": "8bit"},
            ])
        
        # Classifier models
        if model_type == 'hf_classifier':
            # Usually just float precisions, no quantization
            pass
        
        return options
