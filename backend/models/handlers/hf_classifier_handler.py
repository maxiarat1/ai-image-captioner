"""
HuggingFace Classifier Handler
Handles classification models (ViT, etc.)
"""
import torch
from PIL import Image
from typing import Any, Dict, List, Optional
import logging
from transformers import AutoProcessor, AutoModelForImageClassification
from .base_handler import BaseModelHandler

logger = logging.getLogger(__name__)


class HuggingFaceClassifierHandler(BaseModelHandler):
    """Handler for HuggingFace image classification models."""
    
    def load(self, precision: str = None, use_flash_attention: bool = False, **kwargs) -> None:
        """Load HuggingFace classifier model."""
        try:
            from utils.torch_utils import pick_device
            self.device = pick_device(torch)
            
            precision = precision or self.config.get('default_precision', 'float32')
            logger.info("Loading %s on %s with %s precision…", self.model_key, self.device, precision)
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            
            # Prepare model loading kwargs
            model_kwargs = {}
            
            # Handle quantization
            quantization_config = self._create_quantization_config(precision)
            if quantization_config:
                model_kwargs['quantization_config'] = quantization_config
            else:
                model_kwargs['torch_dtype'] = self._get_dtype(precision)
            
            # Load model
            self.model = AutoModelForImageClassification.from_pretrained(
                self.model_id, **model_kwargs
            )
            
            if precision not in ["4bit", "8bit"]:
                self.model.to(self.device)
            
            self.model.eval()
            logger.info("%s loaded successfully", self.model_key)
            
        except Exception as e:
            logger.exception("Error loading %s: %s", self.model_key, e)
            raise
    
    def infer_single(self, image: Image.Image, prompt: Optional[str] = None, 
                    parameters: Optional[Dict] = None) -> str:
        """Classify a single image."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")
        
        try:
            image = self._ensure_rgb(image)
            
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Get predictions
            params = parameters or {}
            top_k = params.get('top_k', 5)
            add_confidence = params.get('add_confidence', True)
            
            # Get top-k predictions
            probs = torch.nn.functional.softmax(logits[0], dim=0)
            top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))
            
            # Format results
            results = []
            for prob, idx in zip(top_probs, top_indices):
                label = self.model.config.id2label[idx.item()]
                if add_confidence:
                    results.append(f"{label} ({prob.item():.2%})")
                else:
                    results.append(label)
            
            return ', '.join(results)
            
        except Exception as e:
            logger.exception("Error classifying image: %s", e)
            return f"Error: {str(e)}"
    
    def infer_batch(self, images: List[Image.Image], prompts: Optional[List[str]] = None,
                   parameters: Optional[Dict] = None) -> List[str]:
        """Classify multiple images."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")
        
        try:
            images = [self._ensure_rgb(img) for img in images]
            
            # Process batch
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Get predictions for each image
            params = parameters or {}
            top_k = params.get('top_k', 5)
            add_confidence = params.get('add_confidence', True)
            
            results = []
            for logit in logits:
                probs = torch.nn.functional.softmax(logit, dim=0)
                top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))
                
                image_results = []
                for prob, idx in zip(top_probs, top_indices):
                    label = self.model.config.id2label[idx.item()]
                    if add_confidence:
                        image_results.append(f"{label} ({prob.item():.2%})")
                    else:
                        image_results.append(label)
                
                results.append(', '.join(image_results))
            
            return results
            
        except Exception as e:
            logger.exception("Error in batch classification: %s", e)
            error_msg = f"Error: {str(e)}"
            return [error_msg] * len(images)
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None and self.processor is not None
    
    # Helper methods
    
    def _ensure_rgb(self, image):
        """Convert image to RGB if needed."""
        if isinstance(image, list):
            return [img.convert('RGB') if img.mode != 'RGB' else img for img in image]
        return image.convert('RGB') if image.mode != 'RGB' else image
    
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
