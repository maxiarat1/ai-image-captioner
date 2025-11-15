"""
HuggingFace Tagger Handler
Handles anime/art tagging models (WD-ViT, WD-EVA02, etc.)
"""
import torch
import csv
from PIL import Image
from typing import Any, Dict, List, Optional
import logging
from transformers import AutoImageProcessor, AutoModelForImageClassification
from huggingface_hub import hf_hub_download
from .base_handler import BaseModelHandler

logger = logging.getLogger(__name__)


class HuggingFaceTaggerHandler(BaseModelHandler):
    """Handler for HuggingFace-based tagging models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.tag_names = []
        self.rating_tags = []
        self.general_tags = []
        self.character_tags = []
    
    def load(self, precision: str = None, use_flash_attention: bool = False, **kwargs) -> None:
        """Load HuggingFace tagger model."""
        try:
            from utils.torch_utils import pick_device
            self.device = pick_device(torch)
            
            precision = precision or self.config.get('default_precision', 'float32')
            logger.info("Loading %s on %s with %s precision…", self.model_key, self.device, precision)
            
            # Load tags CSV if required
            if self.config.get('requires_tags_csv'):
                self._load_tags()
            
            # Load processor
            self.processor = AutoImageProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            
            # Prepare model loading kwargs
            model_kwargs = {'trust_remote_code': True}
            
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
            logger.info("%s loaded successfully with %d tags", self.model_key, len(self.tag_names))
            
        except Exception as e:
            logger.exception("Error loading %s: %s", self.model_key, e)
            raise
    
    def infer_single(self, image: Image.Image, prompt: Optional[str] = None, 
                    parameters: Optional[Dict] = None) -> str:
        """Generate tags for a single image."""
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
                probs = torch.sigmoid(outputs.logits[0]).cpu().numpy()
            
            # Format tags
            return self._format_tags(probs, parameters)
            
        except Exception as e:
            logger.exception("Error generating tags: %s", e)
            return f"Error: {str(e)}"
    
    def infer_batch(self, images: List[Image.Image], prompts: Optional[List[str]] = None,
                   parameters: Optional[Dict] = None) -> List[str]:
        """Generate tags for multiple images using batch processing."""
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
                probs = torch.sigmoid(outputs.logits).cpu().numpy()
            
            # Format tags for each image
            return [self._format_tags(prob, parameters) for prob in probs]
            
        except Exception as e:
            logger.exception("Error in batch tagging: %s", e)
            error_msg = f"Error: {str(e)}"
            return [error_msg] * len(images)
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None and self.processor is not None
    
    # Helper methods
    
    def _load_tags(self):
        """Load tag names from CSV file."""
        try:
            tags_path = hf_hub_download(repo_id=self.model_id, filename="selected_tags.csv")
            
            with open(tags_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tag_name = row['name']
                    tag_category = row.get('category', '0')
                    
                    self.tag_names.append(tag_name)
                    
                    # Categorize tags
                    if tag_category == '9':  # Rating
                        self.rating_tags.append(tag_name)
                    elif tag_category == '4':  # Character
                        self.character_tags.append(tag_name)
                    else:  # General
                        self.general_tags.append(tag_name)
            
            logger.info("Loaded %d tags (%d general, %d character, %d rating)",
                       len(self.tag_names), len(self.general_tags),
                       len(self.character_tags), len(self.rating_tags))
        except Exception as e:
            logger.error("Error loading tags: %s", e)
            raise
    
    def _format_tags(self, probs, parameters: Optional[Dict]) -> str:
        """Format prediction probabilities into tag string."""
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
