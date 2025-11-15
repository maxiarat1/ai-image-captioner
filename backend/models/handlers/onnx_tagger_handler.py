"""
ONNX Tagger Handler
Handles ONNX-based tagging models (WD14 ConvNext, etc.)
"""
import csv
import numpy as np
import cv2
from PIL import Image
from typing import Any, Dict, List, Optional
import logging
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from .base_handler import BaseModelHandler

logger = logging.getLogger(__name__)


class ONNXTaggerHandler(BaseModelHandler):
    """Handler for ONNX-based tagging models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.session = None
        self.tag_names = []
        self.image_size = config.get('image_size', 448)
        self.input_name = config.get('input_tensor_name', 'input_1:0')
    
    def load(self, precision: str = None, use_flash_attention: bool = False, **kwargs) -> None:
        """Load ONNX tagger model."""
        try:
            logger.info("Loading %s model (ONNX)…", self.model_key)
            
            # Download model and tags from HuggingFace
            model_path = hf_hub_download(repo_id=self.model_id, filename="model.onnx")
            tags_path = hf_hub_download(repo_id=self.model_id, filename="selected_tags.csv")
            
            # Initialize ONNX Runtime with CUDA/CPU fallback
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            try:
                self.session = ort.InferenceSession(model_path, providers=providers)
                logger.info("ONNX Runtime using: %s", self.session.get_providers()[0])
            except Exception:
                logger.warning("CUDA provider failed, falling back to CPU")
                self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            
            # Load tags
            with open(tags_path, encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                self.tag_names = [row[1] for row in reader]
            
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
            # Preprocess image
            input_array = self._preprocess_image(image)
            
            # Run ONNX inference
            probs = self.session.run(None, {self.input_name: input_array})[0][0]
            
            # Format tags
            return self._format_tags(probs, parameters)
            
        except Exception as e:
            logger.exception("Error generating tags: %s", e)
            return f"Error: {str(e)}"
    
    def infer_batch(self, images: List[Image.Image], prompts: Optional[List[str]] = None,
                   parameters: Optional[Dict] = None) -> List[str]:
        """Generate tags for multiple images."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")
        
        try:
            # Preprocess all images
            input_arrays = [self._preprocess_image(img) for img in images]
            batch_input = np.concatenate(input_arrays, axis=0)
            
            # Run ONNX inference
            probs_batch = self.session.run(None, {self.input_name: batch_input})[0]
            
            # Format tags for each image
            return [self._format_tags(probs, parameters) for probs in probs_batch]
            
        except Exception as e:
            logger.exception("Error in batch tagging: %s", e)
            error_msg = f"Error: {str(e)}"
            return [error_msg] * len(images)
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.session is not None
    
    def unload(self) -> None:
        """Unload ONNX model."""
        if self.session is not None:
            logger.info("Unloading %s model…", self.model_key)
            self.session = None
    
    # Helper methods
    
    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """WD14-style preprocessing: BGR, pad to square, resize."""
        # Convert to numpy array
        image = np.array(image)
        
        # RGB → BGR
        image = image[:, :, ::-1]
        
        # Pad to square with white background
        size = max(image.shape[0:2])
        pad_x = size - image.shape[1]
        pad_y = size - image.shape[0]
        pad_l = pad_x // 2
        pad_t = pad_y // 2
        image = np.pad(
            image,
            ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)),
            mode="constant",
            constant_values=255,
        )
        
        # Resize to model input size
        interp = cv2.INTER_AREA if size > self.image_size else cv2.INTER_LANCZOS4
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=interp)
        
        # Convert to float32 and add batch dimension
        image = image.astype(np.float32)
        return np.expand_dims(image, axis=0)
    
    def _format_tags(self, probs: np.ndarray, parameters: Optional[Dict]) -> str:
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
