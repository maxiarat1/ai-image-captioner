import torch
from utils.torch_utils import pick_device
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from .base_adapter import BaseModelAdapter
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class WdVitAdapter(BaseModelAdapter):
    """WD Tagger v3 adapter for anime-style image tagging (supports ViT and EVA02 variants)"""

    def __init__(self, model_id="SmilingWolf/wd-vit-large-tagger-v3"):
        super().__init__(model_id)
        self.device = pick_device(torch)
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.id2tag = None
        # Determine tags CSV URL based on model variant
        if "eva02" in model_id:
            self.tags_csv_url = "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/raw/main/selected_tags.csv"
        else:
            self.tags_csv_url = "https://huggingface.co/SmilingWolf/wd-vit-large-tagger-v3/raw/main/selected_tags.csv"

    def load_model(self) -> None:
        """Load WD Tagger model, processor, and tags CSV"""
        try:
            logger.info("Loading %s tagger model on %s…", self.model_name, self.device)

            # Load tags CSV
            logger.info("Loading tags from CSV…")
            tags_df = pd.read_csv(self.tags_csv_url)
            self.id2tag = dict(zip(tags_df.index, tags_df["name"]))
            logger.info("Loaded %d tags", len(self.id2tag))

            # Load processor
            logger.info("Loading image processor…")
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Load model
            logger.info("Loading model…")
            self.model = AutoModelForImageClassification.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=self.dtype
            )
            self.model.to(self.device)
            self.model.eval()

            logger.info("%s tagger model loaded successfully", self.model_name)

        except Exception as e:
            logger.exception("Error loading %s model: %s", self.model_name, e)
            raise

    def generate_caption(self, image: Image.Image, prompt: str = None, parameters: dict = None) -> str:
        """
        Generate tags for the given image.
        Note: This model generates tags, not captions. The 'prompt' parameter is ignored.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Debug logging
            logger.debug("WdVit.generate_caption | params=%s", parameters)

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Get threshold parameter (default 0.3)
            threshold = 0.3
            if parameters and 'threshold' in parameters:
                threshold = parameters['threshold']

            logger.debug("Using threshold: %.2f", threshold)

            # Process image
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            # Run inference
            with torch.no_grad():
                probs = torch.sigmoid(self.model(**inputs).logits[0]).cpu()

            # Filter tags by threshold and sort by confidence
            tags = {
                self.id2tag[i]: probs[i].item()
                for i in range(len(probs))
                if i in self.id2tag and probs[i] > threshold
            }
            tags = dict(sorted(tags.items(), key=lambda x: x[1], reverse=True))

            # Return comma-separated tags
            result = ", ".join(tags.keys())
            logger.debug("Generated %d tags", len(tags))

            return result if result else "No tags detected above threshold"

        except Exception as e:
            logger.exception("Error generating tags: %s", e)
            return f"Error: {str(e)}"

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return (self.model is not None and
                self.processor is not None and
                self.id2tag is not None)

    def get_available_parameters(self) -> list:
        """Get list of available parameters for WD-ViT model"""
        return [
            {
                "name": "Threshold",
                "param_key": "threshold",
                "type": "number",
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "description": "Minimum confidence score for tags to be included (0.0-1.0)"
            }
        ]

    def unload(self) -> None:
        """Unload model and clear tags dictionary"""
        if hasattr(self, 'id2tag'):
            self.id2tag = None
        super().unload()
