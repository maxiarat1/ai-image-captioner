import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageClassification
from .base_adapter import BaseModelAdapter
import logging

logger = logging.getLogger(__name__)

class VitClassifierAdapter(BaseModelAdapter):
    """Google ViT Image Classification adapter - ImageNet-based classification"""

    # Parameters that should not be passed to model inference
    SPECIAL_PARAMS = {'batch_size'}

    def __init__(self, model_id="google/vit-base-patch16-224"):
        super().__init__(model_id)
        self.device = self._init_device(torch)

    def load_model(self) -> None:
        """Load Google ViT model and processor"""
        try:
            logger.info("Loading %s model on %s…", self.model_name, self.device)

            # Load processor
            logger.info("Loading image processor…")
            self.processor = AutoProcessor.from_pretrained(self.model_name)

            # Load model
            logger.info("Loading model…")
            self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

            logger.info("%s model loaded successfully", self.model_name)

        except Exception as e:
            logger.exception("Error loading %s model: %s", self.model_name, e)
            raise

    def _process_predictions(self, logits: torch.Tensor, parameters: dict) -> str:
        """Process logits into formatted classification output"""
        # Parse parameters with defaults
        params = parameters or {}
        top_n = params.get('top_n', 5)
        threshold = params.get('threshold', 0.0)
        add_confidence = params.get('add_confidence', True)
        tag_separator = params.get('tag_separator', ', ')

        # Get probabilities
        probs = logits.softmax(dim=-1)

        # Get top predictions
        top_probs, top_indices = probs.topk(top_n)

        # Build results
        results = []
        for prob, idx in zip(top_probs, top_indices):
            prob_val = prob.item()
            idx_val = idx.item()

            # Apply threshold
            if prob_val <= threshold:
                continue

            # Get label from model config
            label = self.model.config.id2label[idx_val]

            # Format output
            if add_confidence:
                results.append(f"{label}:{prob_val:.3f}")
            else:
                results.append(label)

        # Join with separator
        result = tag_separator.join(results)
        return result if result else "No classifications above threshold"

    def generate_caption(self, image: Image.Image, prompt: str = None, parameters: dict = None) -> str:
        """
        Generate classification labels for the given image.
        Note: This model generates ImageNet class labels. The 'prompt' parameter is ignored.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Convert to RGB if necessary
            image = self._ensure_rgb(image)

            # Process image
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0]  # Single image

            # Process predictions into formatted string
            return self._process_predictions(logits, parameters)

        except Exception as e:
            logger.exception("Error generating classification: %s", e)
            return f"Error: {str(e)}"

    def generate_captions_batch(self, images: list, prompts: list = None, parameters: dict = None) -> list:
        """Generate classifications for multiple images using batch processing"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Convert all images to RGB
            processed_images = self._ensure_rgb(images)

            # Process all images in batch
            inputs = self.processor(images=processed_images, return_tensors="pt", padding=True).to(self.device)

            # Run batch inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Process each image's results
            results = [self._process_predictions(img_logits, parameters) for img_logits in logits]

            return results

        except Exception as e:
            logger.exception("Error generating classifications in batch: %s", e)
            return self._format_batch_error(e, len(images))

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None and self.processor is not None

    def get_available_parameters(self) -> list:
        """Get list of available parameters for ViT Classifier"""
        return [
            {
                "name": "Top N",
                "param_key": "top_n",
                "type": "number",
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Number of top predictions to return (default: 5)"
            },
            {
                "name": "Threshold",
                "param_key": "threshold",
                "type": "number",
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "description": "Minimum confidence score for predictions to be included (default: 0.0)"
            },
            {
                "name": "Add Confidence",
                "param_key": "add_confidence",
                "type": "checkbox",
                "description": "Add confidence scores to each prediction (e.g., 'label:0.850')"
            },
            {
                "name": "Tag Separator",
                "param_key": "tag_separator",
                "type": "text",
                "description": "String to use for separating predictions in output (default: ', ')"
            },
            {
                "name": "Batch Size",
                "param_key": "batch_size",
                "type": "number",
                "min": 1,
                "max": 32,
                "step": 1,
                "description": "Number of images to process simultaneously (higher = faster but more VRAM)"
            }
        ]
