import csv
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from .base_adapter import BaseModelAdapter
import logging

logger = logging.getLogger(__name__)

class Wd14ConvNextAdapter(BaseModelAdapter):
    """WD v1.4 ConvNext Tagger v2 - Fast ONNX-based anime tagging"""

    # Parameters that should not be passed to model inference
    SPECIAL_PARAMS = {'batch_size'}

    def __init__(self, model_id="SmilingWolf/wd-v1-4-convnext-tagger-v2"):
        super().__init__(model_id)
        self.session = None
        self.input_name = None
        self.tag_names = []
        self.image_size = 448  # WD14 default
        self.repo_id = model_id

    def load_model(self) -> None:
        """Load WD14 ConvNext model and tags using ONNX Runtime"""
        try:
            logger.info("Loading %s model (ONNX)…", self.model_name)

            # Download model and tags from HuggingFace (uses default cache)
            logger.info("Downloading model.onnx from HuggingFace…")
            model_path = hf_hub_download(repo_id=self.repo_id, filename="model.onnx")

            logger.info("Downloading selected_tags.csv from HuggingFace…")
            tags_path = hf_hub_download(repo_id=self.repo_id, filename="selected_tags.csv")

            # Determine input tensor name based on model variant
            self.input_name = "input" if ("v3" in self.repo_id or "swinv2" in self.repo_id) else "input_1:0"

            # Initialize ONNX Runtime session with CUDA/CPU fallback
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            try:
                self.session = ort.InferenceSession(model_path, providers=providers)
                logger.info("ONNX Runtime using: %s", self.session.get_providers()[0])
            except Exception:
                logger.warning("CUDA provider failed, falling back to CPU")
                self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

            # Load tags from CSV
            with open(tags_path, encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                self.tag_names = [row[1] for row in reader]

            logger.info("Loaded %d tags", len(self.tag_names))
            logger.info("%s model loaded successfully", self.model_name)

        except Exception as e:
            logger.exception("Error loading %s model: %s", self.model_name, e)
            raise

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """WD14-style preprocessing: BGR, pad to square, resize to 448x448"""
        # Convert PIL Image to numpy array
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

        # Convert to float32 (ONNX expects float input)
        image = image.astype(np.float32)

        return image  # shape: (H, W, C)

    def _process_predictions(self, probs: np.ndarray, parameters: dict) -> str:
        """Process prediction probabilities into formatted tag string"""
        # Parse parameters with defaults (matching test2.py working values)
        params = parameters or {}
        threshold = params.get('threshold', 0.35)
        top_n = params.get('top_n', 50)
        replace_underscores = params.get('replace_underscores', True)
        add_confidence = params.get('add_confidence', False)
        tag_separator = params.get('tag_separator', ', ')
        exclude_tags = params.get('exclude_tags', '')

        # Parse exclude_tags into a set
        excluded_set = set()
        if exclude_tags and exclude_tags.strip():
            excluded_set = {tag.strip().lower() for tag in exclude_tags.split(',')}

        # Get top predictions
        top_indices = probs.argsort()[-top_n:][::-1]

        # Build tag list
        tags = []
        for idx in top_indices:
            confidence = float(probs[idx])

            # Apply threshold
            if confidence <= threshold:
                continue

            tag_name = self.tag_names[idx]

            # Apply exclude filter
            if tag_name.lower() in excluded_set:
                continue

            # Format tag
            if replace_underscores:
                tag_name = tag_name.replace('_', ' ')

            if add_confidence:
                tags.append(f"{tag_name}:{confidence:.2f}")
            else:
                tags.append(tag_name)

        # Join with separator
        result = tag_separator.join(tags)
        return result if result else "No tags detected above threshold"

    def generate_caption(self, image: Image.Image, prompt: str = None, parameters: dict = None) -> str:
        """
        Generate tags for the given image.
        Note: This model generates tags, not captions. The 'prompt' parameter is ignored.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Convert to RGB if necessary
            image = self._ensure_rgb(image)

            # Preprocess image
            x = self._preprocess_image(image)
            x = np.expand_dims(x, 0)  # Add batch dimension: (1, H, W, C)

            # Run inference
            probs = self.session.run(None, {self.input_name: x})[0][0]

            # Process predictions into tag string
            return self._process_predictions(probs, parameters)

        except Exception as e:
            logger.exception("Error generating tags: %s", e)
            return f"Error: {str(e)}"

    def generate_captions_batch(self, images: list, prompts: list = None, parameters: dict = None) -> list:
        """Generate tags for multiple images using batch processing"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Convert all images to RGB
            processed_images = [self._ensure_rgb(img) for img in images]

            # Preprocess all images
            batch_array = np.stack([self._preprocess_image(img) for img in processed_images])

            # Run batch inference
            probs_batch = self.session.run(None, {self.input_name: batch_array})[0]

            # Process each image's results
            results = [self._process_predictions(probs, parameters) for probs in probs_batch]

            return results

        except Exception as e:
            logger.exception("Error generating tags in batch: %s", e)
            return self._format_batch_error(e, len(images))

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return (self.session is not None and
                self.input_name is not None and
                len(self.tag_names) > 0)

    def get_available_parameters(self) -> list:
        """Get list of available parameters for WD14 ConvNext model"""
        return [
            {
                "name": "Threshold",
                "param_key": "threshold",
                "type": "number",
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "description": "Minimum confidence score for tags to be included (default: 0.35)"
            },
            {
                "name": "Top N",
                "param_key": "top_n",
                "type": "number",
                "min": 1,
                "max": 200,
                "step": 1,
                "description": "Maximum number of top-scoring tags to consider (default: 50)"
            },
            {
                "name": "Replace Underscores",
                "param_key": "replace_underscores",
                "type": "checkbox",
                "description": "Replace underscores with spaces in tag names (default: enabled)"
            },
            {
                "name": "Add Confidence",
                "param_key": "add_confidence",
                "type": "checkbox",
                "description": "Add confidence scores to each tag in the output (e.g., 'tag:0.85')"
            },
            {
                "name": "Tag Separator",
                "param_key": "tag_separator",
                "type": "text",
                "description": "String to use for separating tags in output (default: ', ')"
            },
            {
                "name": "Exclude Tags",
                "param_key": "exclude_tags",
                "type": "text",
                "description": "Comma-separated list of tags to exclude from output"
            },
            {
                "name": "Batch Size",
                "param_key": "batch_size",
                "type": "number",
                "min": 1,
                "max": 32,
                "step": 1,
                "description": "Number of images to process simultaneously (higher = faster but more memory)"
            }
        ]

    def unload(self) -> None:
        """Unload model and clear resources"""
        if hasattr(self, 'session') and self.session is not None:
            logger.info("Unloading %s model…", self.model_name)
            self.session = None

        if hasattr(self, 'tag_names'):
            self.tag_names = []

        if hasattr(self, 'input_name'):
            self.input_name = None

        # Note: ONNX Runtime doesn't use CUDA cache like PyTorch
        import gc
        gc.collect()
