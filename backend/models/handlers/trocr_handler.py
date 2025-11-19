"""
TrOCR Model Handler
Custom handler for TrOCR with doctr text detection support.

Two-stage OCR approach:
1. Word-level text detection using doctr's ocr_predictor
2. Text recognition using TrOCR on detected word regions
3. Intelligent line reconstruction by grouping words based on vertical proximity
"""
import torch
from PIL import Image
from typing import Any, Dict, List, Optional
import logging
from .hf_ocr_handler import HuggingFaceOCRHandler

logger = logging.getLogger(__name__)


class TrOCRHandler(HuggingFaceOCRHandler):
    """
    Custom handler for TrOCR with doctr text detection support.

    Two-stage OCR approach:
    1. Word-level text detection using doctr's ocr_predictor
    2. Text recognition using TrOCR on detected word regions
    3. Intelligent line reconstruction by grouping words based on vertical proximity
    """

    # Line grouping defaults
    DEFAULT_LINE_THRESHOLD = 20  # Maximum vertical distance (pixels) to group words into same line
    MIN_CROP_SIZE = 5  # Minimum crop size to skip very small regions

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.detector = None

    def _pre_load_hook(self, precision: str = None, **kwargs) -> str:
        """Setup precision and load doctr detector if needed."""
        precision = precision or self.config.get('default_precision', 'float32')

        # Load doctr text detector
        if self.config.get('requires_text_detection'):
            logger.info("Loading doctr text detector...")
            from doctr.models import ocr_predictor
            self.detector = ocr_predictor(pretrained=True)

        return precision

    def _load_processor(self, **kwargs):
        """Load TrOCR processor with use_fast option."""
        from transformers import TrOCRProcessor

        use_fast = kwargs.get('use_fast', True)
        processor_config = self.config.get('processor_config', {})
        processor_config['use_fast'] = use_fast
        self.processor = TrOCRProcessor.from_pretrained(self.model_id, **processor_config)

    def _load_model(self, model_kwargs: dict, precision: str):
        """Load VisionEncoderDecoderModel for TrOCR."""
        from transformers import VisionEncoderDecoderModel

        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_id, **model_kwargs)

    def _detect_text_regions(self, image: Image.Image) -> List[tuple]:
        """
        Detect text regions at word-level using doctr.
        Returns list of bounding boxes in absolute pixel coordinates (x1, y1, x2, y2).
        """
        import tempfile
        import os
        from doctr.io import DocumentFile

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
            image.save(tmp_path, format='PNG')

        try:
            doc = DocumentFile.from_images(tmp_path)
            result = self.detector(doc)
            exported = result.export()

            if not exported.get('pages'):
                return []

            page = exported['pages'][0]
            width, height = image.size
            boxes = []

            # Extract word-level bounding boxes
            for block in page.get('blocks', []):
                for line in block.get('lines', []):
                    for word in line.get('words', []):
                        ((x1, y1), (x2, y2)) = word['geometry']
                        # Convert normalized coordinates to absolute pixels
                        x1, y1, x2, y2 = (
                            int(x1 * width), int(y1 * height),
                            int(x2 * width), int(y2 * height)
                        )
                        boxes.append((x1, y1, x2, y2))

            return boxes
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _group_words_into_lines(self, word_results: List[Dict], line_threshold: int = None) -> List[str]:
        """
        Group recognized words into lines based on vertical proximity.

        Args:
            word_results: List of dicts with 'bbox' (x1, y1, x2, y2) and 'text'
            line_threshold: Maximum vertical distance (pixels) to group words into same line

        Returns:
            List of strings, each representing a line of text
        """
        if not word_results:
            return []

        if line_threshold is None:
            line_threshold = self.DEFAULT_LINE_THRESHOLD

        # Sort words by top-to-bottom first, then left-to-right
        word_results.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))

        lines = []
        current_line = []

        for word_result in word_results:
            if not current_line:
                current_line.append(word_result)
                continue

            prev_y = current_line[-1]["bbox"][1]  # y1 coordinate of previous word
            curr_y = word_result["bbox"][1]  # y1 coordinate of current word

            if abs(curr_y - prev_y) < line_threshold:
                # Same line
                current_line.append(word_result)
            else:
                # New line - sort current line left-to-right and save
                current_line.sort(key=lambda w: w["bbox"][0])
                line_text = " ".join(w["text"] for w in current_line if w["text"])
                if line_text:
                    lines.append(line_text)
                current_line = [word_result]

        # Handle last line
        if current_line:
            current_line.sort(key=lambda w: w["bbox"][0])
            line_text = " ".join(w["text"] for w in current_line if w["text"])
            if line_text:
                lines.append(line_text)

        return lines

    def _recognize_text_from_crop(self, crop: Image.Image, parameters: Optional[Dict] = None) -> str:
        """
        Recognize text from a cropped image region using TrOCR.
        """
        pixel_values = self.processor(images=crop, return_tensors="pt").pixel_values
        model_dtype = next(self.model.parameters()).dtype
        pixel_values = pixel_values.to(device=self.device, dtype=model_dtype)

        gen_params = self._filter_generation_params(parameters)

        # Ensure dtype alignment and safe inference
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=model_dtype):
            generated_ids = self.model.generate(pixel_values, **gen_params)

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()

    def infer_single(self, image: Image.Image, prompt: Optional[str] = None,
                    parameters: Optional[Dict] = None) -> str:
        """Extract text using TrOCR with doctr text detection."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")

        try:
            image = self._ensure_rgb(image)

            # Extract OCR-specific parameters
            params = parameters or {}
            line_separator = params.get("line_separator", "\\n")
            # Handle escaped newline from frontend
            if line_separator == "\\n":
                line_separator = "\n"
            line_threshold = params.get("line_threshold", 20)

            # If detector is available, use two-stage OCR
            if self.detector:
                # Detect word-level bounding boxes
                boxes = self._detect_text_regions(image)
                num_boxes = len(boxes)
                logger.info("TrOCR detected %d word box(es) in image", num_boxes)

                if num_boxes == 0:
                    logger.info("No text regions detected in image")
                    return ""

                # Recognize text from each word box
                word_results = []
                for (x1, y1, x2, y2) in boxes:
                    # Skip very small crops
                    if (x2 - x1) < self.MIN_CROP_SIZE or (y2 - y1) < self.MIN_CROP_SIZE:
                        continue

                    crop = image.crop((x1, y1, x2, y2))
                    text = self._recognize_text_from_crop(crop, parameters)
                    if text:
                        word_results.append({"bbox": (x1, y1, x2, y2), "text": text})

                logger.info("Successfully recognized %d words", len(word_results))

                # If no readable text was recognized, return empty
                if not word_results:
                    logger.info("No readable text recognized in detected regions")
                    return ""

                # Group words into lines based on vertical proximity
                lines = self._group_words_into_lines(word_results, line_threshold)

                # Apply final line separator
                final_text = line_separator.join(lines)
                logger.info("Total lines reconstructed: %d", len(lines))
                return final_text if final_text else ""

            else:
                # Fallback: process whole image without detection
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

    def is_loaded(self) -> bool:
        """Check if model and processor are loaded (detector is optional)."""
        return self.model is not None and self.processor is not None
