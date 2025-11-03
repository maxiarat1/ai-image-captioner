import logging
from typing import List

import torch
from PIL import Image
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from .base_adapter import BaseModelAdapter

logger = logging.getLogger(__name__)


class TrOCRAdapter(BaseModelAdapter):
    """
    Adapter for Microsoft TrOCR model with doctr text detection.

    Two-stage OCR approach:
    1. Word-level text detection using doctr's ocr_predictor
    2. Text recognition using TrOCR on detected word regions
    3. Intelligent line reconstruction by grouping words based on vertical proximity
    """

    SPECIAL_PARAMS = {"batch_size", "use_fast", "line_separator", "sort_boxes", "line_threshold"}

    def __init__(self, model_id: str = "microsoft/trocr-large-printed"):
        super().__init__(model_id)
        self.model_id = model_id
        self.device = self._init_device(torch)
        self.detector = None

    def load_model(self, use_fast: bool = False) -> None:
        try:
            logger.info(
                "Loading TrOCR model %s on %s with float16 precision…",
                self.model_id,
                self.device,
            )

            # Load doctr text detector
            logger.info("Loading doctr text detector...")
            self.detector = ocr_predictor(pretrained=True)

            # Load TrOCR processor
            self.processor = TrOCRProcessor.from_pretrained(self.model_id, use_fast=use_fast)
            self._setup_pad_token()

            # Load TrOCR model with float16
            self.model = VisionEncoderDecoderModel.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
            )

            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info("TrOCR model loaded successfully")

        except Exception as e:
            logger.exception("Error loading TrOCR model %s: %s", self.model_id, e)
            raise

    def _detect_text_regions(self, image: Image.Image) -> List[tuple]:
        """
        Detect text regions at word-level using doctr.
        Returns list of bounding boxes in absolute pixel coordinates.
        """
        import tempfile, os
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

    def _sort_boxes(self, boxes: List[tuple], sort_order: str) -> List[tuple]:
        """
        Sort text region boxes according to the specified reading order.

        Args:
            boxes: List of (x1, y1, x2, y2) tuples in absolute pixel coordinates
            sort_order: One of "top-to-bottom", "left-to-right", or "none"

        Returns:
            Sorted list of boxes
        """
        if sort_order == "top-to-bottom":
            # Sort by y1 (top edge), then by x1 (left edge) for ties
            return sorted(boxes, key=lambda box: (box[1], box[0]))
        elif sort_order == "left-to-right":
            # Sort by x1 (left edge), then by y1 (top edge) for ties
            return sorted(boxes, key=lambda box: (box[0], box[1]))
        else:
            # "none" - return as-is (detection order)
            return boxes

    def _group_words_into_lines(self, word_results: List[dict], line_threshold: int = 20) -> List[str]:
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

    def _recognize_text_from_crop(self, crop: Image.Image, parameters: dict = None) -> str:
        """
        Recognize text from a cropped image region using TrOCR.
        """
        pixel_values = self.processor(images=crop, return_tensors="pt").pixel_values
        model_dtype = next(self.model.parameters()).dtype
        pixel_values = pixel_values.to(device=self.device, dtype=model_dtype)

        gen_params = self._filter_generation_params(parameters, self.SPECIAL_PARAMS)
        gen_params = self._sanitize_generation_params(gen_params)

        # Ensure dtype alignment and safe inference
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=model_dtype):
            generated_ids = self.model.generate(pixel_values, **gen_params)

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()

    def generate_caption(self, image: Image.Image, _prompt: str = None, parameters: dict = None) -> str:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            image = self._ensure_rgb(image)

            # Extract OCR-specific parameters
            line_separator = parameters.get("line_separator", "\\n") if parameters else "\\n"
            # Handle escaped newline from frontend
            if line_separator == "\\n":
                line_separator = "\n"
            sort_boxes = parameters.get("sort_boxes", "none") if parameters else "none"
            line_threshold = parameters.get("line_threshold", 20) if parameters else 20

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
                if (x2 - x1) < 5 or (y2 - y1) < 5:
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

        except Exception as e:
            logger.exception("Error generating OCR with TrOCR: %s", e)
            return f"Error: {str(e)}"

    def generate_captions_batch(
        self, images: List[Image.Image], _prompts: List[str] = None, parameters: dict = None
    ) -> List[str]:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        try:
            return [self.generate_caption(img, None, parameters) for img in images]
        except Exception as e:
            logger.exception("Error generating batch OCR with TrOCR: %s", e)
            return self._format_batch_error(e, len(images))

    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None and self.detector is not None

    def get_available_parameters(self) -> list:
        return [
            {
                "name": "Use Fast Processor",
                "param_key": "use_fast",
                "type": "checkbox",
                "description": "Use fast tokenizer (requires model reload)"
            },
            {
                "name": "Do Sample",
                "param_key": "do_sample",
                "type": "checkbox",
                "description": "Enable sampling mode (default: disabled for deterministic OCR). Required for temperature/top_p.",
                "group": "sampling"
            },
            {
                "name": "Temperature",
                "param_key": "temperature",
                "type": "number",
                "min": 0.1,
                "max": 2,
                "step": 0.1,
                "description": "Sampling temperature (requires do_sample=true)",
                "depends_on": "do_sample",
                "group": "sampling"
            },
            {
                "name": "Top P",
                "param_key": "top_p",
                "type": "number",
                "min": 0,
                "max": 1,
                "step": 0.01,
                "description": "Nucleus sampling threshold (requires do_sample=true)",
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
                "description": "Number of beams for beam search (conflicts with do_sample=true)",
                "group": "beam_search"
            },
            {
                "name": "Length Penalty",
                "param_key": "length_penalty",
                "type": "number",
                "min": 0,
                "max": 2,
                "step": 0.1,
                "description": "Length penalty for beam search (requires num_beams>1)",
                "depends_on": "num_beams",
                "group": "beam_search"
            },
            {
                "name": "Early Stopping",
                "param_key": "early_stopping",
                "type": "checkbox",
                "description": "Stop beam search when all beams finish (requires num_beams>1)",
                "depends_on": "num_beams",
                "group": "beam_search"
            },
            {
                "name": "Max New Tokens",
                "param_key": "max_new_tokens",
                "type": "number",
                "min": 1,
                "max": 512,
                "step": 1,
                "description": "Maximum tokens per text region (default: 256)",
                "group": "general"
            },
            {
                "name": "Min New Tokens",
                "param_key": "min_new_tokens",
                "type": "number",
                "min": 0,
                "max": 100,
                "step": 1,
                "description": "Minimum number of tokens to generate per region",
                "group": "general"
            },
            {
                "name": "Repetition Penalty",
                "param_key": "repetition_penalty",
                "type": "number",
                "min": 1.0,
                "max": 2.0,
                "step": 0.1,
                "description": "Penalty for repeating tokens (works in both modes)",
                "group": "general"
            },
            {
                "name": "No Repeat N-gram Size",
                "param_key": "no_repeat_ngram_size",
                "type": "number",
                "min": 0,
                "max": 5,
                "step": 1,
                "description": "Prevent repeating n-grams of this size (works in both modes)",
                "group": "general"
            },
            {
                "name": "Batch Size",
                "param_key": "batch_size",
                "type": "number",
                "min": 1,
                "max": 8,
                "step": 1,
                "description": "Number of images to process simultaneously"
            },
            {
                "name": "Line Separator",
                "param_key": "line_separator",
                "type": "select",
                "options": [
                    {"value": "\\n", "label": "Newline (\\n)"},
                    {"value": " ", "label": "Space ( )"},
                    {"value": ", ", "label": "Comma-Space (, )"},
                    {"value": " | ", "label": "Pipe ( | )"},
                ],
                "description": "Character(s) to join detected text lines"
            },
            {
                "name": "Line Threshold",
                "param_key": "line_threshold",
                "type": "number",
                "min": 5,
                "max": 100,
                "step": 5,
                "description": "Maximum vertical distance (pixels) to group words into same line (default: 20)"
            },
            {
                "name": "Sort Reading Order",
                "param_key": "sort_boxes",
                "type": "select",
                "options": [
                    {"value": "none", "label": "None (detection order)"},
                    {"value": "top-to-bottom", "label": "Top to Bottom"},
                    {"value": "left-to-right", "label": "Left to Right"},
                ],
                "description": "Order for reading text regions"
            },
        ]

    def unload(self) -> None:
        if hasattr(self, "detector") and self.detector is not None:
            logger.info("Unloading doctr detector...")
            del self.detector
            self.detector = None
        super().unload()
