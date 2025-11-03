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
    1. Text region detection using doctr's ocr_predictor
    2. Text recognition using TrOCR on detected regions
    """

    SPECIAL_PARAMS = {"batch_size", "use_fast", "line_separator", "sort_boxes"}

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
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
            image.save(tmp_path, format='PNG')
        try:
            doc = DocumentFile.from_images(tmp_path)
            result = self.detector(doc)
            if not result.pages:
                return []
            page = result.pages[0]
            boxes = []
            for block in page.blocks:
                for line in block.lines:
                    (x_min, y_min), (x_max, y_max) = line.geometry
                    boxes.append((x_min, y_min, x_max, y_max))
            return boxes
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _sort_boxes(self, boxes: List[tuple], sort_order: str) -> List[tuple]:
        """
        Sort text region boxes according to the specified reading order.

        Args:
            boxes: List of (x_min, y_min, x_max, y_max) tuples in normalized coordinates
            sort_order: One of "top-to-bottom", "left-to-right", or "none"

        Returns:
            Sorted list of boxes
        """
        if sort_order == "top-to-bottom":
            # Sort by y_min (top edge), then by x_min (left edge) for ties
            return sorted(boxes, key=lambda box: (box[1], box[0]))
        elif sort_order == "left-to-right":
            # Sort by x_min (left edge), then by y_min (top edge) for ties
            return sorted(boxes, key=lambda box: (box[0], box[1]))
        else:
            # "none" - return as-is (detection order)
            return boxes

    def _recognize_text_from_crop(self, crop: Image.Image, parameters: dict = None) -> str:
        """
        Recognize text from a cropped image region using TrOCR.
        """
        pixel_values = self.processor(images=crop, return_tensors="pt").pixel_values
        model_dtype = next(self.model.parameters()).dtype
        pixel_values = pixel_values.to(device=self.device, dtype=model_dtype)

        gen_params = self._filter_generation_params(parameters, self.SPECIAL_PARAMS)
        gen_params = self._sanitize_generation_params(gen_params)
        gen_params.setdefault("max_new_tokens", 256)

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

            boxes = self._detect_text_regions(image)
            num_boxes = len(boxes)
            logger.info("TrOCR detected %d text box(es) in image", num_boxes)

            # Sort boxes if requested
            if num_boxes > 0 and sort_boxes != "none":
                boxes = self._sort_boxes(boxes, sort_boxes)

            recognized_lines = []

            if num_boxes == 0:
                logger.info("No boxes detected, processing whole image")
                text = self._recognize_text_from_crop(image, parameters)
                if text:
                    recognized_lines.append(text)
            else:
                width, height = image.size
                for x_min, y_min, x_max, y_max in boxes:
                    left, top = int(x_min * width), int(y_min * height)
                    right, bottom = int(x_max * width), int(y_max * height)
                    crop = image.crop((left, top, right, bottom))
                    text = self._recognize_text_from_crop(crop, parameters)
                    if text:
                        recognized_lines.append(text)

            final_text = line_separator.join(recognized_lines)
            logger.info("Total lines recognized: %d", len(recognized_lines))
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
