import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from .base_adapter import BaseModelAdapter
import logging

logger = logging.getLogger(__name__)

class Blip2Adapter(BaseModelAdapter):
    # Parameters that should not be passed to model.generate()
    SPECIAL_PARAMS = {'batch_size', 'precision'}

    def __init__(self, model_id: str = "Salesforce/blip2-opt-2.7b-coco"):
        super().__init__(model_id)
        self.model_id = model_id
        self.device = self._init_device(torch)
        self.model_dtype = None

    def load_model(self, precision: str = "float16") -> None:
        try:
            logger.info("Loading BLIP2 model %s on %s with precision=%s…", self.model_id, self.device, precision)

            # Prefer fast image processor/tokenizer when available to avoid HF warning
            self.processor = Blip2Processor.from_pretrained(self.model_id, use_fast=True)

            # Determine dtype and quantization config
            quantization_config = None
            if precision in ["4bit", "8bit"]:
                quantization_config = self._create_quantization_config(precision)
            
            model_dtype = self._get_dtype(precision)

            # Build model kwargs
            model_kwargs = {"device_map": {"": 0}} if self.device == "cuda" else {}

            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                # For quantized models, set compute dtype explicitly when reasonable
                model_kwargs["torch_dtype"] = torch.float16
            elif model_dtype != "auto":
                model_kwargs["torch_dtype"] = model_dtype

            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_id,
                **model_kwargs
            )

            # Move to device if not using quantization (which handles device placement)
            if not quantization_config:
                self.model.to(self.device)

            self.model.eval()
            self.model_dtype = model_kwargs.get("torch_dtype", torch.float32)

            # Setup pad token for batch processing
            self._setup_pad_token()

            logger.info("BLIP2 model loaded successfully")
        except Exception as e:
            logger.exception("Error loading BLIP2 model: %s", e)
            raise

    def generate_caption(self, image: Image.Image, prompt: str = None, parameters: dict = None) -> str:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            image = self._ensure_rgb(image)

            # Process inputs with optional prompt
            if prompt and prompt.strip():
                inputs = self.processor(images=image, text=prompt.strip(), return_tensors="pt")
            else:
                inputs = self.processor(images=image, return_tensors="pt")

            # Move inputs to device with proper dtype
            inputs = self._move_inputs_to_device(inputs, self.device, self.model_dtype)

            # Filter generation parameters
            gen_params = self._filter_generation_params(parameters, self.SPECIAL_PARAMS)

            logger.debug("BLIP2 params: %s", gen_params if gen_params else "defaults")

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **gen_params)

            caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            return caption

        except Exception as e:
            logger.exception("Error generating caption: %s", e)
            return f"Error: {str(e)}"

    def generate_captions_batch(self, images: list, prompts: list = None, parameters: dict = None) -> list:
        """Generate captions for multiple images using batch processing"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Ensure we have prompts for all images
            if prompts is None:
                prompts = [None] * len(images)
            elif len(prompts) != len(images):
                raise ValueError(f"Number of prompts ({len(prompts)}) must match number of images ({len(images)})")

            # Convert all images to RGB
            processed_images = self._ensure_rgb(images)

            # Build generation parameters
            gen_params = self._filter_generation_params(parameters, self.SPECIAL_PARAMS)

            # Process batch with padding
            if prompts[0] and prompts[0].strip():
                # With prompts
                text_prompts = [p.strip() if p else "" for p in prompts]
                inputs = self.processor(
                    images=processed_images,
                    text=text_prompts,
                    return_tensors="pt",
                    padding=True
                )
            else:
                # Without prompts (unconditional generation)
                inputs = self.processor(
                    images=processed_images,
                    return_tensors="pt",
                    padding=True
                )

            # Move inputs to device with proper dtype
            inputs = self._move_inputs_to_device(inputs, self.device, self.model_dtype)

            # Generate for batch
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **gen_params)

            # Batch decode
            captions = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )

            return [caption.strip() if caption else "Unable to generate description." for caption in captions]

        except Exception as e:
            logger.exception("Error generating captions in batch with BLIP2: %s", e)
            return self._format_batch_error(e, len(images))

    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def get_available_parameters(self) -> list:
        return [
            {
                "name": "Precision",
                "param_key": "precision",
                "type": "select",
                "options": [
                    {"label": "FP32 (High Quality)", "value": "float32"},
                    {"label": "FP16 (Balanced)", "value": "float16"},
                    {"label": "BF16 (Best for A100/H100)", "value": "bfloat16"},
                    {"label": "8-bit (Low VRAM)", "value": "8bit"},
                    {"label": "4-bit (Minimal VRAM)", "value": "4bit"}
                ],
                "description": "Precision mode - lower precision uses less VRAM (requires model reload)"
            },
            {
                "name": "Max Length",
                "param_key": "max_length",
                "type": "number",
                "min": 1,
                "max": 500,
                "step": 1,
                "description": "Maximum number of tokens to generate"
            },
            {
                "name": "Min Length",
                "param_key": "min_length",
                "type": "number",
                "min": 0,
                "max": 100,
                "step": 1,
                "description": "Minimum number of tokens to generate"
            },
            {
                "name": "Num Beams",
                "param_key": "num_beams",
                "type": "number",
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Number of beams for beam search"
            },
            {
                "name": "Temperature",
                "param_key": "temperature",
                "type": "number",
                "min": 0,
                "max": 2,
                "step": 0.1,
                "description": "Sampling temperature for randomness"
            },
            {
                "name": "Top P",
                "param_key": "top_p",
                "type": "number",
                "min": 0,
                "max": 1,
                "step": 0.01,
                "description": "Nucleus sampling probability threshold"
            },
            {
                "name": "Top K",
                "param_key": "top_k",
                "type": "number",
                "min": 0,
                "max": 200,
                "step": 1,
                "description": "Top-k sampling: limit to k highest probability tokens"
            },
            {
                "name": "Repetition Penalty",
                "param_key": "repetition_penalty",
                "type": "number",
                "min": 1,
                "max": 2,
                "step": 0.1,
                "description": "Penalty for repeating tokens"
            },
            {
                "name": "Length Penalty",
                "param_key": "length_penalty",
                "type": "number",
                "min": 0,
                "max": 2,
                "step": 0.1,
                "description": "Exponential penalty to the length for beam search"
            },
            {
                "name": "Batch Size",
                "param_key": "batch_size",
                "type": "number",
                "min": 1,
                "max": 16,
                "step": 1,
                "description": "Number of images to process simultaneously (higher = faster but more VRAM)"
            }
        ]
