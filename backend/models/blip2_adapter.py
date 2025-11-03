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

    def load_model(self, precision: str = "bfloat16", *args, **kwargs) -> None:
        try:
            logger.info("Loading BLIP2 model %s on %s with precision=%s…", self.model_id, self.device, precision)

            # Prefer fast image processor/tokenizer when available to avoid HF warning
            self.processor = Blip2Processor.from_pretrained(self.model_id, use_fast=True)

            # Determine dtype - only support bfloat16 and float32 for BLIP2
            if precision == "bfloat16":
                dtype = torch.bfloat16
            elif precision == "float32":
                dtype = torch.float32
            else:
                logger.warning("Unsupported precision '%s' for BLIP2; falling back to float32", precision)
                dtype = torch.float32

            # Build model kwargs (device placement handled by from_pretrained/device move)
            model_kwargs = {"device_map": {"": 0}} if self.device == "cuda" else {}
            model_kwargs["torch_dtype"] = dtype

            # Load model with chosen dtype
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_id,
                **model_kwargs
            )

            # Move to device (when necessary, set dtype on CUDA)
            if self.device == "cuda":
                self.model.to(self.device, dtype=dtype)
            else:
                # Some devices/torch builds may not support bfloat16 on CPU; keep as loaded
                self.model.to(self.device)

            self.model.eval()
            self.model_dtype = dtype

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
            gen_params = self._sanitize_generation_params(gen_params)

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
            gen_params = self._sanitize_generation_params(gen_params)

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
                    {"label": "FP32", "value": "float32"},
                    {"label": "BF16 (bfloat16)", "value": "bfloat16"}
                ],
                "description": "Precision mode - BLIP2 supports FP32 and BF16 (requires model reload)"
            },
            {
                "name": "Do Sample",
                "param_key": "do_sample",
                "type": "checkbox",
                "description": "Enable sampling mode (required for temperature, top_p, top_k). Conflicts with num_beams>1.",
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
                "description": "Nucleus sampling probability (requires do_sample=true)",
                "depends_on": "do_sample",
                "group": "sampling"
            },
            {
                "name": "Top K",
                "param_key": "top_k",
                "type": "number",
                "min": 0,
                "max": 200,
                "step": 1,
                "description": "Top-k sampling (requires do_sample=true)",
                "depends_on": "do_sample",
                "group": "sampling"
            },
            {
                "name": "Num Beams",
                "param_key": "num_beams",
                "type": "number",
                "min": 1,
                "max": 20,
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
                "name": "Max Length",
                "param_key": "max_length",
                "type": "number",
                "min": 1,
                "max": 500,
                "step": 1,
                "description": "Maximum number of tokens to generate",
                "group": "general"
            },
            {
                "name": "Min Length",
                "param_key": "min_length",
                "type": "number",
                "min": 0,
                "max": 100,
                "step": 1,
                "description": "Minimum number of tokens to generate",
                "group": "general"
            },
            {
                "name": "Repetition Penalty",
                "param_key": "repetition_penalty",
                "type": "number",
                "min": 1,
                "max": 2,
                "step": 0.1,
                "description": "Penalty for repeating tokens (works in both modes)",
                "group": "general"
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
