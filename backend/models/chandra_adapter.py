import logging
from typing import List

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from chandra.model.schema import BatchInputItem
from chandra.model.hf import generate_hf

from .base_adapter import BaseModelAdapter

logger = logging.getLogger(__name__)


class ChandraAdapter(BaseModelAdapter):
    """
    Adapter for Chandra OCR model (datalab-to/chandra).

    Advanced layout-aware OCR model with support for tables, equations,
    and various document types using task-specific prompt templates.
    """

    # Parameters that should not be passed to model.generate()
    SPECIAL_PARAMS = {"precision", "use_flash_attention", "batch_size", "prompt_type"}

    def __init__(self, model_id: str = "datalab-to/chandra"):
        super().__init__(model_id)
        self.model_id = model_id
        self.device = self._init_device(torch)
        self.quantization_config = None

    def load_model(self, precision: str = "float16", use_flash_attention: bool = False) -> None:
        try:
            logger.info(
                "Loading Chandra OCR model %s on %s with %s precision…",
                self.model_id,
                self.device,
                precision,
            )

            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

            # Quantization and dtype setup
            self.quantization_config = self._create_quantization_config(precision)

            model_kwargs = {
                "trust_remote_code": True,
                "quantization_config": self.quantization_config,
                "low_cpu_mem_usage": True,
            }

            # Choose torch dtype
            torch_dtype = None
            if precision not in ["4bit", "8bit"]:
                _dtype = self._get_dtype(precision)
                if _dtype != "auto":
                    torch_dtype = _dtype

            # Optional flash attention
            if use_flash_attention:
                self._setup_flash_attention(model_kwargs, precision, force_bfloat16=False)

            # Load model
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                device_map="auto" if precision in ["4bit", "8bit"] else None,
                **model_kwargs,
            )

            # Attach processor to model (required by generate_hf)
            self.model.processor = self.processor

            # Move to device if not quantized
            if precision not in ["4bit", "8bit"]:
                self.model = self.model.to(self.device)

            self.model.eval()
            logger.info("Chandra OCR model loaded successfully (Precision: %s)", precision)

        except Exception as e:
            logger.exception("Error loading Chandra OCR model %s: %s", self.model_id, e)
            raise

    def generate_caption(self, image: Image.Image, prompt: str = None, parameters: dict = None) -> str:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Ensure image in RGB
            image = self._ensure_rgb(image)

            # Extract prompt_type and max_new_tokens before filtering
            prompt_type = parameters.get("prompt_type", "ocr_layout") if parameters else "ocr_layout"
            max_output_tokens = parameters.get("max_new_tokens") if parameters else None

            # Default prompt if none provided
            if not prompt:
                prompt = "<image>"

            # Create batch input
            batch = [
                BatchInputItem(
                    image=image,
                    prompt=prompt,
                    prompt_type=prompt_type
                )
            ]

            # Filter and sanitize generation parameters
            gen_params = self._filter_generation_params(parameters, self.SPECIAL_PARAMS)
            gen_params = self._sanitize_generation_params(gen_params)

            # Remove max_new_tokens from gen_params (passed separately as max_output_tokens)
            gen_params.pop("max_new_tokens", None)

            logger.debug("Chandra OCR prompt_type: %s, max_output_tokens: %s", prompt_type, max_output_tokens)

            # Generate using Chandra's generate_hf function
            # Note: generate_hf accepts max_output_tokens and **kwargs for generation params
            result = generate_hf(batch, self.model, max_output_tokens=max_output_tokens, **gen_params)[0]

            return result.raw.strip() if result.raw else ""

        except Exception as e:
            logger.exception("Error generating OCR with Chandra: %s", e)
            return f"Error: {str(e)}"

    def generate_captions_batch(
        self, images: List[Image.Image], prompts: List[str] = None, parameters: dict = None
    ) -> List[str]:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Prepare prompts
            if prompts is None:
                prompts = ["<image>"] * len(images)
            elif len(prompts) != len(images):
                raise ValueError(
                    f"Number of prompts ({len(prompts)}) must match number of images ({len(images)})"
                )

            # Extract prompt_type and max_new_tokens
            prompt_type = parameters.get("prompt_type", "ocr_layout") if parameters else "ocr_layout"
            max_output_tokens = parameters.get("max_new_tokens") if parameters else None

            # Ensure RGB
            processed_images = self._ensure_rgb(images)

            # Build batch
            batch = []
            for img, prm in zip(processed_images, prompts):
                p = prm if prm else "<image>"
                batch.append(
                    BatchInputItem(
                        image=img,
                        prompt=p,
                        prompt_type=prompt_type
                    )
                )

            # Filter and sanitize generation parameters
            gen_params = self._filter_generation_params(parameters, self.SPECIAL_PARAMS)
            gen_params = self._sanitize_generation_params(gen_params)

            # Remove max_new_tokens from gen_params (passed separately as max_output_tokens)
            gen_params.pop("max_new_tokens", None)

            logger.debug("Chandra OCR batch size: %d, prompt_type: %s, max_output_tokens: %s",
                        len(batch), prompt_type, max_output_tokens)

            # Generate batch using Chandra's generate_hf
            results = generate_hf(batch, self.model, max_output_tokens=max_output_tokens, **gen_params)

            # Extract text from results
            text_outputs = [result.raw.strip() if result.raw else "" for result in results]

            return text_outputs

        except Exception as e:
            logger.exception("Error generating batch OCR with Chandra: %s", e)
            return self._format_batch_error(e, len(images))

    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def get_available_parameters(self) -> list:
        return [
            {
                "name": "Prompt Type",
                "param_key": "prompt_type",
                "type": "text",
                "description": "Task-specific prompt template (e.g., 'ocr_layout' for layout-aware OCR)",
                "group": "general"
            },
            {
                "name": "Precision",
                "param_key": "precision",
                "type": "select",
                "options": [
                    {"value": "float16", "label": "Float16 (Recommended)"},
                    {"value": "bfloat16", "label": "BFloat16"},
                    {"value": "float32", "label": "Float32 (Full)"},
                    {"value": "4bit", "label": "4-bit Quantized"},
                    {"value": "8bit", "label": "8-bit Quantized"},
                ],
                "description": "Model precision mode (requires model reload)"
            },
            {
                "name": "Use Flash Attention 2",
                "param_key": "use_flash_attention",
                "type": "checkbox",
                "description": "Enable Flash Attention (requires flash-attn package)"
            },
            {
                "name": "Max New Tokens",
                "param_key": "max_new_tokens",
                "type": "number",
                "min": 1,
                "max": 4096,
                "step": 1,
                "description": "Maximum number of tokens to generate (default: 1024)",
                "group": "general"
            },
            {
                "name": "Do Sample",
                "param_key": "do_sample",
                "type": "checkbox",
                "description": "Enable sampling mode (required for temperature, top_p, top_k)",
                "group": "sampling"
            },
            {
                "name": "Temperature",
                "param_key": "temperature",
                "type": "number",
                "min": 0.1,
                "max": 2,
                "step": 0.1,
                "description": "Sampling temperature for generation (requires do_sample=true)",
                "depends_on": "do_sample",
                "group": "sampling"
            },
            {
                "name": "Top P",
                "param_key": "top_p",
                "type": "number",
                "min": 0,
                "max": 1,
                "step": 0.05,
                "description": "Nucleus sampling probability threshold (requires do_sample=true)",
                "depends_on": "do_sample",
                "group": "sampling"
            },
            {
                "name": "Top K",
                "param_key": "top_k",
                "type": "number",
                "min": 0,
                "max": 100,
                "step": 1,
                "description": "Top-k sampling (requires do_sample=true, 0 = disabled)",
                "depends_on": "do_sample",
                "group": "sampling"
            },
            {
                "name": "Repetition Penalty",
                "param_key": "repetition_penalty",
                "type": "number",
                "min": 1.0,
                "max": 2.0,
                "step": 0.1,
                "description": "Penalty for repeating tokens (default: 1.0)",
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
        ]

    def unload(self) -> None:
        if hasattr(self, "quantization_config"):
            self.quantization_config = None
        super().unload()
