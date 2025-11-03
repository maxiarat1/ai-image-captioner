import logging
from typing import List

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from .base_adapter import BaseModelAdapter

logger = logging.getLogger(__name__)


class NanonetsOCRAdapter(BaseModelAdapter):
    """
    Adapter for Nanonets OCR-S model (nanonets/Nanonets-OCR-s).

    Provides OCR/text extraction from document images using the project's
    standard BaseModelAdapter interface.
    """

    # Parameters that should not be passed to model.generate()
    SPECIAL_PARAMS = {"precision", "use_flash_attention", "batch_size"}

    def __init__(self, model_id: str = "nanonets/Nanonets-OCR-s"):
        super().__init__(model_id)
        self.model_id = model_id
        self.device = self._init_device(torch)
        self.quantization_config = None

    def _default_prompt(self) -> str:
        # Default OCR prompt adapted from the user's working script
        return (
            "Extract the text from the above document as if you were reading it naturally.\n"
            "Return the tables in HTML format. Return the equations in LaTeX representation.\n"
            "If there is an image in the document and an image caption is not present, add a small description\n"
            "of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>.\n"
            "Watermarks should be wrapped in <watermark></watermark>.\n"
            "Page numbers should be wrapped in <page_number></page_number>.\n"
            "Prefer using ☐ and ☑ for check boxes."
        )

    def load_model(self, precision: str = "bfloat16", use_flash_attention: bool = False) -> None:
        try:
            logger.info(
                "Loading Nanonets OCR model %s on %s with %s precision…",
                self.model_id,
                self.device,
                precision,
            )

            # Load processor (tokenizer included)
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, use_fast=True, trust_remote_code=True
            )

            # Ensure pad token for batch processing
            self._setup_pad_token()

            # Quantization and dtype setup
            self.quantization_config = self._create_quantization_config(precision)

            model_kwargs = {
                "trust_remote_code": True,
                "quantization_config": self.quantization_config,
            }

            # Choose torch dtype for from_pretrained (do not forward as 'dtype' to __init__)
            torch_dtype = None
            if precision not in ["4bit", "8bit"]:
                _dtype = self._get_dtype(precision)
                if _dtype != "auto":
                    torch_dtype = _dtype

            # Optional flash attention
            if use_flash_attention:
                self._setup_flash_attention(model_kwargs, precision, force_bfloat16=True)

            # Load model
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,  # safe for HF; not forwarded to model __init__
                **model_kwargs,
            )

            # Move to device if not quantized
            if precision not in ["4bit", "8bit"]:
                self.model = self.model.to(self.device)

            self.model.eval()
            logger.info("Nanonets OCR model loaded successfully (Precision: %s)", precision)

        except Exception as e:
            logger.exception("Error loading Nanonets OCR model %s: %s", self.model_id, e)
            raise

    def generate_caption(self, image: Image.Image, prompt: str = None, parameters: dict = None) -> str:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Ensure image in RGB
            image = self._ensure_rgb(image)

            # Default deterministic-ish OCR prompt if none provided
            prompt = (prompt or self._default_prompt()).strip()

            # Build chat-style messages (supported by this processor/model)f
            messages = [
                {"role": "system", "content": "You are a helpful OCR assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]

            # Apply chat template and encode
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(
                text=[text], images=[image], padding=True, return_tensors="pt"
            )

            # Move inputs to device/dtype
            model_dtype = next(self.model.parameters()).dtype
            inputs = self._move_inputs_to_device(inputs, self.device, model_dtype)

            # Generation parameters
            gen_params = self._filter_generation_params(parameters, self.SPECIAL_PARAMS)
            
            # Apply defaults before sanitization
            gen_params.setdefault("do_sample", False)

            # Sanitize after defaults (will remove temperature if do_sample=False)
            gen_params = self._sanitize_generation_params(gen_params)

            logger.debug("Nanonets OCR gen params: %s", gen_params)

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_params)

            # Skip prompt tokens and decode
            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = output_ids[:, prompt_len:]
            text_output = self.processor.tokenizer.batch_decode(
                new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]

            return text_output.strip() if text_output else ""

        except Exception as e:
            logger.exception("Error generating OCR with Nanonets: %s", e)
            return f"Error: {str(e)}"

    def generate_captions_batch(
        self, images: List[Image.Image], prompts: List[str] = None, parameters: dict = None
    ) -> List[str]:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Prepare prompts
            if prompts is None:
                default_prompt = self._default_prompt()
                prompts = [default_prompt] * len(images)
            elif len(prompts) != len(images):
                raise ValueError(
                    f"Number of prompts ({len(prompts)}) must match number of images ({len(images)})"
                )

            # Ensure RGB
            processed_images = self._ensure_rgb(images)

            # Build batch texts and images
            batch_texts = []
            batch_images = []
            for img, prm in zip(processed_images, prompts):
                p = (prm or self._default_prompt()).strip()
                messages = [
                    {"role": "system", "content": "You are a helpful OCR assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": p},
                        ],
                    },
                ]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                batch_texts.append(text)
                batch_images.append(img)

            inputs = self.processor(
                text=batch_texts, images=batch_images, padding=True, return_tensors="pt"
            )

            model_dtype = next(self.model.parameters()).dtype
            inputs = self._move_inputs_to_device(inputs, self.device, model_dtype)

            gen_params = self._filter_generation_params(parameters, self.SPECIAL_PARAMS)
            
            # Apply defaults before sanitization
            gen_params.setdefault("do_sample", False)
            gen_params.setdefault("max_new_tokens", 1024)
            
            # Sanitize after defaults (will remove temperature if do_sample=False)
            gen_params = self._sanitize_generation_params(gen_params)

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **gen_params)

            # Trim input tokens from outputs
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]

            # Decode batch
            text_outputs = self.processor.tokenizer.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            return [t.strip() for t in text_outputs]

        except Exception as e:
            logger.exception("Error generating batch OCR with Nanonets: %s", e)
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
                    {"value": "bfloat16", "label": "BFloat16 (Recommended)"},
                    {"value": "float16", "label": "Float16 (Half)"},
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
                "name": "Do Sample",
                "param_key": "do_sample",
                "type": "checkbox",
                "description": "Enable sampling mode (default: disabled for deterministic OCR). Required for temperature.",
                "group": "sampling"
            },
            {
                "name": "Temperature",
                "param_key": "temperature",
                "type": "number",
                "min": 0.1,
                "max": 2,
                "step": 0.1,
                "description": "Sampling temperature (requires do_sample=true, default: 0.1)",
                "depends_on": "do_sample",
                "group": "sampling"
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
