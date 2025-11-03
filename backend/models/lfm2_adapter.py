import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from .base_adapter import BaseModelAdapter
import logging

logger = logging.getLogger(__name__)

class LFM2Adapter(BaseModelAdapter):
    # Parameters that should not be passed to model.generate()
    SPECIAL_PARAMS = {'batch_size', 'precision', 'use_flash_attention'}

    def __init__(self, model_id: str = "LiquidAI/LFM2-VL-3B"):
        super().__init__(model_id)
        self.model_id = model_id
        self.device = self._init_device(torch)
        self.current_precision_params = None

    def load_model(self, precision: str = "bfloat16", use_flash_attention: bool = False) -> None:
        try:
            logger.info("Loading LFM2 model %s on %s with precision=%s, flash_attention=%s…",
                       self.model_id, self.device, precision, use_flash_attention)

            self.processor = AutoProcessor.from_pretrained(self.model_id)

            # Track precision params for reload detection
            self.current_precision_params = {
                'precision': precision,
                'use_flash_attention': use_flash_attention
            }

            model_kwargs = {
                "device_map": "auto"
            }

            # LFM2 does not support bitsandbytes quantization (4bit/8bit) due to tensor layout incompatibility
            # Only handle float precision
            if precision in ["8bit", "4bit"]:
                logger.warning("LFM2 does not support %s quantization. Falling back to bfloat16.", precision)
                precision = "bfloat16"
            
            model_dtype = self._get_dtype(precision)
            if model_dtype != "auto":
                model_kwargs["torch_dtype"] = model_dtype

            # Setup Flash Attention if requested and available
            if use_flash_attention:
                self._setup_flash_attention(model_kwargs, precision, force_bfloat16=False)

            # Load model
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                **model_kwargs
            )

            self.model.eval()

            logger.info("LFM2 model loaded successfully")
        except Exception as e:
            logger.exception("Error loading LFM2 model: %s", e)
            raise

    def generate_caption(self, image: Image.Image, prompt: str = None, parameters: dict = None) -> str:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            image = self._ensure_rgb(image)

            # Create conversation with chat template format
            if not prompt or not prompt.strip():
                prompt = "What is in this image?"

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt.strip()},
                    ],
                },
            ]

            # Apply chat template
            inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
            ).to(self.model.device)

            gen_params = self._filter_generation_params(parameters, self.SPECIAL_PARAMS)

            logger.debug("LFM2 params: %s", gen_params if gen_params else "defaults")

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_params)

            # Decode and extract the response
            full_response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

            # Extract just the assistant's response (remove the conversation prefix)
            # The response typically includes the full conversation, we want just the answer
            return self._extract_response(full_response, prompt)

        except Exception as e:
            logger.exception("Error generating caption: %s", e)
            return f"Error: {str(e)}"

    def _extract_response(self, full_response: str, prompt: str) -> str:
        """Extract the assistant's response from the full decoded output."""
        # The full response includes the conversation, we want just the final answer
        # Try to find the assistant's response after the prompt
        if "assistant" in full_response.lower():
            parts = full_response.split("assistant", 1)
            if len(parts) > 1:
                # Clean up the response
                response = parts[1].strip()
                # Remove any leading colons or whitespace
                response = response.lstrip(":").strip()
                return response

        # Fallback: just return the full response cleaned up
        return full_response.strip()

    def generate_captions_batch(self, images: list, prompts: list = None, parameters: dict = None) -> list:
        """Generate captions for multiple images (sequential processing for LFM2)"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Ensure we have prompts for all images
            if prompts is None:
                prompts = [None] * len(images)
            elif len(prompts) != len(images):
                raise ValueError(f"Number of prompts ({len(prompts)}) must match number of images ({len(images)})")

            # Process each image sequentially (chat template doesn't support batching easily)
            captions = []
            for image, prompt in zip(images, prompts):
                caption = self.generate_caption(image, prompt, parameters)
                captions.append(caption)

            return captions

        except Exception as e:
            logger.exception("Error generating captions in batch with LFM2: %s", e)
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
                    {"value": "float32", "label": "Float32 (Best quality, most VRAM)"},
                    {"value": "float16", "label": "Float16 (Balanced)"},
                    {"value": "bfloat16", "label": "BFloat16 (Recommended)"}
                ],
                "default": "bfloat16",
                "reload_required": True,
                "description": "Model precision (Note: LFM2 does not support 4-bit/8-bit quantization)"
            },
            {
                "name": "Flash Attention",
                "param_key": "use_flash_attention",
                "type": "checkbox",
                "default": False,
                "reload_required": True,
                "description": "Use Flash Attention 2 for faster inference (requires flash-attn package)"
            },
            {
                "name": "Max New Tokens",
                "param_key": "max_new_tokens",
                "type": "number",
                "min": 1,
                "max": 1024,
                "step": 1,
                "default": 256,
                "description": "Maximum number of new tokens to generate"
            },
            {
                "name": "Temperature",
                "param_key": "temperature",
                "type": "number",
                "min": 0,
                "max": 2,
                "step": 0.1,
                "default": 1.0,
                "description": "Sampling temperature for randomness"
            },
            {
                "name": "Top P",
                "param_key": "top_p",
                "type": "number",
                "min": 0,
                "max": 1,
                "step": 0.01,
                "default": 1.0,
                "description": "Nucleus sampling probability threshold"
            },
            {
                "name": "Top K",
                "param_key": "top_k",
                "type": "number",
                "min": 0,
                "max": 200,
                "step": 1,
                "default": 50,
                "description": "Top-k sampling: limit to k highest probability tokens"
            },
            {
                "name": "Repetition Penalty",
                "param_key": "repetition_penalty",
                "type": "number",
                "min": 1,
                "max": 2,
                "step": 0.1,
                "default": 1.0,
                "description": "Penalty for repeating tokens"
            },
            {
                "name": "Do Sample",
                "param_key": "do_sample",
                "type": "checkbox",
                "default": False,
                "description": "Enable sampling (required for temperature/top_p/top_k)"
            }
        ]
