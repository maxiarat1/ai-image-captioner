import torch
import logging
from PIL import Image
from transformers import AutoProcessor, AutoModel
from .base_adapter import BaseModelAdapter

logger = logging.getLogger(__name__)

class R4BAdapter(BaseModelAdapter):
    # Parameters that should not be passed to model.generate()
    SPECIAL_PARAMS = {'precision', 'use_flash_attention', 'thinking_mode', 'batch_size'}

    def __init__(self):
        super().__init__("YannQi/R-4B")
        self.device = self._init_device(torch)
        self.quantization_config = None

    def _extract_final_result(self, caption: str) -> str:
        """Extract final caption from R-4B output (removes thinking tags)"""
        if not caption:
            return caption
        if "</think>" in caption:
            parts = caption.split("</think>")
            if len(parts) > 1:
                return parts[-1].replace('\n', ' ').strip()
        if caption.startswith("Auto-Thinking Output: "):
            return caption[len("Auto-Thinking Output: "):].strip()
        return caption.strip()

    def _extract_thinking_mode(self, parameters: dict) -> str:
        """Extract and validate thinking_mode parameter"""
        if not parameters:
            return 'auto'
        thinking_mode = parameters.get('thinking_mode', 'auto')
        return thinking_mode if thinking_mode in ['auto', 'short', 'long'] else 'auto'

    def load_model(self, precision="float32", use_flash_attention=False) -> None:
        try:
            logger.info("Loading R-4B model on %s with %s precision…", self.device, precision)

            self.quantization_config = self._create_quantization_config(precision)

            self.processor = AutoProcessor.from_pretrained("YannQi/R-4B", trust_remote_code=True, use_fast=True)

            # Setup pad token for batch processing
            self._setup_pad_token()

            model_kwargs = {"trust_remote_code": True, "quantization_config": self.quantization_config}

            if precision not in ["4bit", "8bit"]:
                model_kwargs["torch_dtype"] = self._get_dtype(precision)

            # Setup flash attention if requested
            if use_flash_attention:
                self._setup_flash_attention(model_kwargs, precision, force_bfloat16=False)

            self.model = AutoModel.from_pretrained("YannQi/R-4B", **model_kwargs)

            if precision not in ["4bit", "8bit"]:
                self.model.to(self.device)

            self.model.eval()
            logger.info("R-4B model loaded successfully (Precision: %s)", precision)

        except Exception as e:
            logger.exception("Error loading R-4B model: %s", e)
            raise

    def generate_caption(self, image: Image.Image, prompt: str = None, parameters: dict = None) -> str:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            image = self._ensure_rgb(image)

            # Extract special parameters
            thinking_mode = self._extract_thinking_mode(parameters)

            # Build generation parameters
            gen_params = self._filter_generation_params(parameters, self.SPECIAL_PARAMS)
            gen_params = self._sanitize_generation_params(gen_params)

            prompt = prompt or "Describe this image."

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }]

            text = self.processor.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True, thinking_mode=thinking_mode)

            inputs = self.processor(images=image, text=text, return_tensors="pt")

            # Move inputs to device with proper dtype
            model_dtype = next(self.model.parameters()).dtype
            inputs = self._move_inputs_to_device(inputs, self.device, model_dtype)

            generated_ids = self.model.generate(**inputs, **gen_params)
            output_ids = generated_ids[0][len(inputs["input_ids"][0]):]
            caption = self.processor.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            final_result = self._extract_final_result(caption)
            return final_result if final_result else "Unable to generate description."

        except Exception as e:
            logger.exception("Error generating caption with R-4B: %s", e)
            return f"Error: {str(e)}"

    def generate_captions_batch(self, images: list, prompts: list = None, parameters: dict = None) -> list:
        """Generate captions for multiple images using batch processing"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Ensure we have prompts for all images
            if prompts is None:
                prompts = ["Describe this image."] * len(images)
            elif len(prompts) != len(images):
                raise ValueError(f"Number of prompts ({len(prompts)}) must match number of images ({len(images)})")

            # Convert all images to RGB
            processed_images = self._ensure_rgb(images)

            # Extract special parameters
            thinking_mode = self._extract_thinking_mode(parameters)

            # Build generation parameters
            gen_params = self._filter_generation_params(parameters, self.SPECIAL_PARAMS)
            gen_params = self._sanitize_generation_params(gen_params)

            # Build batch messages
            batch_messages = []
            batch_texts = []
            for i in range(len(processed_images)):
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": processed_images[i]},
                        {"type": "text", "text": prompts[i]}
                    ]
                }]
                text = self.processor.apply_chat_template(messages, tokenize=False,
                                                         add_generation_prompt=True, thinking_mode=thinking_mode)
                batch_texts.append(text)

            # Process batch with padding
            inputs = self.processor(
                images=processed_images,
                text=batch_texts,
                return_tensors="pt",
                padding=True
            )

            # Move inputs to device with proper dtype
            model_dtype = next(self.model.parameters()).dtype
            inputs = self._move_inputs_to_device(inputs, self.device, model_dtype)

            # Generate for batch
            generated_ids = self.model.generate(**inputs, **gen_params)

            # Trim input tokens from outputs
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]

            # Batch decode
            captions = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            # Extract final results (remove thinking tags)
            results = []
            for caption in captions:
                final_result = self._extract_final_result(caption)
                results.append(final_result if final_result else "Unable to generate description.")

            return results

        except Exception as e:
            logger.exception("Error generating captions in batch with R-4B: %s", e)
            return self._format_batch_error(e, len(images))

    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def get_available_parameters(self) -> list:
        return [
            {
                "name": "Max New Tokens",
                "param_key": "max_new_tokens",
                "type": "number",
                "min": 1,
                "max": 32768,
                "step": 1,
                "description": "Maximum number of new tokens to generate",
                "group": "general"
            },
            {
                "name": "Do Sample",
                "param_key": "do_sample",
                "type": "checkbox",
                "description": "Enable sampling (required for temperature/top_p/top_k to work)",
                "group": "mode"
            },
            {
                "name": "Temperature",
                "param_key": "temperature",
                "type": "number",
                "min": 0.1,
                "max": 2,
                "step": 0.1,
                "description": "Sampling temperature for randomness",
                "group": "sampling",
                "requires": "do_sample"
            },
            {
                "name": "Top P",
                "param_key": "top_p",
                "type": "number",
                "min": 0,
                "max": 1,
                "step": 0.01,
                "description": "Nucleus sampling probability threshold",
                "group": "sampling",
                "requires": "do_sample"
            },
            {
                "name": "Top K",
                "param_key": "top_k",
                "type": "number",
                "min": 0,
                "max": 200,
                "step": 1,
                "description": "Top-k sampling: limit to k highest probability tokens",
                "group": "sampling",
                "requires": "do_sample"
            },
            {
                "name": "Thinking Mode",
                "param_key": "thinking_mode",
                "type": "select",
                "options": [
                    {"value": "auto", "label": "Auto"},
                    {"value": "short", "label": "Short"},
                    {"value": "long", "label": "Long"}
                ],
                "description": "Verbosity of reasoning process",
                "group": "general"
            },
            {
                "name": "Precision",
                "param_key": "precision",
                "type": "select",
                "options": [
                    {"value": "float32", "label": "Float32 (Full)"},
                    {"value": "float16", "label": "Float16 (Half)"},
                    {"value": "bfloat16", "label": "BFloat16"},
                    {"value": "4bit", "label": "4-bit Quantized"},
                    {"value": "8bit", "label": "8-bit Quantized"}
                ],
                "description": "Model precision mode (requires model reload)",
                "group": "advanced"
            },
            {
                "name": "Use Flash Attention 2",
                "param_key": "use_flash_attention",
                "type": "checkbox",
                "description": "Enable Flash Attention for better performance (requires flash-attn package)",
                "group": "advanced"
            },
            {
                "name": "Batch Size",
                "param_key": "batch_size",
                "type": "number",
                "min": 1,
                "max": 8,
                "step": 1,
                "description": "Number of images to process simultaneously (higher = faster but more VRAM)",
                "group": "advanced"
            }
        ]

    def unload(self) -> None:
        if hasattr(self, 'current_precision_params'):
            delattr(self, 'current_precision_params')
        if hasattr(self, 'quantization_config'):
            self.quantization_config = None
        super().unload()
