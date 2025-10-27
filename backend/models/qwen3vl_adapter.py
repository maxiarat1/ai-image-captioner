import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from .base_adapter import BaseModelAdapter
import logging

logger = logging.getLogger(__name__)

class Qwen3VLAdapter(BaseModelAdapter):
    """Qwen3-VL model adapter for advanced vision-language tasks (supports 4B and 8B variants)"""

    # Parameters that should not be passed to model.generate()
    SPECIAL_PARAMS = {'precision', 'use_flash_attention', 'batch_size'}

    def __init__(self, model_id="Qwen/Qwen3-VL-8B-Instruct"):
        super().__init__(model_id)
        self.device = self._init_device(torch)
        self.quantization_config = None
        self.model_id = model_id  # Store for later use

    def load_model(self, precision="auto", use_flash_attention=False) -> None:
        """Load Qwen3-VL model and processor with configurable precision and optimizations"""
        try:
            from utils.torch_utils import force_cpu_mode

            logger.info("Loading %s on %s (precision=%s)…", self.model_id, self.device, precision)

            # Create quantization config for 4bit/8bit precision
            self.quantization_config = self._create_quantization_config(precision)

            # Load processor
            logger.debug("Loading processor…")
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            logger.debug("Processor loaded")

            # Setup pad token for batch processing
            self._setup_pad_token()

            # Prepare model loading arguments
            model_kwargs = {
                "device_map": "auto",
                "quantization_config": self.quantization_config,
            }

            # Set dtype based on precision
            if precision not in ["4bit", "8bit"]:
                dtype = self._get_dtype(precision)
                if dtype == "auto" or force_cpu_mode():
                    model_kwargs["dtype"] = "auto"
                else:
                    model_kwargs["dtype"] = dtype

            # Add Flash Attention support if available and requested
            if use_flash_attention:
                self._setup_flash_attention(model_kwargs, precision, force_bfloat16=True)

            # Load model
            logger.info("Loading model weights…")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id,
                **model_kwargs
            )
            logger.info("Model loaded")

            self.model.eval()

            logger.info("%s ready (precision=%s)", self.model_id, precision)

        except Exception as e:
            logger.exception("Error loading Qwen3-VL model: %s", e)
            raise

    def generate_caption(self, image: Image.Image, prompt: str = None, parameters: dict = None) -> str:
        """Generate caption for the given image using Qwen3-VL with configurable parameters"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Debug logging
            logger.debug("Qwen3VL.generate_caption | prompt=%s | params=%s", (prompt or ""), parameters)

            # Convert to RGB if necessary
            image = self._ensure_rgb(image)

            # Build generation parameters - only include what user explicitly provided
            gen_params = self._filter_generation_params(parameters, self.SPECIAL_PARAMS)

            logger.debug("Generation parameters: %s", gen_params if gen_params else "defaults")

            # Prepare prompt
            if not prompt:
                prompt = "Describe this image."

            # Create conversation messages in the format expected by Qwen3-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # Apply chat template and tokenize
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )

            # Move inputs to the model's device
            inputs = inputs.to(self.model.device)

            # Generate response - library will use its own defaults for unspecified params
            generated_ids = self.model.generate(**inputs, **gen_params)

            # Trim the generated output to remove input tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            # Decode response (single image)
            caption = self.processor.decode(
                generated_ids_trimmed[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            return caption.strip() if caption else "Unable to generate description."

        except Exception as e:
            logger.exception("Error generating caption with Qwen3-VL: %s", e)
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

            # Build generation parameters
            gen_params = self._filter_generation_params(parameters, self.SPECIAL_PARAMS)

            # Build batch messages
            batch_messages = [
                [{"role": "user", "content": [
                    {"type": "image", "image": processed_images[i]},
                    {"type": "text", "text": prompts[i]}
                ]}]
                for i in range(len(processed_images))
            ]

            # Process batch with padding
            inputs = self.processor.apply_chat_template(
                batch_messages,
                tokenize=True,
                padding=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )

            # Move inputs to device
            inputs = inputs.to(self.model.device)

            # Generate for batch
            generated_ids = self.model.generate(**inputs, **gen_params)

            # Trim input tokens from outputs
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            # Batch decode
            captions = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            result = [caption.strip() if caption else "Unable to generate description." for caption in captions]
            return result

        except Exception as e:
            logger.exception("Error generating captions in batch with Qwen3-VL: %s", e)
            return self._format_batch_error(e, len(images))

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None and self.processor is not None

    def get_available_parameters(self) -> list:
        """Get list of available parameters for Qwen3-VL model"""
        return [
            {
                "name": "Max New Tokens",
                "param_key": "max_new_tokens",
                "type": "number",
                "min": 1,
                "max": 16384,
                "step": 1,
                "description": "Maximum number of new tokens to generate (recommended: 128-16384)"
            },
            {
                "name": "Temperature",
                "param_key": "temperature",
                "type": "number",
                "min": 0,
                "max": 2,
                "step": 0.1,
                "description": "Sampling temperature for randomness (recommended: 0.7)"
            },
            {
                "name": "Top P",
                "param_key": "top_p",
                "type": "number",
                "min": 0,
                "max": 1,
                "step": 0.01,
                "description": "Nucleus sampling probability threshold (recommended: 0.8)"
            },
            {
                "name": "Top K",
                "param_key": "top_k",
                "type": "number",
                "min": 0,
                "max": 200,
                "step": 1,
                "description": "Top-k sampling: limit to k highest probability tokens (recommended: 20)"
            },
            {
                "name": "Repetition Penalty",
                "param_key": "repetition_penalty",
                "type": "number",
                "min": 0.5,
                "max": 2.0,
                "step": 0.1,
                "description": "Penalty for repeating tokens (recommended: 1.0)"
            },
            {
                "name": "Presence Penalty",
                "param_key": "presence_penalty",
                "type": "number",
                "min": 0,
                "max": 2.0,
                "step": 0.1,
                "description": "Penalty for using previously used tokens (recommended: 1.5)"
            },
            {
                "name": "Batch Size",
                "param_key": "batch_size",
                "type": "number",
                "min": 1,
                "max": 8,
                "step": 1,
                "description": "Number of images to process simultaneously (higher = faster but more VRAM)"
            },
            {
                "name": "Precision",
                "param_key": "precision",
                "type": "select",
                "options": [
                    {"value": "auto", "label": "Auto (Default)"},
                    {"value": "float32", "label": "Float32 (Full)"},
                    {"value": "float16", "label": "Float16 (Half)"},
                    {"value": "bfloat16", "label": "BFloat16"},
                    {"value": "4bit", "label": "4-bit Quantized"},
                    {"value": "8bit", "label": "8-bit Quantized"}
                ],
                "description": "Model precision mode (requires model reload)"
            },
            {
                "name": "Use Flash Attention 2",
                "param_key": "use_flash_attention",
                "type": "checkbox",
                "description": "Enable Flash Attention 2 for better acceleration and memory saving (requires flash-attn package, forces bfloat16)"
            }
        ]

    def unload(self) -> None:
        """
        Unload the Qwen3-VL model and clear precision parameters.
        This ensures complete cleanup when changing precision settings.
        """
        # Clear precision tracking
        if hasattr(self, 'current_precision_params'):
            delattr(self, 'current_precision_params')

        if hasattr(self, 'quantization_config'):
            self.quantization_config = None

        # Call parent unload method
        super().unload()
