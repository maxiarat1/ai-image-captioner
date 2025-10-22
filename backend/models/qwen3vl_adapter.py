import torch
from utils.torch_utils import pick_device, force_cpu_mode
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from .base_adapter import BaseModelAdapter
import logging

logger = logging.getLogger(__name__)

class Qwen3VLAdapter(BaseModelAdapter):
    """Qwen3-VL model adapter for advanced vision-language tasks (supports 4B and 8B variants)"""

    def __init__(self, model_id="Qwen/Qwen3-VL-8B-Instruct"):
        super().__init__(model_id)
        self.device = pick_device(torch)
        self.quantization_config = None
        self.model_id = model_id  # Store for later use

    def _create_quantization_config(self, precision):
        """Create quantization configuration based on precision parameter"""
        if precision == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif precision == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
        return None

    def _get_torch_dtype(self, precision):
        """Get torch dtype from precision parameter"""
        if precision == "auto":
            return "auto"

        precision_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }
        return precision_map.get(precision, "auto")

    def load_model(self, precision="auto", use_flash_attention=False) -> None:
        """Load Qwen3-VL model and processor with configurable precision and optimizations"""
        try:
            logger.info("Loading %s on %s (precision=%s)…", self.model_id, self.device, precision)

            # Create quantization config for 4bit/8bit precision
            self.quantization_config = self._create_quantization_config(precision)

            # Load processor
            logger.debug("Loading processor…")
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            if self.processor is None:
                raise RuntimeError("Processor loading failed - AutoProcessor.from_pretrained returned None")
            logger.debug("Processor loaded")

            # Prepare model loading arguments
            model_kwargs = {
                "device_map": "auto",
                "quantization_config": self.quantization_config,
            }

            # Set dtype based on precision
            if precision not in ["4bit", "8bit"]:
                dtype = self._get_torch_dtype(precision)
                if dtype == "auto" or force_cpu_mode():
                    model_kwargs["torch_dtype"] = "auto"
                else:
                    model_kwargs["torch_dtype"] = dtype

            # Add Flash Attention support if available and requested
            if use_flash_attention and not force_cpu_mode() and torch.cuda.is_available():
                try:
                    import flash_attn
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    # Override dtype to bfloat16 for flash attention (recommended by Qwen)
                    if precision not in ["4bit", "8bit"]:
                        model_kwargs["torch_dtype"] = torch.bfloat16
                    logger.info("Using Flash Attention 2 (dtype=bfloat16)")
                except ImportError:
                    logger.debug("Flash Attention not available; using default attention")

            # Load model
            logger.info("Loading model weights…")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id,
                **model_kwargs
            )
            if self.model is None:
                raise RuntimeError("Model loading failed - Qwen3VLForConditionalGeneration.from_pretrained returned None")
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
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Build generation parameters - only include what user explicitly provided
            gen_params = {}

            # Process parameters - only include what was explicitly set
            if parameters:
                logger.debug("Processing provided parameters…")

                # Get all generation param keys (excluding special params like precision, use_flash_attention)
                generation_param_keys = [p['param_key'] for p in self.get_available_parameters()
                                        if p['type'] in ['number'] and p['param_key'] not in ['precision', 'use_flash_attention']]

                # Only pass parameters that were explicitly set by the user
                for param_key in generation_param_keys:
                    if param_key in parameters:
                        gen_params[param_key] = parameters[param_key]
            else:
                logger.debug("No parameters provided, using defaults")

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

            # Decode response
            caption = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]  # Get first result since we're processing one image

            return caption.strip() if caption else "Unable to generate description."

        except Exception as e:
            logger.exception("Error generating caption with Qwen3-VL: %s", e)
            return f"Error: {str(e)}"

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
