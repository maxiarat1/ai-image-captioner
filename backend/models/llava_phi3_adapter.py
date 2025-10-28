import torch
import logging
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from .base_adapter import BaseModelAdapter

logger = logging.getLogger(__name__)

class LlavaPhiAdapter(BaseModelAdapter):
    # Parameters that should not be passed to model.generate()
    SPECIAL_PARAMS = {'precision', 'use_flash_attention', 'batch_size'}

    def __init__(self, model_id: str = "xtuner/llava-phi-3-mini-hf"):
        """
        Initialize LLaVA-Phi-3-Mini adapter.

        This is a compact vision-language model based on Phi-3,
        optimized for efficient image captioning and visual question answering.
        """
        super().__init__(model_id)
        self.model_id = model_id
        self.device = self._init_device(torch)
        self.quantization_config = None

    def _format_prompt(self, prompt: str) -> str:
        """
        Format prompt with LLaVA-Phi-3 special tokens.

        Template: <|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n
        """
        if not prompt or not prompt.strip():
            prompt = "Describe this image in detail."
        else:
            prompt = prompt.strip()

        # Ensure <image> token is present
        if "<image>" not in prompt:
            formatted_prompt = f"<|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n"
        else:
            # User already included <image> token, just wrap with chat template
            formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"

        return formatted_prompt

    def load_model(self, precision="float16", use_flash_attention=False) -> None:
        try:
            logger.info("Loading LLaVA-Phi-3-Mini model %s on %s with %s precision…",
                       self.model_id, self.device, precision)

            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )

            # Manually set patch_size if not set (LLaVA processor bug workaround)
            if hasattr(self.processor, 'patch_size') and self.processor.patch_size is None:
                if hasattr(self.processor, 'image_processor') and hasattr(self.processor.image_processor, 'patch_size'):
                    self.processor.patch_size = self.processor.image_processor.patch_size
                else:
                    # Default patch size for LLaVA models
                    self.processor.patch_size = 14
                    logger.warning("patch_size not found in config, using default: 14")

            # Setup pad token for batch processing
            self._setup_pad_token()

            # Create quantization config if needed
            self.quantization_config = self._create_quantization_config(precision)

            model_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "quantization_config": self.quantization_config
            }

            # Set dtype based on precision
            if precision not in ["4bit", "8bit"]:
                dtype = self._get_dtype(precision)
                if dtype != "auto":
                    model_kwargs["dtype"] = dtype

            # Setup flash attention if requested
            if use_flash_attention:
                self._setup_flash_attention(
                    model_kwargs, precision, force_bfloat16=False
                )

            # Load model
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id,
                **model_kwargs
            )

            # Move to device if not using quantization
            if precision not in ["4bit", "8bit"]:
                self.model = self.model.to(self.device)

            self.model.eval()
            logger.info("LLaVA-Phi-3-Mini model loaded successfully (Precision: %s)", precision)

        except Exception as e:
            logger.exception("Error loading LLaVA-Phi-3-Mini model %s: %s", self.model_id, e)
            raise

    def generate_caption(self, image: Image.Image, prompt: str = None, parameters: dict = None) -> str:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            image = self._ensure_rgb(image)

            # Format prompt with special tokens
            formatted_prompt = self._format_prompt(prompt)

            # Process inputs (use keyword arguments for LLaVA processor)
            inputs = self.processor(
                text=formatted_prompt,
                images=image,
                return_tensors='pt'
            )

            # Move inputs to device with proper dtype
            model_dtype = next(self.model.parameters()).dtype
            inputs = self._move_inputs_to_device(inputs, self.device, model_dtype)

            # Build generation parameters
            gen_params = self._filter_generation_params(parameters, self.SPECIAL_PARAMS)

            # Set defaults if not provided
            if 'max_new_tokens' not in gen_params:
                gen_params['max_new_tokens'] = 200
            if 'do_sample' not in gen_params:
                gen_params['do_sample'] = False  # Greedy decoding by default

            logger.debug("LLaVA-Phi-3 params: %s", gen_params)

            # Generate output
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_params)

            # Decode output, skipping input tokens
            prompt_length = inputs["input_ids"].shape[1]
            new_tokens = output_ids[0][prompt_length:]
            caption = self.processor.decode(new_tokens, skip_special_tokens=True).strip()

            return caption if caption else "Unable to generate description."

        except Exception as e:
            logger.exception("Error generating caption with LLaVA-Phi-3: %s", e)
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

            # Format all prompts
            formatted_prompts = [self._format_prompt(p) for p in prompts]

            # Process batch with padding
            inputs = self.processor(
                text=formatted_prompts,
                images=processed_images,
                return_tensors='pt',
                padding=True
            )

            # Move inputs to device with proper dtype
            model_dtype = next(self.model.parameters()).dtype
            inputs = self._move_inputs_to_device(inputs, self.device, model_dtype)

            # Build generation parameters
            gen_params = self._filter_generation_params(parameters, self.SPECIAL_PARAMS)

            # Set defaults if not provided
            if 'max_new_tokens' not in gen_params:
                gen_params['max_new_tokens'] = 200
            if 'do_sample' not in gen_params:
                gen_params['do_sample'] = False

            # Generate for batch
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **gen_params)

            # Trim input tokens from outputs
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]

            # Batch decode
            captions = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True
            )

            # Clean up results
            results = [caption.strip() if caption.strip() else "Unable to generate description."
                      for caption in captions]

            return results

        except Exception as e:
            logger.exception("Error generating captions in batch with LLaVA-Phi-3: %s", e)
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
                "max": 500,
                "step": 1,
                "description": "Maximum number of new tokens to generate (default: 200)"
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
                "name": "Do Sample",
                "param_key": "do_sample",
                "type": "checkbox",
                "description": "Enable sampling instead of greedy decoding (default: disabled)"
            },
            {
                "name": "Num Beams",
                "param_key": "num_beams",
                "type": "number",
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Number of beams for beam search (1 = no beam search)"
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
                "name": "Precision",
                "param_key": "precision",
                "type": "select",
                "options": [
                    {"value": "float16", "label": "Float16 (Recommended)"},
                    {"value": "bfloat16", "label": "BFloat16"},
                    {"value": "float32", "label": "Float32 (Full)"},
                    {"value": "4bit", "label": "4-bit Quantized"},
                    {"value": "8bit", "label": "8-bit Quantized"}
                ],
                "description": "Model precision mode (requires model reload)"
            },
            {
                "name": "Use Flash Attention 2",
                "param_key": "use_flash_attention",
                "type": "checkbox",
                "description": "Enable Flash Attention for better performance (requires flash-attn package)"
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

    def unload(self) -> None:
        if hasattr(self, 'current_precision_params'):
            delattr(self, 'current_precision_params')
        if hasattr(self, 'quantization_config'):
            self.quantization_config = None
        super().unload()
