import torch
import logging
from PIL import Image
from transformers import AutoModelForCausalLM
from .base_adapter import BaseModelAdapter

logger = logging.getLogger(__name__)

class JanusAdapter(BaseModelAdapter):
    # Parameters that should not be passed to model.generate()
    SPECIAL_PARAMS = {'precision', 'use_flash_attention', 'batch_size'}

    def __init__(self, model_id: str = "deepseek-ai/Janus-1.3B"):
        """
        Initialize Janus adapter with specified model variant.

        Supported models:
        - deepseek-ai/Janus-1.3B
        - deepseek-ai/JanusFlow-1.3B
        - deepseek-ai/Janus-Pro-1B
        - deepseek-ai/Janus-Pro-7B
        """
        super().__init__(model_id)
        self.model_id = model_id
        self.device = self._init_device(torch)
        self.vl_chat_processor = None
        self.quantization_config = None

    def load_model(self, precision="bfloat16", use_flash_attention=False) -> None:
        try:
            logger.info("Loading Janus model %s on %s with %s precision…", self.model_id, self.device, precision)

            # Import Janus-specific classes
            try:
                from janus.models import VLChatProcessor
                from transformers import AutoImageProcessor
            except ImportError:
                raise ImportError(
                    "Janus models require the 'janus' package. "
                    "Install it via: pip install git+https://github.com/deepseek-ai/Janus.git"
                )

            # Load processor with explicit fast image processor class
            self.vl_chat_processor = VLChatProcessor.from_pretrained(
                self.model_id,
                fast_image_processor_class=AutoImageProcessor
            )
            self.processor = self.vl_chat_processor  # Alias for base class compatibility

            # Setup pad token for batch processing and configure tokenizer legacy behavior
            self._setup_pad_token()

            # Set new tokenizer behavior (legacy=False) for modern tokenization
            if hasattr(self.vl_chat_processor, 'tokenizer') and hasattr(self.vl_chat_processor.tokenizer, 'init_kwargs'):
                # Update the tokenizer to use new behavior
                self.vl_chat_processor.tokenizer.legacy = False

            # Create quantization config if needed
            self.quantization_config = self._create_quantization_config(precision)

            model_kwargs = {
                "trust_remote_code": True,
                "quantization_config": self.quantization_config
            }

            # Set dtype based on precision
            if precision not in ["4bit", "8bit"]:
                dtype = self._get_dtype(precision)
                if dtype != "auto":
                    model_kwargs["torch_dtype"] = dtype

            # Setup flash attention if requested
            if use_flash_attention:
                self._setup_flash_attention(model_kwargs, precision, force_bfloat16=True)

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)

            # Move to device if not using quantization
            if precision not in ["4bit", "8bit"]:
                self.model = self.model.to(self.device)

            self.model.eval()
            logger.info("Janus model %s loaded successfully (Precision: %s)", self.model_id, precision)

        except Exception as e:
            logger.exception("Error loading Janus model %s: %s", self.model_id, e)
            raise

    def generate_caption(self, image: Image.Image, prompt: str = None, parameters: dict = None) -> str:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            image = self._ensure_rgb(image)

            # Build generation parameters (filter to only valid params for this model)
            gen_params = self._filter_generation_params(parameters, self.SPECIAL_PARAMS)
            gen_params = self._sanitize_generation_params(gen_params)

            # Use default prompt if none provided
            if not prompt or not prompt.strip():
                prompt = "Describe this image in detail."

            # Prepare conversation in Janus format
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{prompt}",
                    "images": [image],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]

            # Prepare inputs (we already have PIL images, so pass directly)
            prepare_inputs = self.vl_chat_processor(
                conversations=conversation,
                images=[image],
                force_batchify=True
            ).to(self.model.device)

            # Run image encoder to get the image embeddings
            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

            # Get tokenizer
            tokenizer = self.vl_chat_processor.tokenizer

            # Set default generation parameters if not provided
            default_params = {
                'pad_token_id': tokenizer.eos_token_id,
                'bos_token_id': tokenizer.bos_token_id,
                'eos_token_id': tokenizer.eos_token_id,
                'max_new_tokens': 512,
                'do_sample': False,
                'use_cache': True
            }

            # Merge with user parameters (user params take precedence)
            final_params = {**default_params, **gen_params}

            # Run the model to get the response
            with torch.no_grad():
                outputs = self.model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    **final_params
                )

            # Decode the output
            caption = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

            return caption.strip() if caption else "Unable to generate description."

        except Exception as e:
            logger.exception("Error generating caption with Janus: %s", e)
            return f"Error: {str(e)}"

    def generate_captions_batch(self, images: list, prompts: list = None, parameters: dict = None) -> list:
        """Generate captions for multiple images using batch processing"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Ensure we have prompts for all images
            if prompts is None:
                prompts = ["Describe this image in detail."] * len(images)
            elif len(prompts) != len(images):
                raise ValueError(f"Number of prompts ({len(prompts)}) must match number of images ({len(images)})")

            # Convert all images to RGB
            processed_images = self._ensure_rgb(images)

            # Build generation parameters
            gen_params = self._filter_generation_params(parameters, self.SPECIAL_PARAMS)
            gen_params = self._sanitize_generation_params(gen_params)

            # Process each image one at a time (Janus processes images individually)
            results = []
            tokenizer = self.vl_chat_processor.tokenizer

            for image, prompt in zip(processed_images, prompts):
                prompt_text = prompt if prompt and prompt.strip() else "Describe this image in detail."

                # Prepare conversation
                conversation = [
                    {
                        "role": "<|User|>",
                        "content": f"<image_placeholder>\n{prompt_text}",
                        "images": [image],
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]

                # Prepare inputs (we already have PIL images, so pass directly)
                prepare_inputs = self.vl_chat_processor(
                    conversations=conversation,
                    images=[image],
                    force_batchify=True
                ).to(self.model.device)

                # Run image encoder
                inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

                # Set default generation parameters
                default_params = {
                    'pad_token_id': tokenizer.eos_token_id,
                    'bos_token_id': tokenizer.bos_token_id,
                    'eos_token_id': tokenizer.eos_token_id,
                    'max_new_tokens': 512,
                    'do_sample': False,
                    'use_cache': True
                }

                # Merge with user parameters
                final_params = {**default_params, **gen_params}

                # Generate
                with torch.no_grad():
                    outputs = self.model.language_model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=prepare_inputs.attention_mask,
                        **final_params
                    )

                # Decode
                caption = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
                results.append(caption.strip() if caption else "Unable to generate description.")

            return results

        except Exception as e:
            logger.exception("Error generating captions in batch with Janus: %s", e)
            return self._format_batch_error(e, len(images))

    def is_loaded(self) -> bool:
        return self.model is not None and self.vl_chat_processor is not None

    def get_available_parameters(self) -> list:
        return [
            {
                "name": "Max New Tokens",
                "param_key": "max_new_tokens",
                "type": "number",
                "min": 1,
                "max": 2048,
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
                "name": "Precision",
                "param_key": "precision",
                "type": "select",
                "options": [
                    {"value": "bfloat16", "label": "BFloat16 (Recommended)"},
                    {"value": "float16", "label": "Float16 (Half)"},
                    {"value": "float32", "label": "Float32 (Full)"},
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
        if hasattr(self, 'vl_chat_processor') and self.vl_chat_processor is not None:
            del self.vl_chat_processor
            self.vl_chat_processor = None
        if hasattr(self, 'current_precision_params'):
            delattr(self, 'current_precision_params')
        if hasattr(self, 'quantization_config'):
            self.quantization_config = None
        super().unload()