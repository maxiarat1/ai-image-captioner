import torch
import logging
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from chandra.model.hf import generate_hf
from chandra.model.schema import BatchInputItem
from .base_adapter import BaseModelAdapter

logger = logging.getLogger(__name__)

class ChandraAdapter(BaseModelAdapter):
    # Parameters that should not be passed to generate_hf()
    # Note: generate_hf only accepts max_output_tokens parameter, doesn't pass **kwargs to model.generate()
    SPECIAL_PARAMS = {'precision', 'use_flash_attention', 'batch_size', 'max_new_tokens', 'chandra_preset'}

    def __init__(self, model_id="datalab-to/chandra"):
        super().__init__(model_id)
        self.model_id = model_id  # Explicitly store model_id
        self.device = self._init_device(torch)

    def load_model(self, precision="bfloat16", use_flash_attention=False) -> None:
        try:
            # Chandra OCR only supports float precisions (no quantization)
            # Quantization produces corrupted output due to vision-language architecture
            supported_precisions = ["float32", "float16", "bfloat16"]
            if precision not in supported_precisions:
                logger.warning("Chandra OCR only supports float precisions. Falling back to bfloat16.")
                precision = "bfloat16"

            logger.info("Loading Chandra OCR model on %s with %s precision…", self.device, precision)

            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

            # Setup pad token for batch processing
            self._setup_pad_token()

            # Build model kwargs with torch dtype
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": self._get_dtype(precision)
            }

            # Setup flash attention if requested
            if use_flash_attention:
                self._setup_flash_attention(model_kwargs, precision, force_bfloat16=True)

            self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **model_kwargs)
            self.model.to(self.device)

            # Attach processor to model (required by Chandra's generate_hf)
            self.model.processor = self.processor

            # Configure generation settings for deterministic OCR output
            if self.processor.tokenizer.pad_token_id is not None:
                self.model.generation_config.pad_token_id = self.processor.tokenizer.pad_token_id

            # Use greedy decoding (do_sample=False) for deterministic OCR
            self.model.generation_config.do_sample = False
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None
            self.model.generation_config.top_k = None

            self.model.eval()
            logger.info("Chandra OCR model loaded successfully (Precision: %s)", precision)

        except Exception as e:
            logger.exception("Error loading Chandra OCR model: %s", e)
            raise

    def generate_caption(self, image: Image.Image, prompt: str = None, parameters: dict = None) -> str:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            image = self._ensure_rgb(image)

            # Extract max_output_tokens (Chandra's generate_hf uses this parameter name)
            max_output_tokens = parameters.get("max_new_tokens", 512) if parameters else 512

            # Extract Chandra preset selection
            chandra_preset = parameters.get("chandra_preset", "user_prompt") if parameters else "user_prompt"

            # Create BatchInputItem based on preset selection
            if chandra_preset == "ocr_layout":
                # Force use of Chandra's OCR layout preset (ignore user prompt)
                batch_item = BatchInputItem(image=image, prompt_type="ocr_layout")
            elif chandra_preset == "ocr":
                # Force use of Chandra's simple OCR preset (ignore user prompt)
                batch_item = BatchInputItem(image=image, prompt_type="ocr")
            else:  # user_prompt (default)
                # Use custom user prompt if provided, otherwise default to ocr_layout
                if prompt:
                    batch_item = BatchInputItem(image=image, prompt=prompt)
                else:
                    batch_item = BatchInputItem(image=image, prompt_type="ocr_layout")

            batch = [batch_item]

            # Generate using Chandra's generate_hf
            # Note: This function internally handles chat template, processing, and generation
            with torch.inference_mode():
                results = generate_hf(batch, self.model, max_output_tokens=max_output_tokens)

            # Extract raw text from GenerationResult
            caption = results[0].raw if results and hasattr(results[0], 'raw') else ""

            return caption.strip() if caption else "Unable to generate description."

        except Exception as e:
            logger.exception("Error generating caption with Chandra OCR: %s", e)
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

            # Extract max_output_tokens
            max_output_tokens = parameters.get("max_new_tokens", 512) if parameters else 512

            # Extract Chandra preset selection
            chandra_preset = parameters.get("chandra_preset", "user_prompt") if parameters else "user_prompt"

            # Build batch items based on preset selection
            batch = []
            for i, img in enumerate(processed_images):
                if chandra_preset == "ocr_layout":
                    # Force use of Chandra's OCR layout preset (ignore user prompt)
                    batch_item = BatchInputItem(image=img, prompt_type="ocr_layout")
                elif chandra_preset == "ocr":
                    # Force use of Chandra's simple OCR preset (ignore user prompt)
                    batch_item = BatchInputItem(image=img, prompt_type="ocr")
                else:  # user_prompt (default)
                    # Use custom user prompt if provided, otherwise default to ocr_layout
                    if prompts[i]:
                        batch_item = BatchInputItem(image=img, prompt=prompts[i])
                    else:
                        batch_item = BatchInputItem(image=img, prompt_type="ocr_layout")
                batch.append(batch_item)

            # Generate for batch using Chandra's generate_hf
            with torch.inference_mode():
                results = generate_hf(batch, self.model, max_output_tokens=max_output_tokens)

            # Extract raw text from GenerationResult objects
            captions = []
            for result in results:
                caption = result.raw if hasattr(result, 'raw') else ""
                captions.append(caption.strip() if caption else "Unable to generate description.")

            return captions

        except Exception as e:
            logger.exception("Error generating captions in batch with Chandra OCR: %s", e)
            return self._format_batch_error(e, len(images))

    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def get_available_parameters(self) -> list:
        # Note: Chandra's generate_hf only supports max_output_tokens parameter
        # Other generation parameters (do_sample, temperature, etc.) are not passed through
        return [
            {
                "name": "Max New Tokens",
                "param_key": "max_new_tokens",
                "type": "number",
                "min": 1,
                "max": 4096,
                "step": 1,
                "description": "Maximum number of tokens to generate",
                "group": "general"
            },
            {
                "name": "Chandra Preset",
                "param_key": "chandra_preset",
                "type": "select",
                "options": [
                    {"value": "user_prompt", "label": "User Prompt (Default)"},
                    {"value": "ocr_layout", "label": "OCR with Layout (Chandra Preset)"},
                    {"value": "ocr", "label": "Simple OCR (Chandra Preset)"}
                ],
                "description": "Choose between custom prompt or Chandra's predefined OCR prompts. OCR presets ignore user-entered text.",
                "group": "advanced"
            },
            {
                "name": "Precision",
                "param_key": "precision",
                "type": "select",
                "options": [
                    {"value": "bfloat16", "label": "BFloat16 (Recommended)"},
                    {"value": "float16", "label": "Float16"},
                    {"value": "float32", "label": "Float32 (Full)"}
                ],
                "description": "Model precision mode (requires model reload). Quantization not supported.",
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
        super().unload()
