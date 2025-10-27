import torch
import logging
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from .base_adapter import BaseModelAdapter

logger = logging.getLogger(__name__)

class DeepSeekOCRAdapter(BaseModelAdapter):
    # Parameters that should not be passed to model.infer()
    SPECIAL_PARAMS = {'precision', 'use_flash_attention', 'batch_size'}

    def __init__(self, model_id: str = "deepseek-ai/DeepSeek-OCR"):
        """
        Initialize DeepSeek-OCR adapter.

        This model is specialized for OCR and document conversion tasks,
        capable of converting images to markdown format.
        """
        super().__init__(model_id)
        self.model_id = model_id
        self.device = self._init_device(torch)
        self.tokenizer = None
        self.quantization_config = None

    def load_model(self, precision="bfloat16", use_flash_attention=False) -> None:
        try:
            logger.info("Loading DeepSeek-OCR model %s on %s with %s precision…",
                       self.model_id, self.device, precision)

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                use_fast=True
            )
            self.processor = self.tokenizer  # Alias for base class compatibility

            # Create quantization config if needed
            self.quantization_config = self._create_quantization_config(precision)

            model_kwargs = {
                "trust_remote_code": True,
                "use_safetensors": True,
                "quantization_config": self.quantization_config
            }

            # Set dtype based on precision
            if precision not in ["4bit", "8bit"]:
                dtype = self._get_dtype(precision)
                if dtype != "auto":
                    model_kwargs["torch_dtype"] = dtype

            # Setup flash attention if requested
            if use_flash_attention:
                flash_enabled = self._setup_flash_attention(
                    model_kwargs, precision, force_bfloat16=True
                )
                # Rename key for DeepSeek-OCR
                if flash_enabled and "attn_implementation" in model_kwargs:
                    model_kwargs["_attn_implementation"] = model_kwargs.pop("attn_implementation")

            # Load model
            self.model = AutoModel.from_pretrained(self.model_id, **model_kwargs)

            # Move to device and set precision if not using quantization
            if precision not in ["4bit", "8bit"]:
                self.model = self.model.to(self.device)
                if precision == "bfloat16" and torch.cuda.is_available():
                    self.model = self.model.to(torch.bfloat16)

            self.model.eval()
            logger.info("DeepSeek-OCR model loaded successfully (Precision: %s)", precision)

        except Exception as e:
            logger.exception("Error loading DeepSeek-OCR model %s: %s", self.model_id, e)
            raise

    def generate_caption(self, image: Image.Image, prompt: str = None, parameters: dict = None) -> str:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            image = self._ensure_rgb(image)

            # Use default OCR prompt if none provided
            if not prompt or not prompt.strip():
                prompt = "<|grounding|>Convert the document to markdown."
            else:
                prompt = prompt.strip()

            # Ensure <image> token is present (required by DeepSeek-OCR)
            if "<image>" not in prompt:
                prompt = "<image>\n" + prompt

            # Extract DeepSeek-OCR specific parameters
            base_size = parameters.get('base_size', 1024) if parameters else 1024
            image_size = parameters.get('image_size', 640) if parameters else 640
            crop_mode = parameters.get('crop_mode', True) if parameters else True
            test_compress = parameters.get('test_compress', True) if parameters else True

            # Save image temporarily (DeepSeek-OCR infer() requires file path)
            import tempfile
            import os

            # Create temporary directory for input and output
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_img_path = os.path.join(tmp_dir, 'input.png')
                image.save(tmp_img_path, format='PNG')

                output_dir = os.path.join(tmp_dir, 'output')
                os.makedirs(output_dir, exist_ok=True)

                # Run inference using DeepSeek-OCR's infer method
                # eval_mode=True is required to get the text output returned
                result = self.model.infer(
                    self.tokenizer,
                    prompt=prompt,
                    image_file=tmp_img_path,
                    output_path=output_dir,
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode,
                    save_results=False,  # Don't save to disk
                    test_compress=test_compress,
                    eval_mode=True,  # Required to return the output text
                )

                # Convert result to string
                caption = str(result).strip() if result else "Unable to extract text."
                return caption

        except Exception as e:
            logger.exception("Error generating caption with DeepSeek-OCR: %s", e)
            return f"Error: {str(e)}"

    def generate_captions_batch(self, images: list, prompts: list = None, parameters: dict = None) -> list:
        """Generate captions for multiple images (processes sequentially)"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Ensure we have prompts for all images
            if prompts is None:
                prompts = ["<|grounding|>Convert the document to markdown."] * len(images)
            elif len(prompts) != len(images):
                raise ValueError(f"Number of prompts ({len(prompts)}) must match number of images ({len(images)})")

            # Convert all images to RGB
            processed_images = self._ensure_rgb(images)

            # Process each image one at a time (DeepSeek-OCR processes images individually)
            results = []
            for image, prompt in zip(processed_images, prompts):
                caption = self.generate_caption(image, prompt, parameters)
                results.append(caption)

            return results

        except Exception as e:
            logger.exception("Error generating captions in batch with DeepSeek-OCR: %s", e)
            return self._format_batch_error(e, len(images))

    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def get_available_parameters(self) -> list:
        return [
            {
                "name": "Base Size",
                "param_key": "base_size",
                "type": "number",
                "min": 512,
                "max": 2048,
                "step": 64,
                "description": "Base resolution for image processing (default: 1024)"
            },
            {
                "name": "Image Size",
                "param_key": "image_size",
                "type": "number",
                "min": 320,
                "max": 1024,
                "step": 32,
                "description": "Target image size for inference (default: 640)"
            },
            {
                "name": "Crop Mode",
                "param_key": "crop_mode",
                "type": "checkbox",
                "description": "Enable intelligent cropping for better OCR results (default: enabled)"
            },
            {
                "name": "Test Compress",
                "param_key": "test_compress",
                "type": "checkbox",
                "description": "Memory-friendly compression mode (default: enabled)"
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
                "max": 8,
                "step": 1,
                "description": "Number of images to process simultaneously (higher = faster but more VRAM)"
            }
        ]

    def unload(self) -> None:
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if hasattr(self, 'current_precision_params'):
            delattr(self, 'current_precision_params')
        if hasattr(self, 'quantization_config'):
            self.quantization_config = None
        super().unload()
