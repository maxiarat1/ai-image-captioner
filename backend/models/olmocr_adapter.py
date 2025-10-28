import torch
import logging
import base64
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from .base_adapter import BaseModelAdapter

logger = logging.getLogger(__name__)

class OlmOCRAdapter(BaseModelAdapter):
    # Parameters that should not be passed to model.generate()
    SPECIAL_PARAMS = {'precision', 'use_flash_attention', 'batch_size'}

    def __init__(self, model_id: str = "allenai/olmOCR-2-7B-1025"):
        """
        Initialize olmOCR adapter.

        This model is specialized for OCR and document text extraction,
        capable of extracting structured text from images and documents.
        """
        super().__init__(model_id)
        self.model_id = model_id
        self.processor_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        self.device = self._init_device(torch)
        self.quantization_config = None

    def _build_default_ocr_prompt(self) -> str:
        """Build the default OCR prompt for text extraction"""
        try:
            from olmocr.prompts import build_no_anchoring_v4_yaml_prompt
            return build_no_anchoring_v4_yaml_prompt()
        except ImportError:
            logger.warning("olmocr library not available, using basic OCR prompt")
            # Fallback prompt if olmocr is not installed
            return "Extract all text from this image. Preserve the layout and formatting."

    def _extract_text_from_yaml(self, output: str) -> str:
        """
        Extract clean text from YAML-formatted OCR output.

        The model outputs structured YAML with metadata. This extracts just the text content.
        """
        if not output:
            return output

        # If output doesn't look like YAML, return as-is
        if not output.strip().startswith('---'):
            return output.strip()

        # Extract text after the YAML frontmatter
        lines = output.split('\n')
        in_frontmatter = False
        text_lines = []

        for line in lines:
            if line.strip() == '---':
                if not in_frontmatter:
                    in_frontmatter = True
                else:
                    in_frontmatter = False
                continue

            if not in_frontmatter and line.strip():
                text_lines.append(line)

        # If we extracted text, return it; otherwise return original
        extracted = '\n'.join(text_lines).strip()
        return extracted if extracted else output.strip()

    def load_model(self, precision="bfloat16", use_flash_attention=False) -> None:
        try:
            logger.info("Loading olmOCR model %s on %s with %s precision…",
                       self.model_id, self.device, precision)

            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.processor_id,
                trust_remote_code=True,
                use_fast=True
            )

            # Setup pad token for batch processing
            self._setup_pad_token()

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
                    model_kwargs["dtype"] = dtype

            # Setup flash attention if requested
            if use_flash_attention:
                self._setup_flash_attention(
                    model_kwargs, precision, force_bfloat16=True
                )

            # Load model
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                **model_kwargs
            )

            # Move to device if not using quantization
            if precision not in ["4bit", "8bit"]:
                self.model = self.model.to(self.device)

            self.model.eval()
            logger.info("olmOCR model loaded successfully (Precision: %s)", precision)

        except Exception as e:
            logger.exception("Error loading olmOCR model %s: %s", self.model_id, e)
            raise

    def generate_caption(self, image: Image.Image, prompt: str = None, parameters: dict = None) -> str:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            image = self._ensure_rgb(image)

            # Use default OCR prompt if none provided
            if not prompt or not prompt.strip():
                prompt = self._build_default_ocr_prompt()
            else:
                prompt = prompt.strip()

            # Convert image to base64 for the chat template
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Build messages in the format expected by the processor
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    ],
                }
            ]

            # Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Process inputs
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )

            # Move inputs to device with proper dtype
            model_dtype = next(self.model.parameters()).dtype
            inputs = self._move_inputs_to_device(inputs, self.device, model_dtype)

            # Build generation parameters
            gen_params = self._filter_generation_params(parameters, self.SPECIAL_PARAMS)

            # Set defaults if not provided
            if 'temperature' not in gen_params:
                gen_params['temperature'] = 0.1
            if 'max_new_tokens' not in gen_params:
                gen_params['max_new_tokens'] = 512
            if 'do_sample' not in gen_params:
                gen_params['do_sample'] = True

            logger.debug("olmOCR params: %s", gen_params)

            # Generate output
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_params)

            # Decode output (skip input tokens)
            prompt_length = inputs["input_ids"].shape[1]
            new_tokens = output_ids[:, prompt_length:]
            text_output = self.processor.tokenizer.batch_decode(
                new_tokens,
                skip_special_tokens=True
            )[0]

            # Extract clean text from YAML format if applicable
            caption = self._extract_text_from_yaml(text_output)
            return caption if caption else "Unable to extract text."

        except Exception as e:
            logger.exception("Error generating caption with olmOCR: %s", e)
            return f"Error: {str(e)}"

    def generate_captions_batch(self, images: list, prompts: list = None, parameters: dict = None) -> list:
        """Generate captions for multiple images using batch processing"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Ensure we have prompts for all images
            if prompts is None:
                default_prompt = self._build_default_ocr_prompt()
                prompts = [default_prompt] * len(images)
            elif len(prompts) != len(images):
                raise ValueError(f"Number of prompts ({len(prompts)}) must match number of images ({len(images)})")

            # Convert all images to RGB
            processed_images = self._ensure_rgb(images)

            # Build batch messages and texts
            batch_texts = []
            batch_images = []

            for i, image in enumerate(processed_images):
                prompt = prompts[i].strip() if prompts[i] else self._build_default_ocr_prompt()

                # Convert image to base64
                buffer = BytesIO()
                image.save(buffer, format='PNG')
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                        ],
                    }
                ]

                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                batch_texts.append(text)
                batch_images.append(image)

            # Process batch with padding
            inputs = self.processor(
                text=batch_texts,
                images=batch_images,
                padding=True,
                return_tensors="pt",
            )

            # Move inputs to device with proper dtype
            model_dtype = next(self.model.parameters()).dtype
            inputs = self._move_inputs_to_device(inputs, self.device, model_dtype)

            # Build generation parameters
            gen_params = self._filter_generation_params(parameters, self.SPECIAL_PARAMS)

            # Set defaults if not provided
            if 'temperature' not in gen_params:
                gen_params['temperature'] = 0.1
            if 'max_new_tokens' not in gen_params:
                gen_params['max_new_tokens'] = 512
            if 'do_sample' not in gen_params:
                gen_params['do_sample'] = True

            # Generate for batch
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **gen_params)

            # Trim input tokens from outputs
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]

            # Batch decode
            text_outputs = self.processor.tokenizer.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True
            )

            # Extract clean text from each output
            results = []
            for text_output in text_outputs:
                caption = self._extract_text_from_yaml(text_output)
                results.append(caption if caption else "Unable to extract text.")

            return results

        except Exception as e:
            logger.exception("Error generating captions in batch with olmOCR: %s", e)
            return self._format_batch_error(e, len(images))

    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def get_available_parameters(self) -> list:
        return [
            {
                "name": "Temperature",
                "param_key": "temperature",
                "type": "number",
                "min": 0,
                "max": 2,
                "step": 0.1,
                "description": "Sampling temperature for randomness (default: 0.1)"
            },
            {
                "name": "Max New Tokens",
                "param_key": "max_new_tokens",
                "type": "number",
                "min": 1,
                "max": 2048,
                "step": 1,
                "description": "Maximum number of tokens to generate (default: 512)"
            },
            {
                "name": "Do Sample",
                "param_key": "do_sample",
                "type": "checkbox",
                "description": "Enable sampling instead of greedy decoding (default: enabled)"
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
        if hasattr(self, 'current_precision_params'):
            delattr(self, 'current_precision_params')
        if hasattr(self, 'quantization_config'):
            self.quantization_config = None
        super().unload()
