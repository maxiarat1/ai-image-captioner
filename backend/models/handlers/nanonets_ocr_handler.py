"""
Nanonets OCR Model Handler
Custom handler for Nanonets OCR model which requires chat template format
with system and user messages.
"""
import torch
from PIL import Image
from typing import Dict, List, Optional
import logging
from .hf_ocr_handler import HuggingFaceOCRHandler

logger = logging.getLogger(__name__)


class NanonetsOCRHandler(HuggingFaceOCRHandler):
    """
    Custom handler for Nanonets OCR model which requires chat template format
    with system and user messages.
    """

    def _default_prompt(self) -> str:
        """Default OCR prompt for comprehensive text extraction."""
        return (
            "Extract the text from the above document as if you were reading it naturally.\n"
            "Return the tables in HTML format. Return the equations in LaTeX representation.\n"
            "If there is an image in the document and an image caption is not present, add a small description\n"
            "of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>.\n"
            "Watermarks should be wrapped in <watermark></watermark>.\n"
            "Page numbers should be wrapped in <page_number></page_number>.\n"
            "Prefer using ☐ and ☑ for check boxes."
        )

    def infer_single(self, image: Image.Image, prompt: Optional[str] = None,
                    parameters: Optional[Dict] = None) -> str:
        """Extract text using Nanonets OCR with chat template format."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")

        try:
            image = self._ensure_rgb(image)

            # Use default OCR prompt if none provided
            prompt_text = (prompt or self._default_prompt()).strip()

            # Build chat-style messages (required by Nanonets processor)
            messages = [
                {"role": "system", "content": "You are a helpful OCR assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_text},
                    ],
                },
            ]

            # Apply chat template to get formatted text
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process with both text and images
            inputs = self.processor(
                text=[text], images=[image], padding=True, return_tensors="pt"
            )

            # Move to device and match model dtype
            inputs = self._move_inputs_to_device(inputs, match_model_dtype=True)

            # Prepare generation parameters
            gen_params = self._filter_generation_params(parameters)
            gen_params.setdefault("do_sample", False)  # Deterministic by default for OCR
            gen_params = self._sanitize_generation_params(gen_params)

            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_params)

            # Skip prompt tokens and decode only the generated response
            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = output_ids[:, prompt_len:]
            text_output = self.processor.tokenizer.batch_decode(
                new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]

            return text_output.strip() if text_output else ""

        except Exception as e:
            logger.exception("Error with Nanonets OCR: %s", e)
            return f"Error: {str(e)}"

    def infer_batch(self, images: List[Image.Image], prompts: Optional[List[str]] = None,
                   parameters: Optional[Dict] = None) -> List[str]:
        """Extract text from multiple images using batch processing."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")

        try:
            # Ensure all images are RGB
            images = [self._ensure_rgb(img) for img in images]

            # Prepare prompts (use default if not provided)
            if prompts is None:
                default_prompt = self._default_prompt()
                prompts = [default_prompt] * len(images)
            elif len(prompts) != len(images):
                raise ValueError(f"Number of prompts ({len(prompts)}) must match number of images ({len(images)})")

            # Build chat messages for each image
            texts = []
            for i, (image, prompt_text) in enumerate(zip(images, prompts)):
                prompt_text = (prompt_text or self._default_prompt()).strip()

                messages = [
                    {"role": "system", "content": "You are a helpful OCR assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt_text},
                        ],
                    },
                ]

                # Apply chat template
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                texts.append(text)

            # Process batch with both text and images
            inputs = self.processor(
                text=texts, images=images, padding=True, return_tensors="pt"
            )

            # Move to device and match model dtype
            inputs = self._move_inputs_to_device(inputs, match_model_dtype=True)

            # Prepare generation parameters
            gen_params = self._filter_generation_params(parameters)
            gen_params.setdefault("do_sample", False)  # Deterministic by default for OCR
            gen_params = self._sanitize_generation_params(gen_params)

            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_params)

            # Skip prompt tokens and decode each result
            prompt_len = inputs["input_ids"].shape[1]
            results = []
            for output in output_ids:
                new_tokens = output[prompt_len:]
                text_output = self.processor.tokenizer.decode(
                    new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                results.append(text_output.strip() if text_output else "")

            return results

        except Exception as e:
            logger.exception("Error in batch Nanonets OCR: %s", e)
            error_msg = f"Error: {str(e)}"
            return [error_msg] * len(images)
