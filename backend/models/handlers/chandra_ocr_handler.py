"""
Chandra OCR Model Handler
Custom handler for Chandra OCR model which uses the chandra package's generate_hf function.
"""
import torch
from PIL import Image
from typing import Dict, List, Optional
import logging
from .hf_ocr_handler import HuggingFaceOCRHandler

logger = logging.getLogger(__name__)


class ChandraOCRHandler(HuggingFaceOCRHandler):
    """
    Custom handler for Chandra OCR model which uses the chandra package's generate_hf function.
    """

    # Generation defaults
    DEFAULT_MAX_OUTPUT_TOKENS = 512

    def _pre_load_hook(self, precision: str = None, **kwargs) -> str:
        """Validate that Chandra only uses float precisions (no quantization)."""
        precision = precision or self.config.get('default_precision', 'bfloat16')
        supported_precisions = ["float32", "float16", "bfloat16"]
        if precision not in supported_precisions:
            logger.warning("Chandra OCR only supports float precisions. Falling back to bfloat16.")
            precision = "bfloat16"
        return precision

    def _prepare_model_kwargs(self, precision: str, use_flash_attention: bool) -> dict:
        """Prepare model kwargs without quantization support."""
        model_kwargs = self.config.get('model_config', {}).copy()
        model_kwargs['torch_dtype'] = self._get_dtype(precision)

        # Setup flash attention if requested
        if use_flash_attention:
            self._setup_flash_attention(model_kwargs, precision)

        return model_kwargs

    def _post_load_hook(self):
        """Configure Chandra-specific generation settings."""
        # Attach processor to model (required by Chandra's generate_hf)
        self.model.processor = self.processor

        # Configure generation settings for deterministic OCR output
        if self.processor.tokenizer.pad_token_id is not None:
            self.model.generation_config.pad_token_id = self.processor.tokenizer.pad_token_id

        # Use greedy decoding for deterministic OCR
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

    def infer_single(self, image: Image.Image, prompt: Optional[str] = None,
                    parameters: Optional[Dict] = None) -> str:
        """Extract text using Chandra OCR with generate_hf."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")

        try:
            from chandra.model.hf import generate_hf
            from chandra.model.schema import BatchInputItem

            logger.debug("Chandra single image processing")

            image = self._ensure_rgb(image)

            # Extract parameters
            params = parameters or {}
            max_output_tokens = params.get('max_new_tokens', self.DEFAULT_MAX_OUTPUT_TOKENS)
            chandra_preset = params.get('chandra_preset', 'user_prompt')

            # Create BatchInputItem based on preset selection
            if chandra_preset == "ocr_layout":
                batch_item = BatchInputItem(image=image, prompt_type="ocr_layout")
            elif chandra_preset == "ocr":
                batch_item = BatchInputItem(image=image, prompt_type="ocr")
            else:  # user_prompt (default)
                if prompt:
                    batch_item = BatchInputItem(image=image, prompt=prompt)
                else:
                    batch_item = BatchInputItem(image=image, prompt_type="ocr_layout")

            # Generate using Chandra's generate_hf
            with torch.inference_mode():
                results = generate_hf([batch_item], self.model, max_output_tokens=max_output_tokens)

            # Extract raw text from GenerationResult
            caption = results[0].raw if results and hasattr(results[0], 'raw') else ""
            return caption.strip() if caption else "Unable to generate description."

        except Exception as e:
            logger.exception("Error with Chandra OCR: %s", e)
            return f"Error: {str(e)}"

    def infer_batch(self, images: List[Image.Image], prompts: Optional[List[str]] = None,
                   parameters: Optional[Dict] = None) -> List[str]:
        """
        Extract text from multiple images.
        Note: Chandra processes images sequentially internally, so we process one at a time.
        """
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")

        try:
            logger.info(f"Chandra processing {len(images)} images sequentially")

            # Prepare prompts
            if prompts is None:
                prompts = [None] * len(images)
            elif len(prompts) != len(images):
                raise ValueError(f"Number of prompts ({len(prompts)}) must match number of images ({len(images)})")

            # Process each image individually (Chandra doesn't benefit from batching)
            results = []
            for image, prompt in zip(images, prompts):
                result = self.infer_single(image, prompt, parameters)
                results.append(result)

            return results

        except Exception as e:
            logger.exception("Error in Chandra OCR processing: %s", e)
            error_msg = f"Error: {str(e)}"
            return [error_msg] * len(images)
