import torch
from utils.torch_utils import pick_device
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from .base_adapter import BaseModelAdapter
import logging

logger = logging.getLogger(__name__)

class BlipAdapter(BaseModelAdapter):
    # Parameters that should not be passed to model.generate()
    SPECIAL_PARAMS = {'batch_size'}

    def __init__(self):
        super().__init__("blip-image-captioning-base")
        self.device = pick_device(torch)

    def load_model(self) -> None:
        try:
            logger.info("Loading BLIP model on %s…", self.device)
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model.to(self.device)
            self.model.eval()

            # Setup pad token for batch processing
            self._setup_pad_token()

            logger.info("BLIP model loaded successfully")
        except Exception as e:
            logger.exception("Error loading BLIP model: %s", e)
            raise

    def generate_caption(self, image: Image.Image, prompt: str = None, parameters: dict = None) -> str:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            image = self._ensure_rgb(image)

            if prompt and prompt.strip():
                inputs = self.processor(image, prompt.strip(), return_tensors="pt").to(self.device)
            else:
                inputs = self.processor(image, return_tensors="pt").to(self.device)

            gen_params = self._filter_generation_params(parameters, self.SPECIAL_PARAMS)

            logger.debug("BLIP params: %s", gen_params if gen_params else "defaults")

            with torch.no_grad():
                out = self.model.generate(**inputs, **gen_params)

            return self.processor.decode(out[0], skip_special_tokens=True)

        except Exception as e:
            logger.exception("Error generating caption: %s", e)
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

            # Build generation parameters
            gen_params = self._filter_generation_params(parameters, self.SPECIAL_PARAMS)

            # Process batch with padding
            # For BLIP, we can process with or without prompts
            if prompts[0] and prompts[0].strip():
                # With prompts
                text_prompts = [p.strip() if p else "" for p in prompts]
                inputs = self.processor(
                    images=processed_images,
                    text=text_prompts,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
            else:
                # Without prompts (unconditional generation)
                inputs = self.processor(
                    images=processed_images,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)

            # Generate for batch
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **gen_params)

            # Batch decode
            captions = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )

            return [caption.strip() if caption else "Unable to generate description." for caption in captions]

        except Exception as e:
            logger.exception("Error generating captions in batch with BLIP: %s", e)
            return [f"Error: {str(e)}"] * len(images)

    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def get_available_parameters(self) -> list:
        return [
            {
                "name": "Max Length",
                "param_key": "max_length",
                "type": "number",
                "min": 1,
                "max": 500,
                "step": 1,
                "description": "Maximum number of tokens to generate"
            },
            {
                "name": "Min Length",
                "param_key": "min_length",
                "type": "number",
                "min": 0,
                "max": 100,
                "step": 1,
                "description": "Minimum number of tokens to generate"
            },
            {
                "name": "Num Beams",
                "param_key": "num_beams",
                "type": "number",
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Number of beams for beam search"
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
                "name": "Repetition Penalty",
                "param_key": "repetition_penalty",
                "type": "number",
                "min": 1,
                "max": 2,
                "step": 0.1,
                "description": "Penalty for repeating tokens"
            },
            {
                "name": "Length Penalty",
                "param_key": "length_penalty",
                "type": "number",
                "min": 0,
                "max": 2,
                "step": 0.1,
                "description": "Exponential penalty to the length for beam search"
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