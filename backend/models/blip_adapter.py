import torch
from utils.torch_utils import pick_device
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from .base_adapter import BaseModelAdapter
import logging

logger = logging.getLogger(__name__)

class BlipAdapter(BaseModelAdapter):
    """BLIP model adapter for image captioning"""

    def __init__(self):
        super().__init__("blip-image-captioning-base")
        self.device = pick_device(torch)

    def load_model(self) -> None:
        """Load BLIP model and processor"""
        try:
            logger.info("Loading BLIP model on %s…", self.device)
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model.to(self.device)
            self.model.eval()
            logger.info("BLIP model loaded successfully")
        except Exception as e:
            logger.exception("Error loading BLIP model: %s", e)
            raise

    def generate_caption(self, image: Image.Image, prompt: str = None, parameters: dict = None) -> str:
        """Generate caption for the given image with configurable parameters"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Debug logging
            logger.debug("BLIP.generate_caption | prompt=%s | params=%s", (prompt or ""), parameters)

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Process image with optional text prompt
            if prompt and prompt.strip():
                # Use text-guided image captioning
                logger.debug("Using text-guided captioning")
                inputs = self.processor(image, prompt.strip(), return_tensors="pt").to(self.device)
            else:
                # Use unconditional image captioning
                logger.debug("Using unconditional captioning (no prompt)")
                inputs = self.processor(image, return_tensors="pt").to(self.device)

            # Build generation parameters - only include what user explicitly provided
            gen_params = {}
            if parameters:
                # Get all supported param keys
                supported_params = [p['param_key'] for p in self.get_available_parameters()]

                # Only pass parameters that were explicitly set by the user
                for param_key in supported_params:
                    if param_key in parameters:
                        gen_params[param_key] = parameters[param_key]

            logger.debug("Generation parameters: %s", gen_params if gen_params else "defaults")

            # Generate caption - library will use its own defaults for unspecified params
            with torch.no_grad():
                out = self.model.generate(**inputs, **gen_params)

            # Decode caption
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption

        except Exception as e:
            logger.exception("Error generating caption: %s", e)
            return f"Error: {str(e)}"

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None and self.processor is not None

    def get_available_parameters(self) -> list:
        """Get list of available parameters for BLIP model"""
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
            }
        ]