"""
Janus Model Handler
Custom handler for Janus models (requires janus package).
Uses conversation strategy for prompts and Janus response extraction.
"""
import torch
from PIL import Image
from typing import Dict, List, Optional
import logging
from .hf_vlm_handler import HuggingFaceVLMHandler
from .inference_strategies import ConversationStrategy, JanusResponseStrategy

logger = logging.getLogger(__name__)


class JanusHandler(HuggingFaceVLMHandler):
    """
    Custom handler for Janus models (requires janus package).
    Uses conversation strategy for prompts and Janus response extraction.
    """

    # Generation defaults
    DEFAULT_MAX_NEW_TOKENS = 512

    def _get_prompt_strategy(self):
        """Use conversation strategy for Janus."""
        return ConversationStrategy()

    def _get_response_strategy(self):
        """Use Janus-specific response extraction."""
        return JanusResponseStrategy()

    def _get_processor_class(self):
        """Import VLChatProcessor from janus package for Janus models."""
        try:
            from janus.models import VLChatProcessor
            return VLChatProcessor
        except ImportError:
            logger.error(
                "Janus models require the 'janus' package. "
                "Install it via: pip install git+https://github.com/deepseek-ai/Janus.git"
            )
            raise

    def _get_model_class(self):
        """Use AutoModelForCausalLM for Janus."""
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM

    def _load_processor(self, **kwargs):
        """Load Janus-specific processor with explicit fast image processor class."""
        from transformers import AutoImageProcessor

        processor_class = self._get_processor_class()
        self.processor = processor_class.from_pretrained(
            self.model_id,
            fast_image_processor_class=AutoImageProcessor
        )

    def _build_conversation(self, prompt: Optional[str]) -> List[Dict]:
        """Build Janus conversation format."""
        prompt_text = prompt.strip() if prompt and prompt.strip() else "Describe this image in detail."
        return [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{prompt_text}",
                "images": [],  # Will be set by caller
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

    def _prepare_generation_params(self, parameters: Optional[Dict]) -> Dict:
        """Prepare generation parameters with Janus defaults."""
        gen_params = self._filter_generation_params(parameters)
        gen_params = self._sanitize_generation_params(gen_params)

        # Set default generation parameters
        default_params = {
            'pad_token_id': self.processor.tokenizer.eos_token_id,
            'bos_token_id': self.processor.tokenizer.bos_token_id,
            'eos_token_id': self.processor.tokenizer.eos_token_id,
            'max_new_tokens': self.DEFAULT_MAX_NEW_TOKENS,
            'do_sample': False,
            'use_cache': True
        }

        # Merge with user parameters (user params take precedence)
        return {**default_params, **gen_params}

    def _generate_for_image(self, image: Image.Image, prompt: Optional[str],
                           parameters: Optional[Dict]) -> str:
        """Generate caption for a single image using Janus-specific generation."""
        image = self._ensure_rgb(image)

        # Build conversation with image
        conversation = self._build_conversation(prompt)
        conversation[0]["images"] = [image]

        # Prepare inputs using VLChatProcessor
        prepare_inputs = self.processor(
            conversations=conversation,
            images=[image],
            force_batchify=True
        )

        # Move inputs to device and match model dtype
        prepare_inputs = self._move_inputs_to_device(prepare_inputs, match_model_dtype=True)

        # Prepare generation parameters
        final_params = self._prepare_generation_params(parameters)

        # Generate using Janus-specific method
        with torch.no_grad():
            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
            outputs = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                **final_params
            )

        # Decode
        result = self.processor.tokenizer.decode(
            outputs[0].cpu().tolist(),
            skip_special_tokens=True
        )

        return result.strip() if result else "Unable to generate description."

    def infer_single(self, image: Image.Image, prompt: Optional[str] = None,
                    parameters: Optional[Dict] = None) -> str:
        """Generate caption using Janus-specific conversation format."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")

        try:
            return self._generate_for_image(image, prompt, parameters)
        except Exception as e:
            logger.exception("Error with Janus inference: %s", e)
            return f"Error: {str(e)}"

    def infer_batch(self, images: List[Image.Image], prompts: Optional[List[str]] = None,
                   parameters: Optional[Dict] = None) -> List[str]:
        """Generate captions for multiple images using Janus conversation format."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")

        try:
            # Prepare prompts
            if prompts is None:
                prompts = [None] * len(images)
            elif len(prompts) != len(images):
                raise ValueError(f"Number of prompts ({len(prompts)}) must match number of images ({len(images)})")

            # Process each image separately as Janus processor doesn't support true batching
            results = []
            for image, prompt in zip(images, prompts):
                result = self._generate_for_image(image, prompt, parameters)
                results.append(result)

            return results

        except Exception as e:
            logger.exception("Error in batch Janus generation: %s", e)
            error_msg = f"Error: {str(e)}"
            return [error_msg] * len(images)
