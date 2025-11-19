"""
R4B Model Handler
Custom handler for R4B model with thinking mode support.
R4B only supports float32, 4bit, and 8bit precision modes.
Uses R4B thinking strategy for prompts and thinking tags extraction for responses.
"""
from PIL import Image
from typing import Dict, List, Optional
import logging
from .hf_vlm_handler import HuggingFaceVLMHandler
from .inference_strategies import R4BThinkingStrategy, ThinkingTagsStrategy

logger = logging.getLogger(__name__)


class R4BHandler(HuggingFaceVLMHandler):
    """
    Custom handler for R4B model with thinking mode support.
    R4B only supports float32, 4bit, and 8bit precision modes.
    Uses R4B thinking strategy for prompts and thinking tags extraction for responses.
    """

    def _get_prompt_strategy(self):
        """Use R4B thinking strategy."""
        return R4BThinkingStrategy()

    def _get_response_strategy(self):
        """Use thinking tags extraction strategy."""
        return ThinkingTagsStrategy()

    def infer_single(self, image: Image.Image, prompt: Optional[str] = None,
                    parameters: Optional[Dict] = None) -> str:
        """Generate caption with R4B thinking mode support using strategies."""
        # Extract and set thinking mode in strategy
        thinking_mode = self._extract_thinking_mode(parameters)
        self.prompt_strategy.set_thinking_mode(thinking_mode)

        # Use parent's strategy-based inference
        return super().infer_single(image, prompt, parameters)

    def infer_batch(self, images: List[Image.Image], prompts: Optional[List[str]] = None,
                   parameters: Optional[Dict] = None) -> List[str]:
        """Generate captions for multiple images with R4B thinking mode."""
        # Extract and set thinking mode in strategy
        thinking_mode = self._extract_thinking_mode(parameters)
        self.prompt_strategy.set_thinking_mode(thinking_mode)

        # Use parent's strategy-based inference
        return super().infer_batch(images, prompts, parameters)

    def _extract_thinking_mode(self, parameters: Optional[Dict]) -> str:
        """Extract and validate thinking_mode parameter."""
        if not parameters:
            return 'auto'
        thinking_mode = parameters.get('thinking_mode', 'auto')
        return thinking_mode if thinking_mode in ['auto', 'short', 'long'] else 'auto'
