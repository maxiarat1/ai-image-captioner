"""
LLaVA-Phi Model Handler
Custom handler for LLaVA-Phi-3 model which requires specific prompt formatting
and keyword arguments for processor.
"""
import logging
from .hf_vlm_handler import HuggingFaceVLMHandler
from .inference_strategies import LlavaPhiStrategy, SkipInputTokensStrategy

logger = logging.getLogger(__name__)


class LlavaPhiHandler(HuggingFaceVLMHandler):
    """
    Custom handler for LLaVA-Phi-3 model which requires specific prompt formatting
    and keyword arguments for processor.
    """

    # Processor defaults
    DEFAULT_PATCH_SIZE = 14

    def _get_prompt_strategy(self):
        """Use LLaVA-Phi-3 specific prompt formatting strategy."""
        return LlavaPhiStrategy()

    def _get_response_strategy(self):
        """Use skip input tokens strategy for clean response extraction."""
        return SkipInputTokensStrategy()

    def _post_load_hook(self):
        """Fix LLaVA processor bug: patch_size not set."""
        super()._post_load_hook()

        # Fix patch_size if needed
        if hasattr(self.processor, 'patch_size') and self.processor.patch_size is None:
            if hasattr(self.processor, 'image_processor') and hasattr(self.processor.image_processor, 'patch_size'):
                self.processor.patch_size = self.processor.image_processor.patch_size
                logger.info("Set patch_size from image_processor: %s", self.processor.patch_size)
            else:
                # Default patch size for LLaVA models
                self.processor.patch_size = self.DEFAULT_PATCH_SIZE
                logger.warning("patch_size not found in config, using default: %d", self.DEFAULT_PATCH_SIZE)
