"""
LFM2 Model Handler
Custom handler for LFM2 model which requires chat template format.
"""
import logging
from .hf_vlm_handler import HuggingFaceVLMHandler
from .inference_strategies import LFM2Strategy, AssistantResponseStrategy

logger = logging.getLogger(__name__)


class LFM2Handler(HuggingFaceVLMHandler):
    """
    Custom handler for LFM2 model which requires chat template format.
    """

    def _get_prompt_strategy(self):
        """Use LFM2 chat template formatting strategy."""
        return LFM2Strategy()

    def _get_response_strategy(self):
        """Use assistant response extraction strategy."""
        return AssistantResponseStrategy()
