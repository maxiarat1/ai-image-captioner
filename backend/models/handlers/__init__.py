"""
Model Handlers Package
Reusable handlers for different model types to eliminate adapter redundancy.
"""
from .base_handler import BaseModelHandler
from .hf_vlm_handler import HuggingFaceVLMHandler
from .hf_tagger_handler import HuggingFaceTaggerHandler
from .hf_ocr_handler import HuggingFaceOCRHandler
from .hf_classifier_handler import HuggingFaceClassifierHandler
from .onnx_tagger_handler import ONNXTaggerHandler
from .janus_handler import JanusHandler
from .r4b_handler import R4BHandler
from .trocr_handler import TrOCRHandler
from .lfm2_handler import LFM2Handler
from .llava_phi_handler import LlavaPhiHandler
from .nanonets_ocr_handler import NanonetsOCRHandler
from .chandra_ocr_handler import ChandraOCRHandler

__all__ = [
    'BaseModelHandler',
    'HuggingFaceVLMHandler',
    'HuggingFaceTaggerHandler',
    'HuggingFaceOCRHandler',
    'HuggingFaceClassifierHandler',
    'ONNXTaggerHandler',
    'JanusHandler',
    'R4BHandler',
    'TrOCRHandler',
    'LFM2Handler',
    'LlavaPhiHandler',
    'NanonetsOCRHandler',
    'ChandraOCRHandler',
]
