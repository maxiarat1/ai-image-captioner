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
from .custom_handlers import JanusHandler, R4BHandler, TrOCRHandler, LFM2Handler, LlavaPhiHandler, NanonetsOCRHandler, ChandraOCRHandler

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
