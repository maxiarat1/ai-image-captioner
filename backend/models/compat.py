"""
Backwards Compatibility Layer
This module provides compatibility imports for old adapter class names.
All adapters now use the unified adapter + handler system.

DEPRECATED: Import from models.adapter_factory or use the registry instead.
"""
import warnings
from models.adapter_factory import ModelAdapterFactory

# Create factory instance
_factory = ModelAdapterFactory()


def _create_deprecated_adapter_class(model_key: str, class_name: str):
    """
    Create a deprecated adapter class that wraps the new unified adapter.
    
    This maintains backwards compatibility for any code that imports
    specific adapter classes directly.
    """
    def __init__(self, **kwargs):
        warnings.warn(
            f"{class_name} is deprecated. Use ModelAdapterFactory.create_adapter('{model_key}') instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Create the unified adapter
        adapter = _factory.create_adapter(model_key, **kwargs)
        # Copy all attributes to self
        self.__dict__.update(adapter.__dict__)
        self.__class__ = adapter.__class__
    
    # Create a class dynamically
    return type(class_name, (), {'__init__': __init__})


# Create deprecated classes for backwards compatibility
BlipAdapter = _create_deprecated_adapter_class('blip', 'BlipAdapter')
Blip2Adapter = _create_deprecated_adapter_class('blip2', 'Blip2Adapter')
R4BAdapter = _create_deprecated_adapter_class('r4b', 'R4BAdapter')
WdVitAdapter = lambda **kwargs: _factory.create_adapter(
    'wdvit' if 'model_id' not in kwargs or 'vit' in kwargs['model_id'] else 'wdeva02',
    **kwargs
)
JanusAdapter = lambda **kwargs: _factory.create_adapter(
    'janus-1.3b' if 'model_id' not in kwargs else (
        'janus-1.3b' if '1.3B' in kwargs['model_id'] else
        'janus-pro-1b' if 'Pro-1B' in kwargs['model_id'] else
        'janus-pro-7b'
    ),
    **kwargs
)
NanonetsOCRAdapter = _create_deprecated_adapter_class('nanonets-ocr-s', 'NanonetsOCRAdapter')
ChandraAdapter = _create_deprecated_adapter_class('chandra-ocr', 'ChandraAdapter')
TrOCRAdapter = _create_deprecated_adapter_class('trocr-large-printed', 'TrOCRAdapter')
LlavaPhiAdapter = _create_deprecated_adapter_class('llava-phi3', 'LlavaPhiAdapter')
LFM2Adapter = _create_deprecated_adapter_class('lfm2-vl-3b', 'LFM2Adapter')
Wd14ConvNextAdapter = _create_deprecated_adapter_class('wd14-convnext', 'Wd14ConvNextAdapter')
VitClassifierAdapter = _create_deprecated_adapter_class('vit-classifier', 'VitClassifierAdapter')

__all__ = [
    'BlipAdapter',
    'Blip2Adapter',
    'R4BAdapter',
    'WdVitAdapter',
    'JanusAdapter',
    'NanonetsOCRAdapter',
    'ChandraAdapter',
    'TrOCRAdapter',
    'LlavaPhiAdapter',
    'LFM2Adapter',
    'Wd14ConvNextAdapter',
    'VitClassifierAdapter',
]
