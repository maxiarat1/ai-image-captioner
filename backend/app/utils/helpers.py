"""
Helper utilities for model parameters and image processing.
"""
import io
from typing import Optional, Dict
from pathlib import Path
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from config import PRECISION_DEFAULTS


def extract_precision_params(model_name: str, parameters: Optional[Dict]) -> Optional[Dict[str, any]]:
    """
    Extract precision parameters for a model, falling back to defaults.

    Args:
        model_name: Name of the model
        parameters: User-provided parameters dict

    Returns:
        Dict with precision and use_flash_attention, or None if not applicable
    """
    # Return None if no parameters provided or model doesn't have precision defaults
    if not parameters or model_name not in PRECISION_DEFAULTS:
        return None

    defaults = PRECISION_DEFAULTS[model_name]

    # Extract only precision-related parameters
    return {
        'precision': parameters.get('precision', defaults['precision']),
        'use_flash_attention': parameters.get('use_flash_attention', defaults['use_flash_attention'])
    }


def embed_caption_in_image(image: Image.Image, caption: str, filename: str) -> bytes:
    """
    Embed caption into image metadata based on file format.

    Supports JPEG (EXIF), PNG (text chunk), and converts other formats to JPEG with EXIF.

    Args:
        image: PIL Image object
        caption: Caption text to embed
        filename: Original filename to determine format

    Returns:
        Image bytes with embedded caption
    """
    output = io.BytesIO()
    file_ext = Path(filename).suffix.lower()

    # Determine format from image.format or filename extension
    if image.format == 'JPEG' or file_ext in ('.jpg', '.jpeg'):
        # JPEG: Use EXIF tag 0x010E (ImageDescription)
        exif = image.getexif()
        exif[0x010E] = caption
        image.save(output, format='JPEG', exif=exif, quality=95)

    elif image.format == 'PNG' or file_ext == '.png':
        # PNG: Use text chunk
        metadata = PngInfo()
        metadata.add_text("Description", caption)
        image.save(output, format='PNG', pnginfo=metadata)

    else:
        # Other formats: Convert to JPEG with EXIF
        # Convert modes that can't be saved as JPEG
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')

        exif = Image.Exif()
        exif[0x010E] = caption
        image.save(output, format='JPEG', exif=exif, quality=95)

    return output.getvalue()
