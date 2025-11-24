"""
Helper utilities for model parameters and image processing.
"""
import io
from typing import Optional, Dict
from pathlib import Path
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from app.models import get_factory


def extract_precision_params(model_name: str, parameters: Optional[Dict]) -> Optional[Dict[str, any]]:
    """
    Extract precision parameters for a model, falling back to defaults.

    Args:
        model_name: Name of the model
        parameters: User-provided parameters dict

    Returns:
        Dict with precision and use_flash_attention, or None if not applicable
    """
    # Return None if no parameters provided
    if not parameters:
        return None

    # Get defaults from factory
    factory = get_factory()
    defaults = factory.get_precision_defaults(model_name)

    # Return None if model doesn't have precision defaults
    if not defaults:
        return None

    # Extract only precision-related parameters
    return {
        'precision': parameters.get('precision', defaults['precision']),
        'use_flash_attention': parameters.get('use_flash_attention', defaults['use_flash_attention'])
    }


def _create_xmp_packet(caption: str) -> bytes:
    """Create XMP metadata packet with description."""
    # Escape XML special characters
    escaped_caption = (caption
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
        .replace("'", '&apos;'))

    xmp = f'''<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
        xmlns:dc="http://purl.org/dc/elements/1.1/">
      <dc:description>
        <rdf:Alt>
          <rdf:li xml:lang="x-default">{escaped_caption}</rdf:li>
        </rdf:Alt>
      </dc:description>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>'''
    return xmp.encode('utf-8')


def embed_caption_in_image(image: Image.Image, caption: str, filename: str) -> bytes:
    """
    Embed caption into image metadata based on file format.

    Uses XMP metadata (dc:description) which is visible in most file managers.
    Also includes EXIF ImageDescription for JPEG and PNG text chunks for compatibility.

    Args:
        image: PIL Image object
        caption: Caption text to embed
        filename: Original filename to determine format

    Returns:
        Image bytes with embedded caption
    """
    output = io.BytesIO()
    file_ext = Path(filename).suffix.lower()
    xmp_data = _create_xmp_packet(caption)

    # Determine format from image.format or filename extension
    if image.format == 'JPEG' or file_ext in ('.jpg', '.jpeg'):
        # JPEG: Use EXIF tag 0x010E (ImageDescription) + XMP
        exif = image.getexif()
        exif[0x010E] = caption
        image.save(output, format='JPEG', exif=exif, quality=95, xmp=xmp_data)

    elif image.format == 'PNG' or file_ext == '.png':
        # PNG: Use text chunk + XMP
        metadata = PngInfo()
        metadata.add_text("Description", caption)
        # Add XMP as iTXt chunk
        metadata.add_text("XML:com.adobe.xmp", xmp_data.decode('utf-8'), zip=True)
        image.save(output, format='PNG', pnginfo=metadata)

    else:
        # Other formats: Convert to JPEG with EXIF + XMP
        # Convert modes that can't be saved as JPEG
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')

        exif = Image.Exif()
        exif[0x010E] = caption
        image.save(output, format='JPEG', exif=exif, quality=95, xmp=xmp_data)

    return output.getvalue()
