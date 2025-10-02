import io
import base64
from PIL import Image
from typing import Union

def process_uploaded_image(image_data: Union[str, bytes]) -> Image.Image:
    """Process uploaded image data and return PIL Image"""
    try:
        if isinstance(image_data, str):
            # Handle base64 encoded image
            if image_data.startswith('data:image'):
                # Remove data URL prefix
                image_data = image_data.split(',')[1]

            # Decode base64
            image_bytes = base64.b64decode(image_data)
        else:
            # Handle bytes directly
            image_bytes = image_data

        # Create PIL Image
        image = Image.open(io.BytesIO(image_bytes))

        # Store original format before any conversion
        original_format = image.format

        # Convert RGBA to RGB (JPEG doesn't support alpha channel)
        if image.mode == 'RGBA':
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])  # Use alpha channel as mask
            rgb_image.format = original_format  # Preserve original format
            image = rgb_image

        return image

    except Exception as e:
        raise ValueError(f"Invalid image data: {str(e)}")

def validate_image_format(image: Image.Image) -> bool:
    """Validate if image format is supported"""
    supported_formats = ['JPEG', 'PNG', 'JPG', 'WEBP', 'BMP']
    return image.format in supported_formats