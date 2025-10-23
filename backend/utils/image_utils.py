import io
import base64
from pathlib import Path
from PIL import Image
from typing import Union

def load_image(source: Union[str, bytes, Path]) -> Image.Image:
    """
    Load and normalize image from various sources.
    
    Args:
        source: File path, base64 string, or raw bytes
        
    Returns:
        PIL Image in RGB mode with validated format
        
    Raises:
        ValueError: If image cannot be loaded or format is unsupported
    """
    try:
        # Handle file path
        if isinstance(source, (str, Path)) and not str(source).startswith('data:'):
            path = Path(source)
            if not path.exists():
                raise ValueError(f"Image file not found: {source}")
            with open(path, 'rb') as f:
                image_bytes = f.read()
        # Handle base64
        elif isinstance(source, str) and source.startswith('data:image'):
            image_bytes = base64.b64decode(source.split(',')[1])
        # Handle raw bytes
        else:
            image_bytes = source

        # Open image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Validate format
        if image.format not in {'JPEG', 'PNG', 'JPG', 'WEBP', 'BMP'}:
            raise ValueError(f"Unsupported image format: {image.format}")
        
        # Normalize to RGB
        if image.mode == 'RGBA':
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])
            rgb_image.format = image.format
            return rgb_image
        elif image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image

    except Exception as e:
        raise ValueError(f"Failed to load image: {str(e)}")

def image_to_base64(image: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
    """Convert PIL Image to base64 data URL"""
    buffered = io.BytesIO()
    image.save(buffered, format=format, quality=quality)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"