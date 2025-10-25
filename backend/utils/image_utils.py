import io
import base64
from pathlib import Path
from PIL import Image
from typing import Union

def load_image(source: Union[str, bytes, Path]) -> Image.Image:
    try:
        if isinstance(source, (str, Path)) and not str(source).startswith('data:'):
            path = Path(source)
            if not path.exists():
                raise ValueError(f"Image file not found: {source}")
            with open(path, 'rb') as f:
                image_bytes = f.read()
        elif isinstance(source, str) and source.startswith('data:image'):
            image_bytes = base64.b64decode(source.split(',')[1])
        else:
            image_bytes = source

        image = Image.open(io.BytesIO(image_bytes))
        
        if image.format not in {'JPEG', 'PNG', 'JPG', 'WEBP', 'BMP'}:
            raise ValueError(f"Unsupported image format: {image.format}")
        
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
    buffered = io.BytesIO()
    image.save(buffered, format=format, quality=quality)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"