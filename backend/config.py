import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TEMP_UPLOAD_DIR = BASE_DIR / "temp_uploads"
THUMBNAIL_DIR = BASE_DIR / "thumbnails"

DATABASE_PATH = DATA_DIR / "images.duckdb"

SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
THUMBNAIL_SIZE = (150, 150)
MAX_FILE_SIZE = None  # No limit for local application

IMAGE_CACHE_MAX_SIZE = 30
THUMBNAIL_CACHE_MAX_SIZE = 100

PRECISION_DEFAULTS = {
    'r4b': {'precision': 'float32', 'use_flash_attention': False},
    'blip2': {'precision': 'bfloat16', 'use_flash_attention': False},
    'lfm2-vl-3b': {'precision': 'bfloat16', 'use_flash_attention': False},
    'llava-phi3': {'precision': 'float16', 'use_flash_attention': False},
    'nanonets-ocr-s': {'precision': 'bfloat16', 'use_flash_attention': False}
}

DATA_DIR.mkdir(exist_ok=True)
TEMP_UPLOAD_DIR.mkdir(exist_ok=True)
THUMBNAIL_DIR.mkdir(exist_ok=True)
