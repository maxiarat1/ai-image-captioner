import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TEMP_UPLOAD_DIR = BASE_DIR / "temp_uploads"
THUMBNAIL_DIR = BASE_DIR / "thumbnails"

DATABASE_PATH = DATA_DIR / "images.db"

SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
THUMBNAIL_SIZE = (150, 150)
MAX_FILE_SIZE = 16 * 1024 * 1024

IMAGE_CACHE_MAX_SIZE = 30
THUMBNAIL_CACHE_MAX_SIZE = 100

DATA_DIR.mkdir(exist_ok=True)
TEMP_UPLOAD_DIR.mkdir(exist_ok=True)
THUMBNAIL_DIR.mkdir(exist_ok=True)
