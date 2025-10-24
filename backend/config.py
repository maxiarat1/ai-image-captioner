"""
Configuration constants for AI Image Captioner backend.
"""
import os
from pathlib import Path

# Directory paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TEMP_UPLOAD_DIR = BASE_DIR / "temp_uploads"
THUMBNAIL_DIR = BASE_DIR / "thumbnails"

# Database
DATABASE_PATH = DATA_DIR / "images.db"

# Image settings
SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
THUMBNAIL_SIZE = (150, 150)
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Cache settings
IMAGE_CACHE_MAX_SIZE = 30  # Keep last 30 PIL images in memory
THUMBNAIL_CACHE_MAX_SIZE = 100  # Keep last 100 thumbnails in memory

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
TEMP_UPLOAD_DIR.mkdir(exist_ok=True)
THUMBNAIL_DIR.mkdir(exist_ok=True)
