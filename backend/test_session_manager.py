#!/usr/bin/env python3
"""
Quick test script for session manager.
Run: python test_session_manager.py
"""
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from session_manager import SessionManager
from config import DATABASE_PATH, TEMP_UPLOAD_DIR

def test_session_manager():
    """Test basic session manager functionality."""
    print("=" * 60)
    print("Testing Session Manager")
    print("=" * 60)

    # Initialize
    print("\n1. Initializing session manager...")
    sm = SessionManager()
    print(f"   ✓ Database created at: {DATABASE_PATH}")
    print(f"   ✓ Temp upload dir: {TEMP_UPLOAD_DIR}")

    # Test pre-registration
    print("\n2. Testing file pre-registration...")
    test_files = [
        {'filename': 'test1.jpg', 'size': 1024},
        {'filename': 'test2.png', 'size': 2048},
        {'filename': 'test3.jpg', 'size': 1536}
    ]
    image_ids = sm.register_files(test_files)
    print(f"   ✓ Pre-registered {len(image_ids)} files")
    print(f"   ✓ Image IDs: {image_ids[:2]}... (showing first 2)")

    # Test listing
    print("\n3. Testing image listing...")
    result = sm.list_images(page=1, per_page=10)
    print(f"   ✓ Total images: {result['total']}")
    print(f"   ✓ Pages: {result['pages']}")
    print(f"   ✓ First image: {result['images'][0]['filename']}")

    # Test getting path
    print("\n4. Testing path retrieval...")
    first_id = image_ids[0]
    path = sm.get_image_path(first_id)
    print(f"   ✓ Image path: {path}")

    # Test metadata
    print("\n5. Testing metadata retrieval...")
    metadata = sm.get_image_metadata(first_id)
    print(f"   ✓ Filename: {metadata['filename']}")
    print(f"   ✓ Uploaded: {metadata['uploaded']}")
    print(f"   ✓ Size: {metadata['size']} bytes")

    # Test clearing
    print("\n6. Testing clear all...")
    count = sm.clear_all()
    print(f"   ✓ Cleared {count} images")

    result = sm.list_images()
    print(f"   ✓ Remaining images: {result['total']}")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)

if __name__ == '__main__':
    test_session_manager()
