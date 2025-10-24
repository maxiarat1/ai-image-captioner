"""
Session manager for handling image metadata and storage.
Uses SQLite for lightweight persistent storage.
"""
import sqlite3
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from PIL import Image
import logging

from config import (
    DATABASE_PATH,
    TEMP_UPLOAD_DIR,
    SUPPORTED_IMAGE_FORMATS,
)

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages image metadata and file storage using SQLite."""

    def __init__(self):
        """Initialize database connection and create tables."""
        self.db_path = DATABASE_PATH
        self._init_database()
        logger.info("Session manager initialized with database: %s", self.db_path)

    def _init_database(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                image_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER,
                width INTEGER,
                height INTEGER,
                is_uploaded BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def _get_connection(self):
        """Get a database connection."""
        return sqlite3.connect(self.db_path)

    def register_folder(self, folder_path: str) -> List[Dict]:
        """
        Scan a folder and register all images.
        Images are already on disk, so is_uploaded=True.

        Args:
            folder_path: Filesystem path to folder

        Returns:
            List of image metadata dicts
        """
        folder = Path(folder_path).resolve()
        if not folder.exists():
            raise ValueError(f"Folder does not exist: {folder_path}")
        if not folder.is_dir():
            raise ValueError(f"Path is not a directory: {folder_path}")

        images = []
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            for file_path in sorted(folder.iterdir()):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in SUPPORTED_IMAGE_FORMATS:
                    continue

                try:
                    # Get image dimensions
                    with Image.open(file_path) as img:
                        width, height = img.size

                    image_id = str(uuid.uuid4())
                    file_size = file_path.stat().st_size

                    cursor.execute("""
                        INSERT INTO images
                        (image_id, filename, file_path, file_size, width, height, is_uploaded)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (image_id, file_path.name, str(file_path), file_size, width, height, True))

                    images.append({
                        'image_id': image_id,
                        'filename': file_path.name,
                        'size': file_size,
                        'width': width,
                        'height': height,
                        'uploaded': True
                    })

                except Exception as e:
                    logger.warning("Failed to process %s: %s", file_path, e)
                    continue

            conn.commit()
            logger.info("Registered %d images from folder: %s", len(images), folder_path)

        finally:
            conn.close()

        return images

    def register_files(self, file_metadata_list: List[Dict]) -> List[str]:
        """
        Pre-register files before upload.
        Creates database records with is_uploaded=False.

        Args:
            file_metadata_list: List of {filename, size} dicts

        Returns:
            List of image_ids
        """
        image_ids = []
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            for metadata in file_metadata_list:
                image_id = str(uuid.uuid4())
                filename = metadata['filename']
                file_size = metadata.get('size', 0)

                # Store path as future temp path
                future_path = str(TEMP_UPLOAD_DIR / f"{image_id}{Path(filename).suffix}")

                cursor.execute("""
                    INSERT INTO images
                    (image_id, filename, file_path, file_size, is_uploaded)
                    VALUES (?, ?, ?, ?, ?)
                """, (image_id, filename, future_path, file_size, False))

                image_ids.append(image_id)

            conn.commit()
            logger.info("Pre-registered %d files", len(image_ids))

        finally:
            conn.close()

        return image_ids

    def save_uploaded_file(self, image_id: str, file_data, filename: str) -> bool:
        """
        Save an uploaded file and mark as uploaded.

        Args:
            image_id: Image ID from pre-registration
            file_data: File object or bytes
            filename: Original filename

        Returns:
            True if successful
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Get the stored path
            cursor.execute("SELECT file_path FROM images WHERE image_id = ?", (image_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Image ID not found: {image_id}")

            file_path = Path(row[0])

            # Save file
            if hasattr(file_data, 'save'):
                file_data.save(file_path)
            else:
                file_path.write_bytes(file_data)

            # Get dimensions
            with Image.open(file_path) as img:
                width, height = img.size

            # Update database
            cursor.execute("""
                UPDATE images
                SET is_uploaded = ?, width = ?, height = ?, file_size = ?
                WHERE image_id = ?
            """, (True, width, height, file_path.stat().st_size, image_id))

            conn.commit()
            logger.debug("Saved uploaded file: %s", image_id)
            return True

        except Exception as e:
            logger.error("Failed to save uploaded file %s: %s", image_id, e)
            return False
        finally:
            conn.close()

    def get_image_path(self, image_id: str) -> Optional[str]:
        """
        Get the filesystem path for an image.

        Args:
            image_id: Image ID

        Returns:
            Filesystem path or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT file_path FROM images WHERE image_id = ?", (image_id,))
            row = cursor.fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def list_images(self, page: int = 1, per_page: int = 50, search: str = "") -> Dict:
        """
        Get paginated list of images.

        Args:
            page: Page number (1-indexed)
            per_page: Items per page
            search: Optional filename search

        Returns:
            Dict with images, total, page, pages
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Build query
            where_clause = ""
            params = []
            if search:
                where_clause = "WHERE filename LIKE ?"
                params.append(f"%{search}%")

            # Get total count
            cursor.execute(f"SELECT COUNT(*) FROM images {where_clause}", params)
            total = cursor.fetchone()[0]

            # Get page of results
            offset = (page - 1) * per_page
            query = f"""
                SELECT image_id, filename, file_size, width, height, is_uploaded
                FROM images
                {where_clause}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """
            cursor.execute(query, params + [per_page, offset])

            images = []
            for row in cursor.fetchall():
                images.append({
                    'image_id': row[0],
                    'filename': row[1],
                    'size': row[2],
                    'width': row[3],
                    'height': row[4],
                    'uploaded': bool(row[5])
                })

            pages = (total + per_page - 1) // per_page

            return {
                'images': images,
                'total': total,
                'page': page,
                'pages': pages,
                'per_page': per_page
            }

        finally:
            conn.close()

    def clear_all(self) -> int:
        """
        Clear all images and temporary files.

        Returns:
            Number of images deleted
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT COUNT(*) FROM images")
            count = cursor.fetchone()[0]

            cursor.execute("DELETE FROM images")
            conn.commit()

            # Clear temp uploads
            if TEMP_UPLOAD_DIR.exists():
                for file_path in TEMP_UPLOAD_DIR.iterdir():
                    try:
                        file_path.unlink()
                    except Exception as e:
                        logger.warning("Failed to delete %s: %s", file_path, e)

            logger.info("Cleared %d images and temp files", count)
            return count

        finally:
            conn.close()

    def get_image_metadata(self, image_id: str) -> Optional[Dict]:
        """
        Get full metadata for an image.

        Args:
            image_id: Image ID

        Returns:
            Metadata dict or None
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT image_id, filename, file_path, file_size, width, height, is_uploaded
                FROM images
                WHERE image_id = ?
            """, (image_id,))

            row = cursor.fetchone()
            if not row:
                return None

            return {
                'image_id': row[0],
                'filename': row[1],
                'file_path': row[2],
                'size': row[3],
                'width': row[4],
                'height': row[5],
                'uploaded': bool(row[6])
            }

        finally:
            conn.close()
