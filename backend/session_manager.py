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

    def __init__(self):
        self.db_path = DATABASE_PATH
        self._init_database()
        logger.info("Session manager initialized with database: %s", self.db_path)

    def _init_database(self):
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
                caption TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("PRAGMA table_info(images)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'caption' not in columns:
            logger.info("Migrating database: adding caption column")
            cursor.execute("ALTER TABLE images ADD COLUMN caption TEXT")

        conn.commit()
        conn.close()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def register_folder(self, folder_path: str) -> List[Dict]:
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
        image_ids = []
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            for metadata in file_metadata_list:
                image_id = str(uuid.uuid4())
                filename = metadata['filename']
                file_size = metadata.get('size', 0)

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
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT file_path FROM images WHERE image_id = ?", (image_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Image ID not found: {image_id}")

            file_path = Path(row[0])

            if hasattr(file_data, 'save'):
                file_data.save(file_path)
            else:
                file_path.write_bytes(file_data)

            with Image.open(file_path) as img:
                width, height = img.size

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
                SELECT image_id, filename, file_size, width, height, is_uploaded, caption
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
                    'uploaded': bool(row[5]),
                    'caption': row[6]
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

    def delete_image(self, image_id: str) -> bool:
        """
        Delete a single image from database and filesystem.

        Args:
            image_id: Image ID to delete

        Returns:
            True if deleted successfully, False if not found or error
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Get file path before deleting record
            cursor.execute("SELECT file_path, is_uploaded FROM images WHERE image_id = ?", (image_id,))
            row = cursor.fetchone()
            if not row:
                logger.warning("Image ID not found for deletion: %s", image_id)
                return False

            file_path = Path(row[0])
            is_uploaded = bool(row[1])

            # Delete database record
            cursor.execute("DELETE FROM images WHERE image_id = ?", (image_id,))
            conn.commit()

            # Delete physical file if it was uploaded (in temp directory)
            if is_uploaded and file_path.exists() and TEMP_UPLOAD_DIR in file_path.parents:
                try:
                    file_path.unlink()
                    logger.debug("Deleted file: %s", file_path)
                except Exception as e:
                    logger.warning("Failed to delete file %s: %s", file_path, e)

            logger.info("Deleted image: %s", image_id)
            return True

        except Exception as e:
            logger.error("Error deleting image %s: %s", image_id, e)
            return False
        finally:
            conn.close()

    def save_caption(self, image_id: str, caption: str) -> bool:
        """
        Save a generated caption for an image.

        Args:
            image_id: Image ID
            caption: Generated caption text

        Returns:
            True if successful, False otherwise
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                UPDATE images
                SET caption = ?
                WHERE image_id = ?
            """, (caption, image_id))

            conn.commit()
            return cursor.rowcount > 0

        except Exception as e:
            logger.error("Failed to save caption for %s: %s", image_id, e)
            return False
        finally:
            conn.close()

    def get_image_metadata(self, image_id: str) -> Optional[Dict]:
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
