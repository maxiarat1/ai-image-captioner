"""
Async Session Manager for non-blocking database operations.

This module provides async wrappers around DuckDB operations to prevent
blocking AI model inference. Database operations run in a thread pool,
allowing concurrent execution with GPU-based inference.
"""

import duckdb
import logging
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

from config import DATABASE_PATH
from utils.async_helpers import run_in_thread

logger = logging.getLogger(__name__)


class AsyncSessionManager:
    """
    Async wrapper for SessionManager that runs DuckDB operations in thread pool.

    This allows database operations to run concurrently with AI inference,
    improving overall throughput especially during batch processing.
    """

    def __init__(self):
        self.db_path = DATABASE_PATH
        self._thread_pool = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="async_db"
        )
        logger.info("Async session manager initialized with database: %s", self.db_path)

    def _get_connection(self):
        """
        Get a new DuckDB connection.

        Note: Each thread should have its own connection for thread safety.
        """
        return duckdb.connect(str(self.db_path))

    async def get_image_path(self, image_id: str) -> Optional[str]:
        """
        Get the filesystem path for an image (async).

        Args:
            image_id: Image ID

        Returns:
            Filesystem path or None if not found
        """
        def _get_path():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT file_path FROM images WHERE image_id = ?", (image_id,))
                row = cursor.fetchone()
                return row[0] if row else None
            finally:
                conn.close()

        return await run_in_thread(_get_path)

    async def get_image_paths_batch(self, image_ids: List[str]) -> Dict[str, Optional[str]]:
        """
        Get filesystem paths for multiple images concurrently.

        Args:
            image_ids: List of image IDs

        Returns:
            Dictionary mapping image_id -> file_path (or None if not found)
        """
        def _get_paths():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                placeholders = ','.join(['?'] * len(image_ids))
                query = f"SELECT image_id, file_path FROM images WHERE image_id IN ({placeholders})"
                cursor.execute(query, image_ids)
                return {row[0]: row[1] for row in cursor.fetchall()}
            finally:
                conn.close()

        return await run_in_thread(_get_paths)

    async def save_caption(self, image_id: str, caption: str) -> bool:
        """
        Save a generated caption for an image (async, non-blocking).

        Args:
            image_id: Image ID
            caption: Generated caption text

        Returns:
            True if successful, False otherwise
        """
        def _save():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
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

        return await run_in_thread(_save)

    async def save_captions_batch(self, captions_data: List[Dict[str, str]]) -> int:
        """
        Save multiple captions concurrently.

        Args:
            captions_data: List of dicts with 'image_id' and 'caption' keys

        Returns:
            Number of successfully saved captions
        """
        def _save_batch():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                success_count = 0

                for item in captions_data:
                    try:
                        cursor.execute("""
                            UPDATE images
                            SET caption = ?
                            WHERE image_id = ?
                        """, (item['caption'], item['image_id']))
                        if cursor.rowcount > 0:
                            success_count += 1
                    except Exception as e:
                        logger.error("Failed to save caption for %s: %s", item['image_id'], e)

                conn.commit()
                return success_count
            finally:
                conn.close()

        return await run_in_thread(_save_batch)

    async def get_image_metadata(self, image_id: str) -> Optional[Dict]:
        """
        Get image metadata by ID (async).

        Args:
            image_id: Image ID

        Returns:
            Metadata dict or None if not found
        """
        def _get_metadata():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT image_id, filename, file_path, file_size, width, height, is_uploaded, caption
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
                    'uploaded': bool(row[6]),
                    'caption': row[7]
                }
            finally:
                conn.close()

        return await run_in_thread(_get_metadata)

    async def list_images(self, page: int = 1, per_page: int = 50, search: str = "") -> Dict:
        """
        Get paginated list of images (async).

        Args:
            page: Page number (1-indexed)
            per_page: Items per page
            search: Optional search term for filename

        Returns:
            Dict with 'images', 'total', 'page', 'pages', 'per_page'
        """
        def _list():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

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

        return await run_in_thread(_list)

    def shutdown(self):
        """
        Shutdown the thread pool gracefully.

        Call this on application shutdown.
        """
        logger.info("Shutting down async session manager...")
        self._thread_pool.shutdown(wait=True)
