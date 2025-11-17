"""
Async Session Manager for non-blocking database operations.

This module provides async wrappers around DuckDB operations to prevent
blocking AI model inference. Database operations run in a thread pool,
allowing concurrent execution with GPU-based inference.
"""

import logging
from typing import List, Dict, Optional

from utils.async_helpers import run_in_thread, shutdown_thread_pools
from .session_manager import SessionManager

logger = logging.getLogger(__name__)


class AsyncSessionManager:
    """
    Async wrapper around SessionManager methods using thread executors.

    Database calls reuse the synchronous implementation but run in a
    background thread via utils.async_helpers.run_in_thread, keeping AI
    inference from blocking on I/O.
    """

    def __init__(self, session_manager: Optional[SessionManager] = None):
        self._session = session_manager or SessionManager()
        logger.info("Async session manager initialized")

    async def get_image_path(self, image_id: str) -> Optional[str]:
        """
        Get the filesystem path for an image (async).

        Args:
            image_id: Image ID

        Returns:
            Filesystem path or None if not found
        """
        return await run_in_thread(self._session.get_image_path, image_id)

    async def get_image_paths_batch(self, image_ids: List[str]) -> Dict[str, Optional[str]]:
        """
        Get filesystem paths for multiple images concurrently.

        Args:
            image_ids: List of image IDs

        Returns:
            Dictionary mapping image_id -> file_path (or None if not found)
        """
        return await run_in_thread(self._session.get_image_paths_batch, image_ids)

    async def save_caption(self, image_id: str, caption: str) -> bool:
        """
        Save a generated caption for an image (async, non-blocking).

        Args:
            image_id: Image ID
            caption: Generated caption text

        Returns:
            True if successful, False otherwise
        """
        return await run_in_thread(self._session.save_caption, image_id, caption)

    async def save_captions_batch(self, captions_data: List[Dict[str, str]]) -> int:
        """
        Save multiple captions concurrently.

        Args:
            captions_data: List of dicts with 'image_id' and 'caption' keys

        Returns:
            Number of successfully saved captions
        """
        return await run_in_thread(self._session.save_captions_batch, captions_data)

    async def get_image_metadata(self, image_id: str) -> Optional[Dict]:
        """
        Get image metadata by ID (async).

        Args:
            image_id: Image ID

        Returns:
            Metadata dict or None if not found
        """
        return await run_in_thread(self._session.get_image_metadata, image_id)

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
        return await run_in_thread(self._session.list_images, page, per_page, search)

    def shutdown(self):
        """
        Shutdown the thread pool gracefully.

        Call this on application shutdown.
        """
        logger.info("Shutting down async session manager...")
        shutdown_thread_pools()
