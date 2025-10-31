"""
Database package for session management.

Provides both synchronous and asynchronous session managers:
- SessionManager: Traditional synchronous DuckDB operations
- AsyncSessionManager: Async operations using thread pools for non-blocking I/O

Usage:
    from database import SessionManager, AsyncSessionManager

    # Synchronous (original)
    session = SessionManager()
    path = session.get_image_path(image_id)

    # Asynchronous (new)
    async_session = AsyncSessionManager()
    path = await async_session.get_image_path(image_id)
"""

from .session_manager import SessionManager
from .async_session_manager import AsyncSessionManager

__all__ = ['SessionManager', 'AsyncSessionManager']
