"""
Database package for session management and execution jobs.

Provides both synchronous and asynchronous session managers:
- SessionManager: Traditional synchronous DuckDB operations
- AsyncSessionManager: Async operations using thread pools for non-blocking I/O
- ExecutionManager: Persistence for graph execution jobs

Usage:
    from database import SessionManager, AsyncSessionManager, ExecutionManager

    # Synchronous (original)
    session = SessionManager()
    path = session.get_image_path(image_id)

    # Asynchronous (new)
    async_session = AsyncSessionManager()
    path = await async_session.get_image_path(image_id)

    # Execution jobs
    exec_manager = ExecutionManager()
    job_id = exec_manager.create_job(graph, image_ids)
"""

from .session_manager import SessionManager
from .async_session_manager import AsyncSessionManager
from .execution_manager import ExecutionManager

__all__ = ['SessionManager', 'AsyncSessionManager', 'ExecutionManager']
