"""
Async utility functions for concurrent operations.

This module provides utilities for running blocking operations asynchronously
using thread pools and async/await patterns.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger(__name__)

# Global thread pool for database operations
# Size is optimized for I/O-bound operations (DuckDB reads/writes)
_db_thread_pool = None
_DB_POOL_SIZE = 4  # Adjust based on system resources


def get_db_thread_pool() -> ThreadPoolExecutor:
    """
    Get or create the global database thread pool.

    Returns:
        ThreadPoolExecutor configured for database operations
    """
    global _db_thread_pool
    if _db_thread_pool is None:
        _db_thread_pool = ThreadPoolExecutor(
            max_workers=_DB_POOL_SIZE,
            thread_name_prefix="db_worker"
        )
        logger.info("Initialized database thread pool with %d workers", _DB_POOL_SIZE)
    return _db_thread_pool


async def run_in_thread(func: Callable, *args, **kwargs) -> Any:
    """
    Run a blocking function in a thread pool.

    Args:
        func: Blocking function to execute
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Result from the function execution
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        get_db_thread_pool(),
        lambda: func(*args, **kwargs)
    )


def async_db_operation(func: Callable) -> Callable:
    """
    Decorator to convert a synchronous database operation to async.

    Usage:
        @async_db_operation
        def my_db_query(conn, param):
            return conn.execute(query, param).fetchall()

        # Can now be called with await
        result = await my_db_query(conn, value)

    Args:
        func: Synchronous function to wrap

    Returns:
        Async wrapper function
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await run_in_thread(func, *args, **kwargs)
    return wrapper


async def gather_with_concurrency(n: int, *tasks):
    """
    Run multiple async tasks with a concurrency limit.

    Useful for batch operations where you want to limit concurrent executions.

    Args:
        n: Maximum number of concurrent tasks
        *tasks: Async tasks/coroutines to execute

    Returns:
        List of results in the same order as input tasks
    """
    semaphore = asyncio.Semaphore(n)

    async def sem_task(task):
        async with semaphore:
            return await task

    return await asyncio.gather(*(sem_task(task) for task in tasks))


def shutdown_thread_pools():
    """
    Shutdown all thread pools gracefully.

    Should be called on application shutdown.
    """
    global _db_thread_pool
    if _db_thread_pool:
        logger.info("Shutting down database thread pool...")
        _db_thread_pool.shutdown(wait=True)
        _db_thread_pool = None
