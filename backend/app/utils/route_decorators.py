"""
Route decorators for standardized error handling and response formatting.

This module provides decorators that can be applied to Flask route handlers
to eliminate boilerplate code and ensure consistent error handling across
all API endpoints.
"""

import logging
import inspect
from functools import wraps
from flask import jsonify

logger = logging.getLogger(__name__)


def handle_route_errors(route_description=None):
    """
    Decorator to standardize error handling across all routes.

    This decorator wraps route handlers to provide consistent error handling,
    response formatting, and logging. It eliminates the need for repetitive
    try-except blocks in every route handler.

    Handles:
    - ValueError → 400 Bad Request (client error)
    - Exception → 500 Internal Server Error (server error)
    - Automatic JSON response formatting via jsonify()
    - Consistent error logging with context

    Args:
        route_description: Optional human-readable description for logging.
                          If not provided, defaults to the function name.

    Returns:
        Decorated function that handles errors and formats responses.

    Usage:
        @bp.route('/endpoint', methods=['POST'])
        @handle_route_errors("processing user request")
        async def my_route():
            # Business logic only - no try-except needed
            result = do_something()
            return {"success": True, "data": result}

        # For routes without description:
        @bp.route('/simple', methods=['GET'])
        @handle_route_errors()
        def simple_route():
            return {"status": "ok"}

    Notes:
        - Works with both sync and async route handlers
        - Automatically wraps dict/list returns in jsonify()
        - Preserves Response objects (doesn't double-wrap)
        - Supports tuple returns like (data, status_code)
    """
    def decorator(f):
        # Determine if the function is async or sync
        is_async = inspect.iscoroutinefunction(f)
        desc = route_description or f.__name__

        if is_async:
            @wraps(f)
            async def async_wrapper(*args, **kwargs):
                try:
                    result = await f(*args, **kwargs)
                    return _format_response(result)
                except ValueError as e:
                    logger.warning("%s - ValueError: %s", desc, str(e))
                    return jsonify({"error": str(e)}), 400
                except Exception as e:
                    logger.exception("Error in %s: %s", desc, e)
                    return jsonify({"error": str(e)}), 500

            return async_wrapper
        else:
            @wraps(f)
            def sync_wrapper(*args, **kwargs):
                try:
                    result = f(*args, **kwargs)
                    return _format_response(result)
                except ValueError as e:
                    logger.warning("%s - ValueError: %s", desc, str(e))
                    return jsonify({"error": str(e)}), 400
                except Exception as e:
                    logger.exception("Error in %s: %s", desc, e)
                    return jsonify({"error": str(e)}), 500

            return sync_wrapper

    return decorator


def _format_response(result):
    """
    Format route handler response for Flask.

    Args:
        result: Return value from route handler

    Returns:
        Properly formatted Flask response
    """
    # If result is already a Response object (has status_code), return as-is
    if hasattr(result, 'status_code'):
        return result

    # If result is a tuple (data, status_code, headers, etc.), handle appropriately
    if isinstance(result, tuple):
        # Wrap first element in jsonify if it's a dict/list
        data = result[0]
        rest = result[1:]
        if isinstance(data, (dict, list)):
            return (jsonify(data), *rest)
        return result

    # For dict/list, wrap in jsonify
    if isinstance(result, (dict, list)):
        return jsonify(result)

    # For everything else, return as-is (strings, Response objects, etc.)
    return result


def success_response(data=None, message=None, **kwargs):
    """
    Build a standardized success response dictionary.

    This helper creates consistent success responses across all endpoints.
    Use it when you want to return a success response from a route handler
    that uses the @handle_route_errors decorator.

    Args:
        data: Optional data payload to include in response
        message: Optional success message
        **kwargs: Additional fields to include in response

    Returns:
        dict: Response dictionary with 'success': True and optional fields

    Usage:
        @bp.route('/process', methods=['POST'])
        @handle_route_errors("processing data")
        def process():
            result = do_work()
            return success_response(
                data=result,
                message="Processing completed",
                items_processed=10
            )
    """
    response = {"success": True}

    if message:
        response["message"] = message

    if data is not None:
        response["data"] = data

    response.update(kwargs)
    return response
