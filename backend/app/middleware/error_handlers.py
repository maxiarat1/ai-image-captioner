"""
Global error handlers for the Flask application.
"""
import logging
from flask import jsonify

logger = logging.getLogger(__name__)


def register_error_handlers(app):
    """
    Register error handlers with the Flask application.

    Args:
        app: Flask application instance
    """

    @app.errorhandler(413)
    def handle_file_too_large(_):
        """Handle file size exceeded error."""
        return jsonify({"error": "File too large"}), 413

    @app.errorhandler(500)
    def handle_internal_error(e):
        """Handle internal server errors."""
        logger.exception("Internal server error: %s", e)
        return jsonify({"error": "Internal server error"}), 500
