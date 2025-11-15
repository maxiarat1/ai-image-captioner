import os
import sys
import logging
import webbrowser
from pathlib import Path
from flask import Flask, send_from_directory
from flask_cors import CORS
if os.environ.get("TAGGER_FORCE_CPU", "0") == "1":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from app.models import CATEGORIES, MODEL_METADATA
from app.services import ModelManager
from app.middleware import register_error_handlers
from app.routes import register_blueprints
from utils.logging_utils import setup_logging
from database import SessionManager, AsyncSessionManager, ExecutionManager
from config import MAX_FILE_SIZE

setup_logging()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Determine frontend path for static files
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path='')
CORS(app)

# Register error handlers
register_error_handlers(app)

# Session managers: sync for simple CRUD, async for AI inference paths
session_manager = SessionManager()
async_session_manager = AsyncSessionManager()
execution_manager = ExecutionManager()

# Active graph executors (job_id -> GraphExecutor)
active_executors = {}

# Model manager for handling model lifecycle
model_manager = ModelManager()

# Register all route blueprints
register_blueprints(app, model_manager, session_manager, async_session_manager,
                   execution_manager, active_executors, CATEGORIES, MODEL_METADATA)


@app.route('/')
def index():
    """Serve the frontend UI."""
    return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    # Determine frontend path - works in both development and PyInstaller bundle
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        base_path = sys._MEIPASS
        frontend_path = os.path.join(base_path, "frontend", "index.html")
    else:
        # Running as normal Python script
        frontend_path = os.path.join(os.path.dirname(__file__), "../frontend/index.html")
        frontend_path = os.path.abspath(frontend_path)
    
    # Open browser only if not running in Docker (where DISPLAY isn't available)
    in_docker = os.path.exists('/.dockerenv') or os.environ.get('RUNNING_IN_DOCKER') == '1'
    
    if not in_docker:
        webbrowser.open(f"file://{frontend_path}")
    else:
        # In Docker, print URL for user to open manually
        logger.info("=" * 60)
        logger.info("AI Image Captioner is running!")
        logger.info("=" * 60)
    
    app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE  # None = no limit
    model_manager.initialize()
    flask_debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=flask_debug, host='0.0.0.0', port=5000)
