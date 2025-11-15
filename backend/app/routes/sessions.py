"""
Session and image management routes.
"""
import logging
from pathlib import Path
from flask import Blueprint, request, jsonify
from utils.image_utils import load_image, image_to_base64
from config import SUPPORTED_IMAGE_FORMATS

logger = logging.getLogger(__name__)

bp = Blueprint('sessions', __name__)


def init_routes(session_manager):
    """Initialize routes with dependencies."""

    @bp.route('/session/register-folder', methods=['POST'])
    def register_folder():
        """Register all images from a folder path."""
        try:
            folder_path = request.get_json().get('folder_path', '')
            if not folder_path:
                return jsonify({"error": "No folder path provided"}), 400

            images = session_manager.register_folder(folder_path)
            return jsonify({"success": True, "images": images, "total": len(images)})
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.exception("Error registering folder: %s", e)
            return jsonify({"error": str(e)}), 500

    @bp.route('/session/register-files', methods=['POST'])
    def register_files():
        """Pre-register files before upload."""
        try:
            data = request.get_json()
            file_metadata_list = data.get('files', [])
            if not file_metadata_list:
                return jsonify({"error": "No files provided"}), 400

            image_ids = session_manager.register_files(file_metadata_list)
            return jsonify({"success": True, "image_ids": image_ids, "total": len(image_ids)})
        except Exception as e:
            logger.exception("Error registering files: %s", e)
            return jsonify({"error": str(e)}), 500

    @bp.route('/upload/batch', methods=['POST'])
    def upload_batch():
        """Upload a batch of files."""
        try:
            if 'files' not in request.files:
                return jsonify({"error": "No files provided"}), 400

            files = request.files.getlist('files')
            image_ids = request.form.getlist('image_ids')

            if len(files) != len(image_ids):
                return jsonify({"error": "Files and image_ids count mismatch"}), 400

            uploaded = 0
            failed = 0

            for file, image_id in zip(files, image_ids):
                if session_manager.save_uploaded_file(image_id, file, file.filename):
                    uploaded += 1
                else:
                    failed += 1

            return jsonify({"success": True, "uploaded": uploaded, "failed": failed, "image_ids": image_ids})
        except Exception as e:
            logger.exception("Error uploading batch: %s", e)
            return jsonify({"error": str(e)}), 500

    @bp.route('/images', methods=['GET'])
    def list_images():
        """Get paginated list of images."""
        try:
            page = int(request.args.get('page', 1))
            per_page = int(request.args.get('per_page', 50))
            search = request.args.get('search', '')

            result = session_manager.list_images(page, per_page, search)
            return jsonify(result)
        except Exception as e:
            logger.exception("Error listing images: %s", e)
            return jsonify({"error": str(e)}), 500

    @bp.route('/image/<image_id>', methods=['GET'])
    def get_image_by_id(image_id):
        """Get full-resolution image by ID."""
        try:
            image_path = session_manager.get_image_path(image_id)
            if not image_path:
                return jsonify({"error": "Image not found"}), 404

            image = load_image(image_path)
            return jsonify({"success": True, "image": image_to_base64(image)})
        except ValueError as e:
            return jsonify({"error": str(e)}), 404
        except Exception as e:
            logger.exception("Error loading image: %s", e)
            return jsonify({"error": str(e)}), 500

    @bp.route('/image/<image_id>/info', methods=['GET'])
    def get_image_info(image_id):
        """Get image metadata by ID."""
        try:
            metadata = session_manager.get_image_metadata(image_id)
            if not metadata:
                return jsonify({"error": "Image not found"}), 404
            return jsonify(metadata)
        except Exception as e:
            logger.exception("Error getting image info: %s", e)
            return jsonify({"error": str(e)}), 500

    @bp.route('/session/clear', methods=['DELETE'])
    def clear_session():
        """Clear all images and temporary files."""
        try:
            count = session_manager.clear_all()
            return jsonify({"success": True, "message": "Session cleared", "deleted_count": count})
        except Exception as e:
            logger.exception("Error clearing session: %s", e)
            return jsonify({"error": str(e)}), 500

    @bp.route('/session/remove/<image_id>', methods=['DELETE'])
    def remove_image(image_id):
        """Remove a single image from the session."""
        try:
            success = session_manager.delete_image(image_id)
            if success:
                return jsonify({"success": True, "message": "Image removed", "image_id": image_id})
            else:
                return jsonify({"success": False, "error": "Image not found"}), 404
        except Exception as e:
            logger.exception("Error removing image %s: %s", image_id, e)
            return jsonify({"error": str(e)}), 500

    @bp.route('/scan-folder', methods=['POST'])
    def scan_folder():
        try:
            folder_path = request.get_json().get('folder_path', '')
            if not folder_path:
                return jsonify({"error": "No folder path provided"}), 400

            folder = Path(folder_path).resolve()
            if not folder.exists():
                return jsonify({"error": "Folder does not exist"}), 404
            if not folder.is_dir():
                return jsonify({"error": "Path is not a directory"}), 400

            images = sorted([
                {'filename': f.name, 'path': str(f), 'size': f.stat().st_size}
                for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_FORMATS
            ], key=lambda x: x['filename'])

            return jsonify({"success": True, "folder": str(folder), "images": images, "count": len(images)})
        except Exception as e:
            logger.exception("Error scanning folder: %s", e)
            return jsonify({"error": str(e)}), 500

    @bp.route('/thumbnail/<image_id>', methods=['GET'])
    def get_thumbnail(image_id):
        """Get thumbnail for an image."""
        try:
            thumbnail_base64 = session_manager.get_thumbnail(image_id)
            if not thumbnail_base64:
                return jsonify({"error": "Thumbnail not found"}), 404
            return jsonify({"success": True, "thumbnail": thumbnail_base64})
        except Exception as e:
            logger.exception("Error getting thumbnail: %s", e)
            return jsonify({"error": str(e)}), 500

    return bp
