"""
Session and image management routes.
"""
import logging
from pathlib import Path
from flask import Blueprint, request, jsonify
from utils.image_utils import load_image, image_to_base64
from config import SUPPORTED_IMAGE_FORMATS
from app.utils.route_decorators import handle_route_errors
from app.utils.request_validators import (
    extract_json_fields, extract_query_params, RequestField,
    non_empty_list, to_int
)

logger = logging.getLogger(__name__)

bp = Blueprint('sessions', __name__)


def init_routes(session_manager):
    """Initialize routes with dependencies."""

    @bp.route('/session/register-folder', methods=['POST'])
    @handle_route_errors("registering folder")
    def register_folder():
        """Register all images from a folder path."""
        data = extract_json_fields(
            RequestField('folder_path', required=True, error_message="No folder path provided")
        )

        images = session_manager.register_folder(data['folder_path'])
        return {"success": True, "images": images, "total": len(images)}

    @bp.route('/session/register-files', methods=['POST'])
    @handle_route_errors("registering files")
    def register_files():
        """Pre-register files before upload."""
        data = extract_json_fields(
            RequestField('files', required=True, validator=non_empty_list,
                        error_message="No files provided")
        )

        image_ids = session_manager.register_files(data['files'])
        return {"success": True, "image_ids": image_ids, "total": len(image_ids)}

    @bp.route('/upload/batch', methods=['POST'])
    @handle_route_errors("uploading batch")
    def upload_batch():
        """Upload a batch of files."""
        if 'files' not in request.files:
            raise ValueError("No files provided")

        files = request.files.getlist('files')
        image_ids = request.form.getlist('image_ids')

        if len(files) != len(image_ids):
            raise ValueError("Files and image_ids count mismatch")

        uploaded = 0
        failed = 0

        for file, image_id in zip(files, image_ids):
            if session_manager.save_uploaded_file(image_id, file, file.filename):
                uploaded += 1
            else:
                failed += 1

        return {"success": True, "uploaded": uploaded, "failed": failed, "image_ids": image_ids}

    @bp.route('/images', methods=['GET'])
    @handle_route_errors("listing images")
    def list_images():
        """Get paginated list of images."""
        params = extract_query_params(
            RequestField('page', default='1', transform=to_int),
            RequestField('per_page', default='50', transform=to_int),
            RequestField('search', default='')
        )

        result = session_manager.list_images(params['page'], params['per_page'], params['search'])
        return result

    @bp.route('/image/<image_id>', methods=['GET'])
    @handle_route_errors("loading image")
    def get_image_by_id(image_id):
        """Get full-resolution image by ID."""
        image_path = session_manager.get_image_path(image_id)
        if not image_path:
            return jsonify({"error": "Image not found"}), 404

        image = load_image(image_path)
        return {"success": True, "image": image_to_base64(image)}

    @bp.route('/image/<image_id>/info', methods=['GET'])
    @handle_route_errors("getting image info")
    def get_image_info(image_id):
        """Get image metadata by ID."""
        metadata = session_manager.get_image_metadata(image_id)
        if not metadata:
            return jsonify({"error": "Image not found"}), 404
        return metadata

    @bp.route('/session/clear', methods=['DELETE'])
    @handle_route_errors("clearing session")
    def clear_session():
        """Clear all images and temporary files."""
        count = session_manager.clear_all()
        return {"success": True, "message": "Session cleared", "deleted_count": count}

    @bp.route('/session/remove/<image_id>', methods=['DELETE'])
    @handle_route_errors("removing image")
    def remove_image(image_id):
        """Remove a single image from the session."""
        success = session_manager.delete_image(image_id)
        if success:
            return {"success": True, "message": "Image removed", "image_id": image_id}
        else:
            return jsonify({"success": False, "error": "Image not found"}), 404

    @bp.route('/scan-folder', methods=['POST'])
    @handle_route_errors("scanning folder")
    def scan_folder():
        data = extract_json_fields(
            RequestField('folder_path', required=True, error_message="No folder path provided")
        )

        folder = Path(data['folder_path']).resolve()
        if not folder.exists():
            raise ValueError("Folder does not exist")
        if not folder.is_dir():
            raise ValueError("Path is not a directory")

        images = sorted([
            {'filename': f.name, 'path': str(f), 'size': f.stat().st_size}
            for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_FORMATS
        ], key=lambda x: x['filename'])

        return {"success": True, "folder": str(folder), "images": images, "count": len(images)}

    @bp.route('/thumbnail/<image_id>', methods=['GET'])
    @handle_route_errors("getting thumbnail")
    def get_thumbnail(image_id):
        """Get thumbnail for an image."""
        thumbnail_base64 = session_manager.get_thumbnail(image_id)
        if not thumbnail_base64:
            return jsonify({"error": "Thumbnail not found"}), 404
        return {"success": True, "thumbnail": thumbnail_base64}

    return bp
