"""
Export routes for saving images with embedded captions.
"""
import io
import zipfile
import logging
from pathlib import Path
from flask import Blueprint, request, jsonify, send_file
from utils.image_utils import load_image
from app.utils import embed_caption_in_image

logger = logging.getLogger(__name__)

bp = Blueprint('export', __name__)


def init_routes():
    """Initialize routes."""

    @bp.route('/export/metadata', methods=['POST'])
    def export_with_metadata():
        try:
            data = request.get_json() if request.is_json else {}

            if 'image_paths' in data and 'captions' in data:
                image_sources = [(Path(p), None) for p in data['image_paths']]
                captions = data['captions']
            elif 'images' in request.files and 'captions' in request.form:
                image_sources = [(None, f) for f in request.files.getlist('images')]
                captions = request.form.getlist('captions')
            else:
                return jsonify({"error": "Missing images or captions"}), 400

            if len(image_sources) != len(captions):
                return jsonify({"error": "Images and captions count mismatch"}), 400

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for (img_path, img_file), caption in zip(image_sources, captions):
                    if img_path:
                        if not img_path.exists():
                            continue
                        image, filename = load_image(img_path), img_path.name
                    else:
                        image, filename = load_image(img_file.read()), img_file.filename

                    image_bytes = embed_caption_in_image(image, caption, filename)
                    zip_file.writestr(filename, image_bytes)

            zip_buffer.seek(0)
            return send_file(zip_buffer, mimetype='application/zip', as_attachment=True,
                            download_name='images_with_metadata.zip')
        except Exception as e:
            logger.exception("Error in export_with_metadata: %s", e)
            return jsonify({"error": str(e)}), 500

    return bp
