"""
Export routes for saving images with embedded captions.
"""
import io
import zipfile
import logging
from pathlib import Path
from flask import Blueprint, request, send_file
from utils.image_utils import load_image
from app.utils import embed_caption_in_image
from app.utils.route_decorators import handle_route_errors

logger = logging.getLogger(__name__)

bp = Blueprint('export', __name__)


def init_routes():
    """Initialize routes."""

    @bp.route('/export/metadata', methods=['POST'])
    @handle_route_errors("exporting with metadata")
    def export_with_metadata():
        data = request.get_json() if request.is_json else {}

        if 'image_paths' in data and 'captions' in data:
            # Filter out None/empty paths
            valid_pairs = [(p, c) for p, c in zip(data['image_paths'], data['captions']) if p]
            if not valid_pairs:
                raise ValueError("No valid image paths provided. Images may have been uploaded directly without server-side paths.")
            paths, captions = zip(*valid_pairs)
            image_sources = [(Path(p), None) for p in paths]
            captions = list(captions)
        elif 'images' in request.files and 'captions' in request.form:
            image_sources = [(None, f) for f in request.files.getlist('images')]
            captions = request.form.getlist('captions')
        else:
            raise ValueError("Missing images or captions")

        if len(image_sources) != len(captions):
            raise ValueError("Images and captions count mismatch")

        zip_buffer = io.BytesIO()
        processed_count = 0
        skipped_paths = []

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for (img_path, img_file), caption in zip(image_sources, captions):
                if img_path:
                    if not img_path.exists():
                        skipped_paths.append(str(img_path))
                        continue
                    image, filename = load_image(img_path), img_path.name
                else:
                    image, filename = load_image(img_file.read()), img_file.filename

                image_bytes = embed_caption_in_image(image, caption, filename)
                zip_file.writestr(filename, image_bytes)
                processed_count += 1

        if processed_count == 0:
            if skipped_paths:
                raise ValueError(f"No images could be processed. Missing files: {', '.join(skipped_paths[:5])}")
            raise ValueError("No images were processed")

        zip_buffer.seek(0)
        return send_file(zip_buffer, mimetype='application/zip', as_attachment=True,
                        download_name='images_with_metadata.zip')

    return bp
