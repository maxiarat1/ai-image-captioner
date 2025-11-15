"""
Caption generation routes.
"""
import json
import asyncio
import logging
from pathlib import Path
from flask import Blueprint, request, jsonify
from utils.image_utils import load_image
from app.utils import extract_precision_params
from app.utils.route_decorators import handle_route_errors

logger = logging.getLogger(__name__)

bp = Blueprint('captions', __name__)


def init_routes(model_manager, async_session_manager):
    """Initialize routes with dependencies."""

    @bp.route('/generate', methods=['POST'])
    @handle_route_errors("generating caption")
    async def generate_caption():
        """Generate caption for a single image (async, non-blocking)."""
        image_id = request.form.get('image_id', '')
        logger.info("Single generate request - image_id: %s", image_id)

        # Async DB lookup for image path
        if image_id:
            image_path = await async_session_manager.get_image_path(image_id)
            if not image_path:
                logger.error("Image not found for image_id: %s", image_id)
                return jsonify({"error": "Image not found"}), 404
            image_source, filename = image_path, Path(image_path).name
        elif request.form.get('image_path', ''):
            image_path = request.form.get('image_path')
            image_source, filename = image_path, Path(image_path).name
        elif 'image' in request.files:
            image_file = request.files['image']
            if not image_file.filename:
                raise ValueError("No image file selected")
            image_source, filename = image_file.read(), image_file.filename
        else:
            logger.error("No valid image source provided. Form: %s", request.form)
            raise ValueError("No image_id, image file, or path provided")

        model_name = request.form.get('model', 'blip')
        prompt = request.form.get('prompt', '') or None

        try:
            parameters = json.loads(request.form.get('parameters', '{}'))
        except json.JSONDecodeError:
            parameters = {}

        precision_params = extract_precision_params(model_name, parameters)
        image = load_image(image_source)
        model_adapter = model_manager.get_model(model_name, precision_params)

        if not model_adapter or not model_adapter.is_loaded():
            raise Exception(f"Model {model_name} not available")

        # AI inference (GPU, potentially slow)
        caption = model_adapter.generate_caption(image, prompt, parameters)

        # Save caption in background (non-blocking, fire-and-forget)
        if image_id:
            asyncio.create_task(async_session_manager.save_caption(image_id, caption))

        return {
            "caption": caption,
            "model": model_adapter.model_name,
            "parameters_used": parameters,
            "image_id": request.form.get('image_id', ''),
            "image_filename": filename or request.form.get('image_filename', '')
        }

    @bp.route('/generate/batch', methods=['POST'])
    @handle_route_errors("generating batch captions")
    async def generate_captions_batch():
        """Generate captions for multiple images (async, optimized for concurrency)."""
        import time
        start_time = time.time()

        data = request.get_json()
        image_ids = data.get('image_ids', [])
        model_name = data.get('model', 'blip')
        prompt = data.get('prompt', '') or None
        parameters = data.get('parameters', {})

        if not image_ids:
            raise ValueError("No image_ids provided")

        # Load all image paths concurrently (async batch operation)
        image_paths_dict = await async_session_manager.get_image_paths_batch(image_ids)

        # Load images and prepare data
        images = []
        filenames = []
        valid_image_ids = []
        for image_id in image_ids:
            image_path = image_paths_dict.get(image_id)
            if image_path:
                images.append(load_image(image_path))
                filenames.append(Path(image_path).name)
                valid_image_ids.append(image_id)

        if not images:
            return jsonify({"error": "No valid images found"}), 404

        # Get model
        precision_params = extract_precision_params(model_name, parameters)
        model_adapter = model_manager.get_model(model_name, precision_params)

        if not model_adapter or not model_adapter.is_loaded():
            raise Exception(f"Model {model_name} not available")

        # Generate captions for batch (GPU inference)
        prompts = [prompt] * len(images) if prompt else None
        captions = model_adapter.generate_captions_batch(images, prompts, parameters)

        elapsed = time.time() - start_time
        logger.info("Batch: %d images with %s → %d captions (%.1fs)",
                   len(images), model_name, len(captions), elapsed)

        # Save all captions concurrently (async batch operation)
        captions_data = [
            {"image_id": img_id, "caption": cap}
            for img_id, cap in zip(valid_image_ids, captions)
        ]
        asyncio.create_task(async_session_manager.save_captions_batch(captions_data))

        # Build results
        results = []
        for image_id, caption, filename in zip(valid_image_ids, captions, filenames):
            results.append({
                "image_id": image_id,
                "caption": caption,
                "image_filename": filename
            })

        return {
            "results": results,
            "model": model_adapter.model_name,
            "parameters_used": parameters
        }

    return bp
