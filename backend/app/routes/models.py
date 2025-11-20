"""
Model information and management routes.
"""
import logging
from flask import Blueprint, request, jsonify
from app.utils.route_decorators import handle_route_errors

logger = logging.getLogger(__name__)

bp = Blueprint('models', __name__)


def init_routes(model_manager, categories, model_metadata):
    """Initialize routes with dependencies."""

    @bp.route('/health', methods=['GET'])
    @handle_route_errors("health check")
    def health_check():
        model_status = {f"{name}_loaded": model is not None and model.is_loaded()
                        for name, model in model_manager.models.items()}
        return {"status": "ok", "models_available": list(model_manager.models.keys()), **model_status}

    @bp.route('/model/info', methods=['GET'])
    @handle_route_errors("getting model info")
    def model_info():
        model_name = request.args.get('model', 'blip')
        info = model_manager.get_model_info(model_name)
        return info

    @bp.route('/models', methods=['GET'])
    @handle_route_errors("listing models")
    def list_models():
        from app.models import get_factory
        factory = get_factory()

        available_models = []
        for name in model_metadata.keys():
            model_config = factory.get_model_config(name) or {}

            available_models.append({
                "name": name,
                "loaded": model_manager.is_loaded(name),
                "description": model_metadata[name]['description'],
                "category": model_metadata[name].get('category', 'general'),
                "vlm_capable": model_metadata[name].get('vlm_capable', False),
                "curate_suitable": model_config.get('curate_suitable', True)
            })

        return {"models": available_models}

    @bp.route('/models/categories', methods=['GET'])
    @handle_route_errors("getting model categories")
    def get_model_categories():
        """Get models organized by categories with category metadata."""
        categories_data = []

        for category_id, category_info in categories.items():
            category_models = [
                {
                    "name": name,
                    "description": model_metadata[name]['description'],
                    "loaded": model_manager.is_loaded(name),
                    "vlm_capable": model_metadata[name].get('vlm_capable', False)
                }
                for name, metadata in model_metadata.items()
                if metadata.get('category') == category_id
            ]

            if category_models:
                categories_data.append({
                    "id": category_id,
                    "name": category_info['name'],
                    "icon": category_info['icon'],
                    "color": category_info['color'],
                    "description": category_info['description'],
                    "models": category_models
                })

        return {"categories": categories_data}

    @bp.route('/models/metadata', methods=['GET'])
    @handle_route_errors("getting models metadata")
    def models_metadata():
        """Get comprehensive metadata about all models for documentation."""
        model_details = {
            'blip': {'display_name': 'BLIP', 'full_name': 'Salesforce BLIP', 'description': 'Fast image captioning', 'speed_score': 80, 'vram_gb': 1.6, 'vram_label': '1.6GB', 'speed_label': 'Fast', 'features': ['Fast processing', 'Low VRAM usage', 'General-purpose captions']},
            'blip2': {'display_name': 'BLIP2', 'full_name': 'BLIP2-OPT-2.7B', 'description': 'Enhanced image captioning', 'speed_score': 80, 'vram_gb': 8.2, 'vram_label': '8.2GB', 'speed_label': 'Fast', 'features': ['Enhanced accuracy', 'Detailed captions', 'General-purpose']},
            'r4b': {'display_name': 'R-4B', 'full_name': 'R-4B Advanced Reasoning', 'description': 'Advanced reasoning model with configurable parameters', 'speed_score': 30, 'vram_gb': 21.9, 'vram_label': '21.9GB', 'speed_label': 'Slow', 'features': ['Advanced reasoning', 'Configurable precision', 'Detailed captions']},
            'wdvit': {'display_name': 'WD-ViT', 'full_name': 'WD-ViT Large Tagger v3', 'description': 'Anime-style image tagging model with ViT backbone', 'speed_score': 80, 'vram_gb': 2, 'vram_label': '2GB', 'speed_label': 'Fast', 'features': ['Anime/manga specialized', 'Tag-based output', 'Character recognition']},
            'wdeva02': {'display_name': 'WD-EVA02', 'full_name': 'WD-EVA02 Large Tagger v3', 'description': 'Anime-style image tagging model with EVA02 backbone (improved accuracy)', 'speed_score': 80, 'vram_gb': 2, 'vram_label': '2GB', 'speed_label': 'Fast', 'features': ['Enhanced accuracy', 'Anime/manga specialized', 'Advanced tagging']},
            'wd14-convnext': {'display_name': 'WD14 ConvNext', 'full_name': 'WD v1.4 ConvNext Tagger v2', 'description': 'Fast ONNX-based anime tagging', 'speed_score': 80, 'vram_gb': 1.7, 'vram_label': '1.7GB', 'speed_label': 'Fast', 'features': ['ONNX optimized', 'Fast inference', 'Anime tagging']},
            'janus-1.3b': {'display_name': 'Janus 1.3B', 'full_name': 'Janus 1.3B', 'description': 'Multimodal vision-language model', 'speed_score': 30, 'vram_gb': 5.4, 'vram_label': '5.4GB', 'speed_label': 'Slow', 'features': ['Multimodal', 'Efficient architecture', 'Vision-language']},
            'janus-pro-1b': {'display_name': 'Janus Pro 1B', 'full_name': 'Janus Pro 1B', 'description': 'Compact professional-grade vision model', 'speed_score': 30, 'vram_gb': 5.4, 'vram_label': '5.4GB', 'speed_label': 'Slow', 'features': ['Professional grade', 'Compact', 'Vision understanding']},
            'janus-pro-7b': {'display_name': 'Janus Pro 7B', 'full_name': 'Janus Pro 7B', 'description': 'Advanced multimodal model with superior reasoning', 'speed_score': 30, 'vram_gb': 16.1, 'vram_label': '16.1GB', 'speed_label': 'Slow', 'features': ['Superior reasoning', 'Advanced multimodal', 'High quality']},
            'lfm2-vl-3b': {'display_name': 'LFM2-VL-3B', 'full_name': 'LiquidAI LFM2-VL-3B', 'description': 'Vision-language model with chat capabilities', 'speed_score': 30, 'vram_gb': 7.2, 'vram_label': '7.2GB', 'speed_label': 'Slow', 'features': ['Chat capabilities', 'Vision-language', 'Conversational']},
            'llava-phi3': {'display_name': 'LLaVA-Phi3', 'full_name': 'LLaVA-Phi-3-Mini', 'description': 'Compact efficient vision-language model', 'speed_score': 30, 'vram_gb': 8.9, 'vram_label': '8.9GB', 'speed_label': 'Slow', 'features': ['Compact', 'Efficient', 'Vision-language']},
            'nanonets-ocr-s': {'display_name': 'Nanonets OCR-S', 'full_name': 'Nanonets OCR-S', 'description': 'Lightweight OCR for tables, equations, and HTML', 'speed_score': 50, 'vram_gb': 10.7, 'vram_label': '10.7GB', 'speed_label': 'Medium', 'features': ['Table extraction', 'Equation recognition', 'HTML output']},
            'chandra-ocr': {'display_name': 'Chandra OCR', 'full_name': 'Chandra OCR', 'description': 'Advanced layout-aware text extraction', 'speed_score': 30, 'vram_gb': 18.6, 'vram_label': '18.6GB', 'speed_label': 'Slow', 'features': ['Layout-aware', 'Table support', 'Equation recognition']},
            'trocr-large-printed': {'display_name': 'TrOCR Large Printed', 'full_name': 'Microsoft TrOCR Large Printed', 'description': 'Transformer-based OCR for printed text', 'speed_score': 50, 'vram_gb': 1.9, 'vram_label': '1.9GB', 'speed_label': 'Medium', 'features': ['Transformer-based', 'Printed text', 'High accuracy']},
            'vit-classifier': {'display_name': 'ViT Classifier', 'full_name': 'Google ViT Base', 'description': 'ImageNet classification model', 'speed_score': 80, 'vram_gb': 1.1, 'vram_label': '1.1GB', 'speed_label': 'Fast', 'features': ['1000 object classes', 'Fast classification', 'ImageNet trained']}
        }

        available_models = [name for name in model_metadata.keys()]
        active_models = {name: details for name, details in model_details.items() if name in available_models}

        return {
            'model_count': len(available_models),
            'models': active_models,
            'export_formats': 4,
            'vram_range': f"{min(m['vram_gb'] for m in active_models.values())}-{max(m['vram_gb'] for m in active_models.values())}" if active_models else "2-16",
            'tech_stack': [
                {'name': 'Salesforce BLIP', 'description': 'Fast image captioning'},
                {'name': 'R-4B', 'description': 'Advanced reasoning model'},
                {'name': 'WD Taggers', 'description': 'Anime-style tagging'},
                {'name': 'PyTorch', 'description': 'Deep learning framework'},
                {'name': 'Flask', 'description': 'REST API backend'},
                {'name': 'Vanilla JavaScript', 'description': 'No-build frontend'},
                {'name': 'CUDA', 'description': 'GPU acceleration'},
                {'name': 'DuckDB', 'description': 'Embedded analytics database'}
            ]
        }

    @bp.route('/model/reload', methods=['POST'])
    @handle_route_errors("reloading model")
    def reload_model():
        data = request.get_json() or {}
        model_name = data.get('model', 'r4b')
        precision_params = data.get('precision_params')

        model_adapter = model_manager.get_model(model_name, precision_params, force_reload=True)
        return {
            "success": True,
            "message": f"Model {model_name} reloaded successfully",
            "loaded": model_adapter.is_loaded()
        }

    @bp.route('/model/unload', methods=['POST'])
    @handle_route_errors("unloading model")
    def unload_model():
        data = request.get_json() or {}
        model_name = data.get('model', 'r4b')

        model_manager.unload_model(model_name)
        return {"success": True, "message": f"Model {model_name} unloaded successfully"}

    return bp
