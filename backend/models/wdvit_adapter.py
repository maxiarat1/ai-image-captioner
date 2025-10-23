import torch
from utils.torch_utils import pick_device
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from .base_adapter import BaseModelAdapter
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class WdVitAdapter(BaseModelAdapter):
    """WD Tagger v3 adapter for anime-style image tagging (supports ViT and EVA02 variants)"""

    def __init__(self, model_id="SmilingWolf/wd-vit-large-tagger-v3"):
        super().__init__(model_id)
        self.device = pick_device(torch)
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.id2tag = None
        self.tags_df = None  # Store full tags dataframe for category info
        # Determine tags CSV URL based on model variant
        if "eva02" in model_id:
            self.tags_csv_url = "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/raw/main/selected_tags.csv"
        else:
            self.tags_csv_url = "https://huggingface.co/SmilingWolf/wd-vit-large-tagger-v3/raw/main/selected_tags.csv"

    def load_model(self) -> None:
        """Load WD Tagger model, processor, and tags CSV"""
        try:
            logger.info("Loading %s tagger model on %s…", self.model_name, self.device)

            # Load tags CSV
            logger.info("Loading tags from CSV…")
            self.tags_df = pd.read_csv(self.tags_csv_url)
            self.id2tag = dict(zip(self.tags_df.index, self.tags_df["name"]))
            logger.info("Loaded %d tags", len(self.id2tag))

            # Load processor
            logger.info("Loading image processor…")
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Load model
            logger.info("Loading model…")
            self.model = AutoModelForImageClassification.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=self.dtype
            )
            self.model.to(self.device)
            self.model.eval()

            logger.info("%s tagger model loaded successfully", self.model_name)

        except Exception as e:
            logger.exception("Error loading %s model: %s", self.model_name, e)
            raise

    def generate_caption(self, image: Image.Image, prompt: str = None, parameters: dict = None) -> str:
        """
        Generate tags for the given image.
        Note: This model generates tags, not captions. The 'prompt' parameter is ignored.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Debug logging
            logger.debug("WdVit.generate_caption | params=%s", parameters)

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Parse parameters with defaults
            params = parameters or {}
            threshold = params.get('threshold', 0.3)
            character_threshold = params.get('character_threshold', threshold)
            exclude_tags = params.get('exclude_tags', '')
            use_character_tags = params.get('use_character_tags', True)
            use_rating_tags = params.get('use_rating_tags', False)
            replace_underscores = params.get('replace_underscores', False)
            add_confidence = params.get('add_confidence', False)
            sort_by = params.get('sort_by', 'confidence')
            limit = params.get('limit', 0)
            tag_separator = params.get('tag_separator', ', ')

            logger.debug("Using threshold: %.2f, character_threshold: %.2f", threshold, character_threshold)

            # Parse exclude_tags into a set
            excluded_set = set()
            if exclude_tags and exclude_tags.strip():
                excluded_set = {tag.strip().lower() for tag in exclude_tags.split(',')}

            # Process image
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            # Run inference
            with torch.no_grad():
                probs = torch.sigmoid(self.model(**inputs).logits[0]).cpu()

            # Build tags dict with category information
            tags = {}
            for i in range(len(probs)):
                if i not in self.id2tag:
                    continue

                tag_name = self.id2tag[i]
                confidence = probs[i].item()

                # Get tag category from dataframe if available
                category = 'general'
                if self.tags_df is not None and i < len(self.tags_df):
                    category_col = self.tags_df.iloc[i].get('category', 'general')
                    if pd.notna(category_col):
                        category = str(category_col).lower()

                # Apply category filters
                if category == 'character' and not use_character_tags:
                    continue
                if category == 'rating' and not use_rating_tags:
                    continue

                # Apply appropriate threshold based on category
                tag_threshold = character_threshold if category == 'character' else threshold
                if confidence <= tag_threshold:
                    continue

                # Apply exclude filter
                if tag_name.lower() in excluded_set:
                    continue

                tags[tag_name] = {
                    'confidence': confidence,
                    'category': category
                }

            # Sort tags
            if sort_by == 'alphabetical':
                sorted_tags = sorted(tags.items(), key=lambda x: x[0])
            elif sort_by == 'category':
                # Sort by category order (rating, character, general), then by confidence within each
                category_order = {'rating': 0, 'character': 1, 'general': 2}
                sorted_tags = sorted(tags.items(),
                                   key=lambda x: (category_order.get(x[1]['category'], 999), -x[1]['confidence']))
            else:  # confidence (default)
                sorted_tags = sorted(tags.items(), key=lambda x: x[1]['confidence'], reverse=True)

            # Apply limit
            if limit > 0:
                sorted_tags = sorted_tags[:limit]

            # Format output tags
            output_tags = []
            for tag_name, tag_info in sorted_tags:
                # Replace underscores if requested
                display_name = tag_name.replace('_', ' ') if replace_underscores else tag_name

                # Add confidence if requested
                if add_confidence:
                    output_tags.append(f"{display_name}:{tag_info['confidence']:.2f}")
                else:
                    output_tags.append(display_name)

            # Join with separator
            result = tag_separator.join(output_tags)
            logger.debug("Generated %d tags", len(output_tags))

            return result if result else "No tags detected above threshold"

        except Exception as e:
            logger.exception("Error generating tags: %s", e)
            return f"Error: {str(e)}"

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return (self.model is not None and
                self.processor is not None and
                self.id2tag is not None)

    def get_available_parameters(self) -> list:
        """Get list of available parameters for WD-ViT model"""
        return [
            {
                "name": "Threshold",
                "param_key": "threshold",
                "type": "number",
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "description": "Minimum confidence score for general tags to be included (0.0-1.0)"
            },
            {
                "name": "Character Threshold",
                "param_key": "character_threshold",
                "type": "number",
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "description": "Separate confidence threshold for character tags (uses general threshold if not set)"
            },
            {
                "name": "Exclude Tags",
                "param_key": "exclude_tags",
                "type": "text",
                "description": "Comma-separated list of tags to exclude from output (e.g., 'black eyes, black hair')"
            },
            {
                "name": "Use Character Tags",
                "param_key": "use_character_tags",
                "type": "checkbox",
                "description": "Include character tags in the output"
            },
            {
                "name": "Use Rating Tags",
                "param_key": "use_rating_tags",
                "type": "checkbox",
                "description": "Include rating tags (general, sensitive, questionable, explicit) in the output"
            },
            {
                "name": "Replace Underscores",
                "param_key": "replace_underscores",
                "type": "checkbox",
                "description": "Replace underscores with spaces in tag names"
            },
            {
                "name": "Add Confidence",
                "param_key": "add_confidence",
                "type": "checkbox",
                "description": "Add confidence scores to each tag in the output (e.g., 'tag_name:0.85')"
            },
            {
                "name": "Sort By",
                "param_key": "sort_by",
                "type": "select",
                "options": [
                    {"value": "confidence", "label": "Confidence (high to low)"},
                    {"value": "alphabetical", "label": "Alphabetical"},
                    {"value": "category", "label": "By Category (rating, character, general)"}
                ],
                "description": "How to sort the output tags"
            },
            {
                "name": "Limit",
                "param_key": "limit",
                "type": "number",
                "min": 0,
                "max": 500,
                "step": 1,
                "description": "Maximum number of tags to return (0 = no limit)"
            },
            {
                "name": "Tag Separator",
                "param_key": "tag_separator",
                "type": "text",
                "description": "String to use for separating tags in output (default: ', ')"
            }
        ]

    def unload(self) -> None:
        """Unload model and clear tags dictionary"""
        if hasattr(self, 'id2tag'):
            self.id2tag = None
        if hasattr(self, 'tags_df'):
            self.tags_df = None
        super().unload()
