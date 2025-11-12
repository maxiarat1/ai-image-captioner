from typing import Any, Dict, List, Optional
from PIL import Image
import logging
import torch
from .curate_base_adapter import CurateBaseAdapter

logger = logging.getLogger(__name__)

class VLMRouterAdapter(CurateBaseAdapter):
    """
    Visual Language Model Router Adapter.

    Uses a VLM (like BLIP2, DeepSeek-VL, etc.) to analyze images and make
    routing decisions based on custom instructions per port.
    """

    def __init__(self, base_model_name: str):
        """
        Initialize VLM Router using an existing caption model.

        Args:
            base_model_name: Name of the base VLM model to use (e.g., 'blip2-opt-2.7b')
        """
        super().__init__(base_model_name, model_type='vlm')
        self.base_adapter = None

    def load_model(self, precision_params: Optional[Dict] = None, get_model_func=None) -> None:
        """Load the underlying VLM model via its adapter.

        Args:
            precision_params: Optional precision parameters (precision, use_flash_attention, etc.)
            get_model_func: Optional model loading function (defaults to importing get_model)
        """
        if get_model_func is None:
            from app import get_model
            get_model_func = get_model

        try:
            # Get the base model adapter (e.g., BLIP2, DeepSeek-VL, etc.)
            # Pass precision_params for proper model loading configuration
            self.base_adapter = get_model_func(self.model_name, precision_params)
            if self.base_adapter and not self.base_adapter.is_loaded():
                self.base_adapter.load_model()
            logger.info(f"VLM Router loaded using base model: {self.model_name} with params: {precision_params}")
        except Exception as e:
            logger.error(f"Failed to load VLM router model {self.model_name}: {e}")
            raise

    def is_loaded(self) -> bool:
        """Check if the underlying model is loaded."""
        return self.base_adapter is not None and self.base_adapter.is_loaded()

    def route_image(self,
                   image: Image.Image,
                   caption: Optional[str],
                   port_configs: List[Dict[str, Any]],
                   parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Route image using VLM analysis.

        The VLM is prompted with routing instructions and analyzes the image
        to determine the appropriate route.

        Args:
            image: PIL Image to analyze
            caption: Optional existing caption
            port_configs: Port configurations with routing instructions
            parameters: Optional model parameters

        Returns:
            Port ID to route to
        """
        if not self.is_loaded():
            raise RuntimeError(f"VLM Router model {self.model_name} is not loaded")

        # Ensure RGB
        image = self._ensure_rgb(image)

        # Extract template from parameters if provided
        template = None
        if parameters:
            template = parameters.get('template', None)

        # Build routing prompt (with template if provided)
        routing_prompt = self._build_routing_prompt(port_configs, context=caption, template=template)

        try:
            # Use the base adapter to analyze the image with the routing prompt
            # Pass parameters for generation control
            if parameters is None:
                parameters = {}

            # Add low temperature for more deterministic routing
            if 'temperature' not in parameters:
                parameters['temperature'] = 0.3

            response = self.base_adapter.generate_caption(
                image,
                prompt=routing_prompt,
                parameters=parameters
            )

            logger.debug(f"VLM routing response: {response}")

            # Parse response to get port ID
            port_id = self._parse_routing_response(response, port_configs)

            if port_id is None:
                # No match found, use default port
                port_id = self._get_default_port(port_configs)
                logger.warning(
                    f"Could not parse routing decision from response '{response}'. "
                    f"Using default port: {port_id}"
                )

            return port_id

        except Exception as e:
            logger.error(f"Error during VLM routing: {e}")
            # On error, route to default port
            default_port = self._get_default_port(port_configs)
            logger.warning(f"Routing to default port {default_port} due to error")
            return default_port

    def route_images_batch(self,
                          images: List[Image.Image],
                          captions: Optional[List[str]],
                          port_configs: List[Dict[str, Any]],
                          parameters: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Route multiple images.

        For VLM routing, we process sequentially since each image may need
        different routing logic based on the response.

        Args:
            images: List of images
            captions: Optional captions
            port_configs: Port configurations
            parameters: Model parameters

        Returns:
            List of port IDs
        """
        if captions is None:
            captions = [None] * len(images)

        results = []
        for img, cap in zip(images, captions):
            try:
                port_id = self.route_image(img, cap, port_configs, parameters)
                results.append(port_id)
            except Exception as e:
                logger.error(f"Error routing image: {e}")
                # Use default port on error
                default_port = self._get_default_port(port_configs)
                results.append(default_port)

        return results

    def get_available_parameters(self) -> list:
        """
        Get available parameters for VLM routing.

        Inherits from the base model adapter's parameters.
        """
        if self.base_adapter:
            return self.base_adapter.get_available_parameters()
        return []

    def _build_routing_prompt(self,
                             port_configs: List[Dict[str, Any]],
                             context: Optional[str] = None,
                             template: Optional[str] = None) -> str:
        """
        Build an optimized routing prompt for VLMs.

        Args:
            port_configs: Port configurations
            context: Optional context
            template: Optional template string with placeholders

        Returns:
            Routing prompt
        """
        # If template is provided, use parent's template resolution logic
        if template and template.strip():
            return self._resolve_routing_template(template, port_configs, context)

        # Default optimized prompt for VLMs
        prompt_parts = [
            "Carefully analyze this image and determine which category it best fits into."
        ]

        if context:
            prompt_parts.append(f"\nAdditional context: {context}")

        prompt_parts.append("\nCategories:")
        for port in port_configs:
            label = port.get('label', 'Unknown')
            instruction = port.get('instruction', '').strip()

            if instruction:
                prompt_parts.append(f"- {label}: {instruction}")
            else:
                prompt_parts.append(f"- {label}")

        # Get default port for explicit mention
        default_port_id = self._get_default_port(port_configs)
        default_port = next((p for p in port_configs if p.get('id') == default_port_id), None)
        default_label = default_port.get('label', 'Unknown') if default_port else 'Unknown'

        prompt_parts.append(
            f"\nIMPORTANT: Respond with ONLY ONE category name exactly as shown above. "
            f"If uncertain, choose '{default_label}'. "
            f"Do not add explanation."
        )

        return "\n".join(prompt_parts)
