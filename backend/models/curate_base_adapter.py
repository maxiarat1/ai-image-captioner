from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class CurateBaseAdapter(ABC):
    """
    Base adapter for routing/curation models.

    Unlike caption generation models, curate adapters analyze images and/or captions
    to make routing decisions, directing workflow to different output ports.
    """

    def __init__(self, model_name: str, model_type: str = 'vlm'):
        """
        Initialize curate adapter.

        Args:
            model_name: Name of the model
            model_type: Type of curation ('vlm', 'classification', 'analyzer')
        """
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.processor = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the model and processor."""
        pass

    @abstractmethod
    def route_image(self,
                   image: Image.Image,
                   caption: Optional[str],
                   port_configs: List[Dict[str, Any]],
                   parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Analyze image and/or caption and return the port ID to route to.

        Args:
            image: PIL Image to analyze
            caption: Optional caption/text context
            port_configs: List of port configurations with id, label, and instruction
            parameters: Optional model parameters

        Returns:
            port_id: The ID of the port to route this image to

        The implementation should:
        1. Analyze the image and/or caption based on port instructions
        2. Determine which port best matches the content
        3. Return the matching port's ID
        4. If no match, return the default port's ID
        """
        pass

    def route_images_batch(self,
                          images: List[Image.Image],
                          captions: Optional[List[str]],
                          port_configs: List[Dict[str, Any]],
                          parameters: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Route multiple images. Default implementation processes sequentially.
        Subclasses can override for optimized batch processing.

        Args:
            images: List of PIL Images
            captions: Optional list of captions corresponding to images
            port_configs: Port configurations
            parameters: Optional model parameters

        Returns:
            List of port IDs corresponding to each image
        """
        if captions is None:
            captions = [None] * len(images)

        return [
            self.route_image(img, cap, port_configs, parameters)
            for img, cap in zip(images, captions)
        ]

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": self.model_name,
            "type": self.model_type,
            "loaded": self.is_loaded(),
            "parameters": self.get_available_parameters()
        }

    def get_available_parameters(self) -> list:
        """
        Get available parameters for this curate model.
        Can be overridden by subclasses.
        """
        return []

    @staticmethod
    def _ensure_rgb(images):
        """Convert image(s) to RGB mode if needed"""
        if isinstance(images, list):
            return [img.convert('RGB') if img.mode != 'RGB' else img for img in images]
        return images.convert('RGB') if images.mode != 'RGB' else images

    def _get_default_port(self, port_configs: List[Dict[str, Any]]) -> Optional[str]:
        """
        Get the default/fallback port from port configurations.

        Args:
            port_configs: List of port configurations

        Returns:
            Port ID of the default port, or first port if no default is set
        """
        # Find port marked as default
        for port in port_configs:
            if port.get('isDefault', False):
                return port.get('id')

        # If no default is marked, use first port as fallback
        if port_configs:
            return port_configs[0].get('id')

        return None

    def _build_routing_prompt(self,
                             port_configs: List[Dict[str, Any]],
                             context: Optional[str] = None) -> str:
        """
        Build a prompt for VLM-based routing.

        Args:
            port_configs: Port configurations with instructions
            context: Optional context (e.g., existing caption)

        Returns:
            Formatted prompt for the model
        """
        prompt_parts = [
            "Analyze this image and determine which category it belongs to.",
            ""
        ]

        if context:
            prompt_parts.append(f"Image description: {context}")
            prompt_parts.append("")

        prompt_parts.append("Available categories:")
        for i, port in enumerate(port_configs):
            label = port.get('label', f'Port {i+1}')
            instruction = port.get('instruction', '')
            prompt_parts.append(f"- {label}: {instruction}")

        prompt_parts.append("")
        prompt_parts.append(
            "Respond with ONLY the exact category name (e.g., 'Route A' or 'High Quality'). "
            "Do not add any explanation or additional text."
        )

        return "\n".join(prompt_parts)

    def _parse_routing_response(self,
                               response: str,
                               port_configs: List[Dict[str, Any]]) -> Optional[str]:
        """
        Parse model response to extract port ID.

        Args:
            response: Raw model response
            port_configs: Port configurations

        Returns:
            Port ID if matched, None otherwise
        """
        response_lower = response.strip().lower()

        # Try exact label match (case-insensitive)
        for port in port_configs:
            label = port.get('label', '').lower()
            if label and label in response_lower:
                return port.get('id')

        # Try finding any port label as substring
        for port in port_configs:
            label = port.get('label', '').lower()
            if label and label in response_lower:
                return port.get('id')

        return None
