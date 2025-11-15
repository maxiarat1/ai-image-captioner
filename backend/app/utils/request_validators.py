"""
Request data extraction and validation utilities.

Provides declarative validators for common request patterns, eliminating
boilerplate code for extracting and validating request data.

Example usage:
    from app.utils.request_validators import extract_json_fields, RequestField

    data = extract_json_fields(
        RequestField('folder_path', required=True),
        RequestField('recursive', default=False)
    )
    folder_path = data['folder_path']
    recursive = data['recursive']
"""

import json
import logging
from typing import Any, Dict, Optional, Callable, List
from flask import request

logger = logging.getLogger(__name__)


class RequestField:
    """
    Declarative field definition for request data extraction.

    Defines how to extract, validate, and transform a single field
    from request data (JSON body, form data, etc.).

    Args:
        name: Field name in the request data
        required: Whether field must be present and non-empty
        default: Default value if field is missing or empty
        transform: Optional function to transform the value
        validator: Optional function to validate the value (return True if valid)
        error_message: Custom error message for required field validation

    Example:
        RequestField('user_id', required=True, error_message="User ID is required")
        RequestField('limit', default=10, validator=lambda x: x > 0)
        RequestField('email', transform=str.lower, validator=is_valid_email)
    """

    def __init__(
        self,
        name: str,
        *,
        required: bool = False,
        default: Any = None,
        transform: Optional[Callable[[Any], Any]] = None,
        validator: Optional[Callable[[Any], bool]] = None,
        error_message: Optional[str] = None
    ):
        self.name = name
        self.required = required
        self.default = default
        self.transform = transform
        self.validator = validator
        self.error_message = error_message or f"No {name} provided"

    def extract_and_validate(self, source: Dict[str, Any]) -> Any:
        """
        Extract and validate this field from a data source.

        Args:
            source: Dictionary to extract field from

        Returns:
            Extracted and validated field value

        Raises:
            ValueError: If field is required but missing, or validation fails
        """
        # Get value with default
        value = source.get(self.name, self.default)

        # Check required (treat empty strings as missing)
        if self.required:
            if value is None or value == '' or (isinstance(value, (list, dict)) and not value):
                raise ValueError(self.error_message)

        # Skip transformation and validation if value is None or default
        if value is None:
            return value

        # Transform value
        if self.transform:
            try:
                value = self.transform(value)
            except Exception as e:
                logger.warning(f"Transform failed for field '{self.name}': {e}")
                raise ValueError(f"Invalid format for {self.name}")

        # Validate
        if self.validator:
            try:
                if not self.validator(value):
                    raise ValueError(f"Invalid {self.name}")
            except TypeError as e:
                logger.warning(f"Validator error for field '{self.name}': {e}")
                raise ValueError(f"Invalid {self.name}")

        return value


def extract_json_fields(*fields: RequestField) -> Dict[str, Any]:
    """
    Extract and validate fields from JSON request body.

    Extracts multiple fields from request.get_json() using declarative
    field definitions. Handles validation, defaults, and transformations.

    Args:
        *fields: RequestField definitions to extract

    Returns:
        Dictionary mapping field names to extracted values

    Raises:
        ValueError: If required field missing or validation fails

    Example:
        data = extract_json_fields(
            RequestField('folder_path', required=True),
            RequestField('recursive', default=False),
            RequestField('max_depth', default=10, validator=lambda x: x > 0)
        )
        # data = {'folder_path': '/path/to/folder', 'recursive': False, 'max_depth': 10}
    """
    source = request.get_json() or {}
    result = {}

    for field in fields:
        result[field.name] = field.extract_and_validate(source)

    return result


def extract_form_fields(*fields: RequestField) -> Dict[str, Any]:
    """
    Extract and validate fields from form data (multipart/form-data).

    Similar to extract_json_fields but works with request.form instead
    of request.get_json().

    Args:
        *fields: RequestField definitions to extract

    Returns:
        Dictionary mapping field names to extracted values

    Raises:
        ValueError: If required field missing or validation fails

    Example:
        data = extract_form_fields(
            RequestField('image_id', required=True),
            RequestField('model', default='blip')
        )
    """
    source = dict(request.form)
    result = {}

    for field in fields:
        result[field.name] = field.extract_and_validate(source)

    return result


def extract_query_params(*fields: RequestField) -> Dict[str, Any]:
    """
    Extract and validate fields from query parameters.

    Args:
        *fields: RequestField definitions to extract

    Returns:
        Dictionary mapping field names to extracted values

    Raises:
        ValueError: If required field missing or validation fails

    Example:
        data = extract_query_params(
            RequestField('page', default=1, transform=int),
            RequestField('per_page', default=50, transform=int)
        )
    """
    source = dict(request.args)
    result = {}

    for field in fields:
        result[field.name] = field.extract_and_validate(source)

    return result


def parse_json_param(param_name: str, default: Any = None, source: str = 'form') -> Any:
    """
    Parse a JSON-encoded parameter with error handling.

    Useful for parsing complex data structures passed as JSON strings
    in form data or query parameters.

    Args:
        param_name: Parameter name
        default: Default value if parsing fails or param missing
        source: Where to get the parameter from ('form', 'args', 'json')

    Returns:
        Parsed JSON value or default

    Example:
        # Parse JSON from form data
        parameters = parse_json_param('parameters', {})

        # Parse JSON from query params
        filters = parse_json_param('filters', [], source='args')
    """
    if source == 'form':
        raw_value = request.form.get(param_name)
    elif source == 'args':
        raw_value = request.args.get(param_name)
    elif source == 'json':
        data = request.get_json() or {}
        raw_value = data.get(param_name)
    else:
        raise ValueError(f"Invalid source: {source}")

    if raw_value is None:
        return default

    try:
        return json.loads(raw_value)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"JSON decode failed for '{param_name}': {e}")
        return default


# ============================================================================
# Common Field Validators
# ============================================================================

def non_empty_list(value: Any) -> bool:
    """Validator: Ensure value is a non-empty list."""
    return isinstance(value, list) and len(value) > 0


def non_empty_dict(value: Any) -> bool:
    """Validator: Ensure value is a non-empty dict."""
    return isinstance(value, dict) and len(value) > 0


def non_empty_string(value: Any) -> bool:
    """Validator: Ensure value is a non-empty string."""
    return isinstance(value, str) and len(value.strip()) > 0


def positive_int(value: Any) -> bool:
    """Validator: Ensure value is a positive integer."""
    try:
        return int(value) > 0
    except (ValueError, TypeError):
        return False


def non_negative_int(value: Any) -> bool:
    """Validator: Ensure value is a non-negative integer."""
    try:
        return int(value) >= 0
    except (ValueError, TypeError):
        return False


def is_bool(value: Any) -> bool:
    """Validator: Ensure value is a boolean."""
    return isinstance(value, bool)


# ============================================================================
# Common Field Transformers
# ============================================================================

def or_none(value: Any) -> Any:
    """Transform: Convert empty string to None."""
    return value if value else None


def to_int(value: Any) -> int:
    """Transform: Convert value to integer."""
    return int(value)


def to_float(value: Any) -> float:
    """Transform: Convert value to float."""
    return float(value)


def to_bool(value: Any) -> bool:
    """Transform: Convert value to boolean (handles string 'true'/'false')."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    return bool(value)


def strip_whitespace(value: Any) -> str:
    """Transform: Strip leading/trailing whitespace from string."""
    if isinstance(value, str):
        return value.strip()
    return value


# ============================================================================
# Convenience Functions
# ============================================================================

def require_json_fields(*field_names: str) -> Dict[str, Any]:
    """
    Quick helper to require multiple fields from JSON body.

    Shorthand for common case where you just need to ensure
    fields exist without complex validation.

    Args:
        *field_names: Names of required fields

    Returns:
        Dictionary with extracted field values

    Raises:
        ValueError: If any required field is missing

    Example:
        data = require_json_fields('user_id', 'action', 'timestamp')
        # Same as:
        # data = extract_json_fields(
        #     RequestField('user_id', required=True),
        #     RequestField('action', required=True),
        #     RequestField('timestamp', required=True)
        # )
    """
    fields = [RequestField(name, required=True) for name in field_names]
    return extract_json_fields(*fields)


def get_json_field(name: str, default: Any = None, required: bool = False) -> Any:
    """
    Extract a single field from JSON body.

    Shorthand for extracting one field without declaring RequestField.

    Args:
        name: Field name
        default: Default value if missing
        required: Whether field is required

    Returns:
        Field value or default

    Raises:
        ValueError: If field is required but missing

    Example:
        user_id = get_json_field('user_id', required=True)
        limit = get_json_field('limit', default=10)
    """
    field = RequestField(name, required=required, default=default)
    data = extract_json_fields(field)
    return data[name]
