"""Shared utilities for LLM response parsing."""

import json
import logging
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)


def extract_json_from_llm_response(response: str) -> Dict[str, Any]:
    """
    Extract and parse JSON from LLM response.

    Many LLM responses include JSON embedded in markdown or text.
    This utility extracts the JSON portion and parses it.

    Args:
        response: Raw LLM response string

    Returns:
        Parsed JSON as dictionary

    Raises:
        ValueError: If no valid JSON found in response
        json.JSONDecodeError: If JSON is malformed

    Example:
        >>> response = "Here's the data: {'key': 'value'}"
        >>> extract_json_from_llm_response(response)
        {'key': 'value'}
    """
    if not response or not isinstance(response, str):
        raise ValueError("Response must be a non-empty string")

    # Try to find JSON object in response
    json_match = re.search(r"\{.*\}", response, re.DOTALL)

    if not json_match:
        raise ValueError(f"No JSON found in response: {response[:100]}...")

    json_str = json_match.group(0)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {json_str[:200]}...")
        raise ValueError(f"Invalid JSON in response: {e}") from e
