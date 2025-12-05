"""Information extraction intelligence."""

import logging
from typing import Any, Dict, List, Optional

from app.hermes.legion.models import RequiredInfoField
from app.shared.utils.service_loader import get_gemini_service

logger = logging.getLogger(__name__)


class InformationExtractor:
    """Extracts information from user messages."""

    def __init__(self):
        """Initialize information extractor."""
        self._gemini_service = None

    @property
    def gemini_service(self):
        """Lazy load Gemini service."""
        if self._gemini_service is None:
            self._gemini_service = get_gemini_service()
        return self._gemini_service

    def extract_information(
        self,
        user_message: str,
        required_info: Dict[str, RequiredInfoField],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Extract information from user message based on required fields.

        Args:
            user_message: User's message
            required_info: Dictionary of required information fields
            conversation_history: Optional conversation history for context

        Returns:
            Dictionary of extracted information
        """
        if not required_info:
            return {}

        # Build extraction prompt
        fields_description = "\n".join(
            [
                f"- {field.field_name} ({field.field_type}): {field.question}"
                for field in required_info.values()
            ]
        )

        # Include conversation history if provided
        history_text = ""
        if conversation_history:
            history_text = "\n".join(
                [
                    f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                    for msg in conversation_history[-5:]  # Last 5 messages
                ]
            )
            history_text = f"Previous conversation:\n{history_text}\n\n"

        prompt = f"""{history_text}Extract the following information from the user's message:

Required fields:
{fields_description}

User message: "{user_message}"

For each field, extract the value if mentioned, otherwise return null.
Respond in JSON format: {{"field_name": "value or null", ...}}"""

        try:
            response = self.gemini_service.generate_gemini_response(
                prompt, persona="hermes"
            )
            # Parse JSON response using robust helper
            from app.hermes.legion.utils.llm_utils import extract_json_from_llm_response

            extracted = extract_json_from_llm_response(response)

            # Filter out null values
            return {k: v for k, v in extracted.items() if v is not None}
        except Exception as e:
            logger.error(f"Information extraction failed: {e}")
            return {}
