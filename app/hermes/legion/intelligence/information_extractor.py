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

    def infer_defaults(
        self,
        required_info: Dict[str, RequiredInfoField],
        task_description: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Infer reasonable defaults for missing required fields.

        Uses task description and conversation context to make intelligent
        assumptions rather than requiring explicit user input.

        This addresses Issue 4: prefer giving answers quickly with reasonable
        defaults over asking for clarification.

        Args:
            required_info: Dictionary of required information fields
            task_description: The task being performed
            conversation_history: Optional conversation history for context

        Returns:
            Dictionary of inferred default values
        """
        if not required_info:
            return {}

        # Build inference prompt
        fields_description = "\n".join(
            [
                f"- {field.field_name} ({field.field_type}): {field.question}"
                for field in required_info.values()
            ]
        )

        # Include conversation history for better inference
        history_text = ""
        if conversation_history:
            history_text = "\n".join(
                [
                    f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                    for msg in conversation_history[-5:]
                ]
            )
            history_text = f"Previous conversation:\n{history_text}\n\n"

        prompt = f"""{history_text}You are an intelligent inference system. Infer reasonable defaults for missing information.

Task: "{task_description}"

Missing fields that need reasonable defaults:
{fields_description}

**Your Philosophy:**
- ALWAYS prefer making a reasonable inference over returning null
- Use context clues from the task and conversation to inform your defaults
- When multiple options are valid, pick the most commonly useful one
- Only return null if there is genuinely no reasonable default

**Common Inference Rules:**
- time_period: "trends" = last 6 months, "history" = comprehensive, otherwise = recent
- programming_language: Default to Python unless context suggests otherwise
- depth/scope: Infer from query complexity - simple queries = concise, complex = comprehensive
- format: Default to the most commonly expected format for the task type
- count/quantity: Default to a reasonable number (e.g., "top 5", "recent 10")

Respond with ONLY valid JSON: {{"field_name": "inferred_value", ...}}
Return a value for EVERY field - use null only as last resort."""

        try:
            response = self.gemini_service.generate_gemini_response(
                prompt, persona="hermes"
            )
            from app.hermes.legion.utils.llm_utils import extract_json_from_llm_response

            inferred = extract_json_from_llm_response(response)

            # Log what we inferred
            if inferred:
                logger.info(
                    f"Inferred defaults for missing fields: {list(inferred.keys())}"
                )

            # Filter out null values
            return {k: v for k, v in inferred.items() if v is not None}
        except Exception as e:
            logger.error(f"Default inference failed: {e}")
            return {}
