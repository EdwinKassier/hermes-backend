"""Orchestration utilities for sub-agent coordination."""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from app.shared.utils.service_loader import get_gemini_service

from .models import RequiredInfoField

logger = logging.getLogger(__name__)


class TaskIdentifier:
    """Identifies task types from user messages."""

    def __init__(self):
        """Initialize task identifier."""
        self._gemini_service = None

    @property
    def gemini_service(self):
        """Lazy load Gemini service."""
        if self._gemini_service is None:
            self._gemini_service = get_gemini_service()
        return self._gemini_service

    def identify_task_type(self, user_message: str) -> Optional[str]:
        """
        Identify task type from user message using AI.

        Args:
            user_message: User's message

        Returns:
            Task type string (research, code, analysis) or None for general conversation
        """
        prompt = f"""Analyze this user message and identify if it requires a specialized agent.

User message: "{user_message}"

Task types:
- research: Research, investigation, finding information
- code: Code generation, programming, technical implementation
- analysis: Data analysis, evaluation, assessment
- general: General conversation, questions, chat

Respond with ONLY the task type (research, code, analysis, or general)."""

        try:
            response = self.gemini_service.generate_gemini_response(
                prompt, persona="hermes"
            )
            task_type = response.strip().lower()
            if task_type in ["research", "code", "analysis"]:
                return task_type
            return None
        except Exception as e:
            logger.error(f"Task identification failed: {e}")
            return None


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
            from .utils.llm_utils import extract_json_from_llm_response

            extracted = extract_json_from_llm_response(response)

            # Filter out null values
            return {k: v for k, v in extracted.items() if v is not None}
        except Exception as e:
            logger.error(f"Information extraction failed: {e}")
            return {}


class IntentDetector:
    """Detects user intent in messages."""

    def __init__(self):
        """Initialize intent detector."""
        self._gemini_service = None

    @property
    def gemini_service(self):
        """Lazy load Gemini service."""
        if self._gemini_service is None:
            self._gemini_service = get_gemini_service()
        return self._gemini_service

    def is_user_answering_question(
        self, user_message: str, pending_questions: List[str]
    ) -> bool:
        """
        Detect if user is answering questions or asking a new question.

        Args:
            user_message: User's message
            pending_questions: List of pending questions

        Returns:
            True if user is answering, False if asking new question
        """
        if not pending_questions:
            return False

        questions_text = "\n".join([f"- {q}" for q in pending_questions])

        prompt = f"""Is the user answering the pending questions or asking a new question?

Pending questions:
{questions_text}

User message: "{user_message}"

Respond with: ANSWERING or NEW_QUESTION"""

        try:
            response = self.gemini_service.generate_gemini_response(
                prompt, persona="hermes"
            )
            return "ANSWERING" in response.upper()
        except Exception as e:
            logger.error(f"Intent detection failed: {e}")
            # Default to answering if detection fails
            return True

    def is_cancellation_intent(self, user_message: str) -> bool:
        """
        Detect if user wants to cancel current task using AI.

        Args:
            user_message: User's message

        Returns:
            True if cancellation intent detected
        """
        # Validate input
        if not isinstance(user_message, str):
            logger.warning(
                f"Invalid input type for cancellation check: {type(user_message)}"
            )
            return False

        user_message = user_message.strip()
        if not user_message:
            return False

        try:
            prompt = f"""Does the user want to CANCEL or STOP the current task?

User message: "{user_message}"

Examples of CANCELLATION intent:
✓ "cancel"
✓ "never mind"
✓ "forget it"
✓ "actually, let's not do this"
✓ "stop this task"
✓ "abort"

Examples of NOT cancellation:
✗ "don't worry, I can wait" (reassurance, not cancellation)
✗ "I never mind waiting" (patience, not cancellation)
✗ "stop here and show me what you have" (wants to see progress)
✗ "don't forget to include X" (instruction, not cancellation)

Respond with ONLY: "CANCEL" or "CONTINUE"
"""

            response = self.gemini_service.generate_gemini_response(
                prompt, persona="hermes"
            )

            is_cancel = "CANCEL" in response.upper()
            logger.info(
                f"AI cancellation detection: {is_cancel} for '{user_message[:50]}'"
            )
            return is_cancel

        except (ConnectionError, TimeoutError) as e:
            logger.warning(
                f"Network error in cancellation detection: {e}, using fallback"
            )
            return self._fallback_cancellation_detection(user_message)

        except Exception as e:
            logger.error(f"AI cancellation detection failed: {e}, using fallback")
            return self._fallback_cancellation_detection(user_message)

    def _fallback_cancellation_detection(self, user_message: str) -> bool:
        """Fallback keyword-based cancellation detection."""
        cancellation_keywords = [
            "cancel",
            "never mind",
            "forget it",
            "stop",
            "abort",
            "don't",
            "skip",
        ]
        message_lower = user_message.lower()
        return any(keyword in message_lower for keyword in cancellation_keywords)
