"""Topic change detection using AI semantic comparison."""

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class TopicChangeDetector:
    """
    Detects when user changes topic mid-conversation using LLM semantic analysis.

    This enables dynamic re-planning when conversations shift direction.
    """

    def __init__(self, gemini_service=None):
        """
        Initialize topic change detector.

        Args:
            gemini_service: Optional GeminiService instance. If None, will lazy load.
        """
        self._gemini_service = gemini_service

    @property
    def gemini_service(self):
        """Lazy load Gemini service."""
        if self._gemini_service is None:
            from ...shared.services.GeminiService import get_gemini_service

            self._gemini_service = get_gemini_service()
        return self._gemini_service

    async def detect_topic_change(
        self,
        current_task_description: str,
        new_user_message: str,
        conversation_history: Optional[list] = None,
    ) -> dict:
        """
        Detect if new message represents a topic change from current task.

        Args:
            current_task_description: Description of the current active task
            new_user_message: Latest message from user
            conversation_history: Optional recent conversation context

        Returns:
            Dict with:
                - is_topic_change: bool
                - confidence: float (0.0-1.0)
                - reason: str explaining the detection
                - new_topic: Optional[str] detected new topic
        """
        prompt = self._build_detection_prompt(
            current_task_description, new_user_message, conversation_history
        )

        try:
            response = await self.gemini_service.generate_text_async(
                prompt=prompt,
                temperature=0.1,  # Low temperature for consistent analysis
                max_output_tokens=200,
            )

            return self._parse_detection_response(response)

        except Exception as e:
            logger.error(f"Topic change detection failed: {e}")
            # Fallback: assume no topic change on error
            return {
                "is_topic_change": False,
                "confidence": 0.0,
                "reason": f"Detection failed: {str(e)}",
                "new_topic": None,
            }

    def _build_detection_prompt(
        self, current_task: str, new_message: str, history: Optional[list]
    ) -> str:
        """Build prompt for LLM topic change detection."""

        context = ""
        if history and len(history) > 0:
            recent = history[-3:]  # Last 3 messages
            context = "\n".join(
                [
                    f"{msg.get('role', 'unknown')}: {msg.get('content', '')[:100]}"
                    for msg in recent
                ]
            )

        return f"""Analyze if the user's new message represents a topic change from their current task.

CURRENT TASK:
{current_task}

NEW USER MESSAGE:
{new_message}

{f"RECENT CONVERSATION:\\n{context}\\n" if context else ""}

Determine:
1. Is this a topic change? (YES/NO)
2. Confidence level (0.0-1.0)
3. Brief reason
4. If yes, what is the new topic?

Examples of topic changes:
- "Actually, let me do something else instead..."
- "Never mind that, I need help with X"
- Completely unrelated new request

Examples of NOT topic changes:
- Follow-up questions about current task
- Clarifications or refinements
- Providing requested information

Respond in this exact format:
TOPIC_CHANGE: [YES/NO]
CONFIDENCE: [0.0-1.0]
REASON: [brief explanation]
NEW_TOPIC: [new topic if YES, otherwise NONE]"""

    def _parse_detection_response(self, response: str) -> dict:
        """Parse LLM response into structured result."""
        lines = response.strip().split("\n")

        result = {
            "is_topic_change": False,
            "confidence": 0.0,
            "reason": "Unable to parse response",
            "new_topic": None,
        }

        for line in lines:
            line = line.strip()

            if line.startswith("TOPIC_CHANGE:"):
                value = line.split(":", 1)[1].strip().upper()
                result["is_topic_change"] = value == "YES"

            elif line.startswith("CONFIDENCE:"):
                try:
                    value = line.split(":", 1)[1].strip()
                    result["confidence"] = float(value)
                except ValueError:
                    result["confidence"] = 0.5  # Default if parse fails

            elif line.startswith("REASON:"):
                result["reason"] = line.split(":", 1)[1].strip()

            elif line.startswith("NEW_TOPIC:"):
                value = line.split(":", 1)[1].strip()
                result["new_topic"] = None if value.upper() == "NONE" else value

        logger.info(
            f"Topic change detection: {result['is_topic_change']} "
            f"(confidence: {result['confidence']:.2f})"
        )

        return result

    def should_trigger_replan(
        self, detection_result: dict, confidence_threshold: float = 0.7
    ) -> bool:
        """
        Determine if detection result should trigger re-planning.

        Args:
            detection_result: Result from detect_topic_change()
            confidence_threshold: Minimum confidence to trigger (default 0.7)

        Returns:
            True if should replan, False otherwise
        """
        return (
            detection_result.get("is_topic_change", False)
            and detection_result.get("confidence", 0.0) >= confidence_threshold
        )
