"""
Resume method for LegionGraphService to continue from interrupts.

Add this method to the LegionGraphService class.
"""

import logging

from app.hermes.models import UserIdentity
from app.hermes.services import AIServiceError, GeminiResponse

logger = logging.getLogger(__name__)


def resume(
    self,
    user_identity: UserIdentity,
    resume_value: dict,
    persona: str = "hermes",
) -> GeminiResponse:
    """
    Resume graph execution from an interrupt.

    This method is called after a user approves/modifies an interrupt,
    passing their response back into the graph to continue execution.

    Args:
        user_identity: User identity information
        resume_value: User's response to the interrupt (e.g., {"action": "approve"})
        persona: AI persona

    Returns:
        GeminiResponse with continued execution or another interrupt

    Raises:
        AIServiceError: If resumption fails

    Example:
        >>> # After receiving interrupt
        >>> service.resume(
        ...     user_identity,
        ...     resume_value={"action": "approve"},
        ...     persona="hermes"
        ... )
    """
    from .utils.input_sanitizer import validate_user_id

    try:
        user_id = validate_user_id(user_identity.user_id)

        logger.info(
            f"Resuming conversation for user {user_id} with value: {resume_value}"
        )

        # Resume from checkpoint with user's response
        response_content, metadata = self._generate_ai_response_with_graph(
            prompt="",  # Prompt not needed when resuming
            user_id=user_id,
            persona=persona,
            resume_value=resume_value,
        )

        # Check if we hit another interrupt
        if response_content == "INTERRUPTED" and metadata.get("interrupted"):
            return GeminiResponse(
                content="",
                user_id=user_id,
                prompt="",
                model_used="gemini-pro",
                metadata={
                    "legion_mode": True,
                    "langgraph_enabled": True,
                    **metadata,
                },
            )

        # Execution completed
        return GeminiResponse(
            content=response_content,
            user_id=user_id,
            prompt="",  # Original prompt not available during resume
            model_used="gemini-pro",
            metadata={
                "legion_mode": True,
                "langgraph_enabled": True,
                "resumed": True,
                **metadata,
            },
        )

    except Exception as e:
        logger.error(f"Error resuming graph: {e}")
        raise AIServiceError(f"Failed to resume execution: {str(e)}")
