"""Hermes Business Logic Services - Orchestrates domain operations."""
from typing import Optional
import logging

from .models import (
    ProcessRequestResult,
    GeminiResponse,
    UserIdentity,
    ResponseMode,
    ConversationContext
)
from .exceptions import (
    HermesServiceError,
    InvalidRequestError,
    AIServiceError,
    TTSServiceError
)
from app.shared.utils.service_loader import get_gemini_service, get_tts_service

logger = logging.getLogger(__name__)


class HermesService:
    """Core Hermes service orchestrating AI and TTS operations."""

    def __init__(self):
        """Initialize the Hermes service."""
        self._gemini_service = None
        self._tts_service = None
        self._conversation_contexts = {}  # In-memory store (use Redis in production)

    @property
    def gemini_service(self):
        """Lazy load Gemini service."""
        if self._gemini_service is None:
            self._gemini_service = get_gemini_service()
        return self._gemini_service

    @property
    def tts_service(self):
        """Lazy load TTS service."""
        if self._tts_service is None:
            self._tts_service = get_tts_service()
        return self._tts_service

    def process_request(
        self,
        text: str,
        user_identity: UserIdentity,
        response_mode: ResponseMode = ResponseMode.TEXT,
        persona: str = 'hermes'
    ) -> ProcessRequestResult:
        """
        Process a user request through the Hermes AI pipeline.
        
        Args:
            text: User's input text
            user_identity: User identity information
            response_mode: How to return the response (text or TTS)
            persona: Which AI persona to use ('hermes' or 'prisma')
            
        Returns:
            ProcessRequestResult with the AI response
            
        Raises:
            InvalidRequestError: If the request is invalid
            AIServiceError: If AI generation fails
            TTSServiceError: If TTS generation fails
        """
        try:
            if not text or not text.strip():
                raise InvalidRequestError("Request text cannot be empty")

            logger.info(f"Processing request for user {user_identity.user_id}")

            # Generate AI response
            gemini_response = self._generate_ai_response(text, user_identity.user_id, persona)

            # Generate TTS if requested
            audio_url = None
            tts_provider = None
            if response_mode == ResponseMode.TTS:
                audio_url, tts_provider = self.generate_tts(gemini_response.content)

            # Build result
            result = ProcessRequestResult(
                message=gemini_response.content,
                response_mode=response_mode,
                audio_url=audio_url,
                user_id=user_identity.user_id,
                tts_provider=tts_provider,
                metadata={
                    "model": gemini_response.model_used,
                    "prompt_length": len(text),
                    "response_length": len(gemini_response.content)
                }
            )

            logger.info(f"Request processed successfully for user {user_identity.user_id}")
            return result

        except InvalidRequestError:
            raise
        except (AIServiceError, TTSServiceError) as e:
            logger.error(f"Service error processing request: {e}")
            raise
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Error processing request: {e}")
            raise HermesServiceError(f"Failed to process request: {str(e)}")

    def chat(
        self,
        message: str,
        user_identity: UserIdentity,
        include_context: bool = True,
        persona: str = 'hermes'
    ) -> GeminiResponse:
        """
        Handle a chat message with conversation context.
        
        Args:
            message: User's chat message
            user_identity: User identity information
            include_context: Whether to include conversation history
            persona: Which AI persona to use ('hermes' or 'prisma')
            
        Returns:
            GeminiResponse with the AI reply
            
        Raises:
            InvalidRequestError: If the message is invalid
            AIServiceError: If AI generation fails
        """
        try:
            if not message or not message.strip():
                raise InvalidRequestError("Chat message cannot be empty")

            user_id = user_identity.user_id
            logger.info(f"Processing chat for user {user_id}")

            # Get or create conversation context
            if include_context:
                context = self._get_conversation_context(user_id)
                context.add_message("user", message)
            
            # Generate AI response
            response = self._generate_ai_response(message, user_id, persona)

            # Update context
            if include_context:
                context.add_message("assistant", response.content)
                self._save_conversation_context(user_id, context)

            logger.info(f"Chat processed successfully for user {user_id}")
            return response

        except InvalidRequestError:
            raise
        except (AIServiceError, ValueError, TypeError) as e:
            logger.error(f"Error processing chat: {e}")
            raise HermesServiceError(f"Failed to process chat: {str(e)}")

    def _generate_ai_response(self, prompt: str, user_id: str, persona: str = 'hermes') -> GeminiResponse:
        """
        Generate AI response using Gemini.
        
        Args:
            prompt: User's prompt
            user_id: User identifier
            persona: Which AI persona to use ('hermes' or 'prisma')
            
        Returns:
            GeminiResponse with generated content
            
        Raises:
            AIServiceError: If AI generation fails
        """
        try:
            result = self.gemini_service.generate_gemini_response_with_rag(
                prompt=prompt,
                user_id=user_id,
                persona=persona
            )

            return GeminiResponse(
                content=result,
                user_id=user_id,
                prompt=prompt,
                model_used="gemini-pro",
                metadata={}
            )

        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.error(f"AI generation failed: {e}")
            raise AIServiceError(f"Failed to generate AI response: {str(e)}")

    def generate_tts(self, text: str) -> tuple[str, str]:
        """
        Generate Text-to-Speech audio (public method).
        
        Args:
            text: Text to synthesize
            
        Returns:
            Tuple of (cloud_url, tts_provider)
            
        Raises:
            TTSServiceError: If TTS generation fails
        """
        try:
            tts_result = self.tts_service.generate_audio(text)
            tts_provider = self.tts_service.tts_provider
            return tts_result['cloud_url'], tts_provider

        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"TTS generation failed: {e}")
            raise TTSServiceError(f"Failed to generate audio: {str(e)}")

    def _get_conversation_context(self, user_id: str) -> ConversationContext:
        """Get or create conversation context for a user."""
        if user_id not in self._conversation_contexts:
            self._conversation_contexts[user_id] = ConversationContext(user_id=user_id)
        return self._conversation_contexts[user_id]

    def _save_conversation_context(self, user_id: str, context: ConversationContext) -> None:
        """Save conversation context."""
        self._conversation_contexts[user_id] = context

    def clear_conversation_context(self, user_id: str) -> None:
        """Clear conversation history for a user."""
        if user_id in self._conversation_contexts:
            del self._conversation_contexts[user_id]
            logger.info(f"Cleared conversation context for user {user_id}")


# Singleton instance
_hermes_service: Optional[HermesService] = None


def get_hermes_service() -> HermesService:
    """Get or create the Hermes service singleton."""
    global _hermes_service
    if _hermes_service is None:
        _hermes_service = HermesService()
    return _hermes_service

