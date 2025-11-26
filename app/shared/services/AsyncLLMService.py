"""
Async wrapper for LLM services.

Provides async interface for LLM calls to prevent blocking in async context.
Wraps the existing GeminiService for backwards compatibility.
"""

import asyncio
import logging
from functools import lru_cache
from typing import Optional

from app.shared.services.GeminiService import GeminiService

logger = logging.getLogger(__name__)


class AsyncLLMService:
    """
    Async wrapper for Gemini LLM service.

    This service wraps synchronous LLM calls in asyncio.to_thread() to prevent
    blocking the event loop in async contexts while maintaining backward compatibility.
    """

    def __init__(self, gemini_service: Optional[GeminiService] = None):
        """
        Initialize async LLM service.

        Args:
            gemini_service: Optional GeminiService instance. If None, creates one.
        """
        self._gemini_service = gemini_service or GeminiService()
        logger.info("AsyncLLMService initialized")

    async def generate_async(
        self, prompt: str, persona: str = "hermes", **kwargs
    ) -> str:
        """
        Generate LLM response asynchronously.

        This wraps the synchronous generate_gemini_response in a thread pool
        to prevent blocking the event loop.

        Args:
            prompt: The prompt to send to the LLM
            persona: The AI persona to use (default: "hermes")
            **kwargs: Additional arguments passed to generate_gemini_response

        Returns:
            Generated response string

        Raises:
            Exception: If LLM generation fails
        """
        try:
            # Execute blocking call in thread pool
            response = await asyncio.to_thread(
                self._gemini_service.generate_gemini_response, prompt, persona, **kwargs
            )
            return response
        except Exception as e:
            logger.error(f"Async LLM generation failed: {e}")
            raise

    def generate_sync(self, prompt: str, persona: str = "hermes", **kwargs) -> str:
        """
        Synchronous fallback for backward compatibility.

        Args:
            prompt: The prompt to send to the LLM
            persona: The AI persona to use
            **kwargs: Additional arguments

        Returns:
            Generated response string
        """
        return self._gemini_service.generate_gemini_response(prompt, persona, **kwargs)

    @property
    def gemini_service(self) -> GeminiService:
        """Access underlying GeminiService for direct operations if needed."""
        return self._gemini_service


@lru_cache(maxsize=1)
def get_async_llm_service() -> AsyncLLMService:
    """
    Get or create the global AsyncLLMService singleton.

    Returns:
        AsyncLLMService instance
    """
    return AsyncLLMService()
