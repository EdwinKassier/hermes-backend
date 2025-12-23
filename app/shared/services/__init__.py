"""
Services package for the application.

This package contains various service implementations that provide business logic
and integration with external systems.
"""

from .CloudStorageService import CloudStorageService

# Import services to make them available when importing from app.services
from .IdentityService import IdentityService
from .LLMService import (  # Backward compat aliases
    GeminiService,
    GeminiServiceError,
    LLMService,
    LLMServiceError,
    PersonaConfig,
)
from .TTSService import TTSService

__all__ = [
    "IdentityService",
    "CloudStorageService",
    "LLMService",
    "LLMServiceError",
    "PersonaConfig",
    "GeminiService",  # Backward compat alias
    "GeminiServiceError",  # Backward compat alias
    "TTSService",
]
