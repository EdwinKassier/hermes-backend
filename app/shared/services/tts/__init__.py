"""
Text-to-Speech service module.

This module provides modular TTS provider implementations and utilities.

The main TTSService class is in the parent module (TTSService.py) to maintain
the conventional import path: from app.shared.services.TTSService import TTSService

This module contains:
- BaseTTSProvider: Abstract base class for all providers
- Provider implementations: ElevenLabs, Google
- Utilities: RateLimiter, CircuitBreaker, SimpleCache

Usage:
    # Standard import (recommended)
    from app.shared.services.TTSService import TTSService

    # Or via factory function
    from app.shared.utils.service_loader import get_tts_service
    tts = get_tts_service()
"""

# Only export components from this submodule
# TTSService is imported directly from app.shared.services.TTSService
from .base_provider import BaseTTSProvider

__all__ = [
    "BaseTTSProvider",
]
