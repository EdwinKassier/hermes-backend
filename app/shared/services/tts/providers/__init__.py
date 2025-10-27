"""TTS Provider implementations."""

from .chatterbox_provider import (
    DEFAULT_AUDIO_PROMPT,
    DEFAULT_CFGW,
    DEFAULT_EXAGGERATION,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    ChatterboxTTSProvider,
)
from .elevenlabs_provider import (
    DEFAULT_MODEL_ID,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_VOICE_ID,
    ElevenLabsTTSProvider,
)
from .google_provider import GoogleTTSProvider

__all__ = [
    "ElevenLabsTTSProvider",
    "GoogleTTSProvider",
    "ChatterboxTTSProvider",
    # Constants
    "DEFAULT_VOICE_ID",
    "DEFAULT_MODEL_ID",
    "DEFAULT_OUTPUT_FORMAT",
    "DEFAULT_AUDIO_PROMPT",
    "DEFAULT_EXAGGERATION",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_SEED",
    "DEFAULT_CFGW",
]
