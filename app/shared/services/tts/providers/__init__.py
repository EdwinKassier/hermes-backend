"""TTS Provider implementations."""

from .elevenlabs_provider import (
    ElevenLabsTTSProvider,
    DEFAULT_VOICE_ID,
    DEFAULT_MODEL_ID,
    DEFAULT_OUTPUT_FORMAT
)
from .google_provider import GoogleTTSProvider
from .chatterbox_provider import (
    ChatterboxTTSProvider,
    DEFAULT_AUDIO_PROMPT,
    DEFAULT_EXAGGERATION,
    DEFAULT_TEMPERATURE,
    DEFAULT_SEED,
    DEFAULT_CFGW
)

__all__ = [
    'ElevenLabsTTSProvider',
    'GoogleTTSProvider',
    'ChatterboxTTSProvider',
    # Constants
    'DEFAULT_VOICE_ID',
    'DEFAULT_MODEL_ID',
    'DEFAULT_OUTPUT_FORMAT',
    'DEFAULT_AUDIO_PROMPT',
    'DEFAULT_EXAGGERATION',
    'DEFAULT_TEMPERATURE',
    'DEFAULT_SEED',
    'DEFAULT_CFGW',
]

