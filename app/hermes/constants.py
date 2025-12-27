"""Hermes Domain Constants - Configuration and constant values."""

from enum import Enum

# API Version
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}/hermes"


# Response Modes
class ResponseModeType(str, Enum):
    """Response mode types."""

    TEXT = "text"
    TTS = "tts"


# AI Model Configuration
DEFAULT_AI_MODEL = "gemini-pro"
MAX_PROMPT_LENGTH = 10000  # characters
MAX_RESPONSE_LENGTH = 5000  # characters

# Conversation Configuration
MAX_CONVERSATION_HISTORY = 50  # messages
DEFAULT_CONVERSATION_LIMIT = 10  # messages to retrieve

# TTS Configuration
DEFAULT_TTS_PROVIDER = "elevenlabs"  # or "google"
DEFAULT_AUDIO_FORMAT = "wav"
MAX_TTS_TEXT_LENGTH = 5000  # characters

# HTTP Status Codes
HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_INTERNAL_ERROR = 500

# Error Messages
ERROR_EMPTY_REQUEST = "Request text cannot be empty"
ERROR_REQUEST_BODY_REQUIRED = "Request body is required"
ERROR_VALIDATION_FAILED = "Request validation failed"
ERROR_UNEXPECTED = "An unexpected error occurred"

# Success Messages
SUCCESS_REQUEST_PROCESSED = "Request processed successfully"
SUCCESS_CHAT_PROCESSED = "Chat processed successfully"
SUCCESS_CONTEXT_CLEARED = "Conversation context cleared"
SUCCESS_AUTH = "Successful Auth"
SUCCESS_VECTOR_SYNC = "Vector sync completed successfully"

# Vector Sync Configuration
DEFAULT_GCS_BUCKET = "ashes-project-hermes-training"
MAX_SYNC_DOCUMENTS = 1000
EMBEDDING_BATCH_SIZE = 50
ERROR_VECTOR_SYNC_FAILED = "Vector sync operation failed"
