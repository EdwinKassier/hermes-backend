"""Prism Domain Constants - Following Attendee best practices."""
from enum import Enum

# Attendee API Configuration
# Official API base URL from https://docs.attendee.dev/api-reference
ATTENDEE_API_BASE_URL = "https://app.attendee.dev"
ATTENDEE_API_VERSION = "api/v1"

# Audio Configuration (per Attendee documentation)
AUDIO_SAMPLE_RATE = 16000  # 16kHz PCM
AUDIO_BIT_DEPTH = 16  # 16-bit
AUDIO_CHANNELS = 1  # Mono
AUDIO_ENCODING = "pcm16"  # Base64-encoded 16-bit PCM

# WebSocket Configuration
WEBSOCKET_PING_INTERVAL = 30  # seconds
WEBSOCKET_TIMEOUT = 300  # 5 minutes
MAX_AUDIO_CHUNK_SIZE = 4096  # bytes

# Session Configuration
MAX_CONVERSATION_HISTORY = 20  # Keep last 20 transcript entries
SESSION_CLEANUP_TIMEOUT = 600  # 10 minutes after session ends

# Gemini Configuration
PRISM_AI_RESPONSE_THRESHOLD = 1  # Number of transcript entries before considering response (respond quickly to any input)
PRISM_MAX_TOKENS = 1000
PRISM_TEMPERATURE = 0.7


class BotState(str, Enum):
    """Attendee bot states per official documentation."""
    IDLE = "idle"
    JOINING = "joining"
    IN_MEETING = "in_meeting"
    LEAVING = "leaving"
    ERROR = "error"


class SessionStatus(str, Enum):
    """Prism session lifecycle states."""
    CREATED = "created"
    BOT_CREATING = "bot_creating"
    BOT_JOINING = "bot_joining"
    ACTIVE = "active"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    CLOSED = "closed"


class WebhookTrigger(str, Enum):
    """Attendee webhook triggers per documentation."""
    BOT_STATE_CHANGE = "bot.state_change"
    TRANSCRIPT_UPDATE = "transcript.update"
    BOT_ERROR = "bot.error"


class AudioMessageType(str, Enum):
    """WebSocket message types for bot audio connection."""
    AUDIO_CHUNK = "audio_chunk"
    TRANSCRIPT = "transcript"
    STATUS = "status"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"


# Error Messages
ERROR_MISSING_MEETING_URL = "meeting_url is required"
ERROR_INVALID_MEETING_URL = "Invalid Google Meet URL format"
ERROR_BOT_CREATION_FAILED = "Failed to create Attendee bot"
ERROR_SESSION_NOT_FOUND = "Session not found"
ERROR_AUDIO_PROCESSING_FAILED = "Audio processing failed"
ERROR_WEBSOCKET_CONNECTION_FAILED = "WebSocket connection failed"

