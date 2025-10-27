"""
Hermes Domain Module - Complete vertical slice of the Hermes AI service.

This module follows Domain-Driven Design principles and contains:
- Business Logic: models.py, services.py
- API Interfaces: routes.py (REST endpoints)
- Background Tasks: tasks.py (Celery async operations)
- Validation: schemas.py (Pydantic validation)
- Error Handling: exceptions.py
- Configuration: constants.py
"""

# Constants
from .constants import (
    API_PREFIX,
    API_VERSION,
    DEFAULT_AI_MODEL,
    ERROR_EMPTY_REQUEST,
    ERROR_REQUEST_BODY_REQUIRED,
    HTTP_BAD_REQUEST,
    HTTP_INTERNAL_ERROR,
    HTTP_OK,
    MAX_PROMPT_LENGTH,
    MAX_RESPONSE_LENGTH,
    SUCCESS_CHAT_PROCESSED,
    SUCCESS_REQUEST_PROCESSED,
    ResponseModeType,
)

# Exceptions
from .exceptions import (
    AIServiceError,
    AuthenticationError,
    ConversationNotFoundError,
    HermesError,
    HermesServiceError,
    InvalidRequestError,
    RateLimitError,
    ResourceNotFoundError,
    TTSServiceError,
)

# Models
from .models import (
    ConversationContext,
    ConversationMessage,
    GeminiResponse,
    ProcessRequestResult,
    ResponseMode,
    TTSResponse,
    UserIdentity,
)

# Routes (Blueprint)
from .routes import hermes

# Schemas
from .schemas import (
    AuthResponseSchema,
    ChatMessageSchema,
    ChatResponseSchema,
    ErrorResponseSchema,
    FileListResponseSchema,
    HealthCheckResponseSchema,
    ProcessRequestResponseSchema,
    ProcessRequestSchema,
)

# Services
from .services import HermesService, get_hermes_service

# Tasks
from .tasks import (
    cleanup_old_conversations,
    generate_tts_batch,
    health_check,
    log_analytics_event,
    process_request_async,
)

# Public API
__all__ = [
    # Blueprint
    "hermes",
    # Models
    "ResponseMode",
    "UserIdentity",
    "ConversationMessage",
    "ConversationContext",
    "GeminiResponse",
    "TTSResponse",
    "ProcessRequestResult",
    # Services
    "HermesService",
    "get_hermes_service",
    # Schemas
    "ProcessRequestSchema",
    "ChatMessageSchema",
    "ProcessRequestResponseSchema",
    "ChatResponseSchema",
    "HealthCheckResponseSchema",
    "ErrorResponseSchema",
    "AuthResponseSchema",
    "FileListResponseSchema",
    # Exceptions
    "HermesError",
    "HermesServiceError",
    "InvalidRequestError",
    "AIServiceError",
    "TTSServiceError",
    "AuthenticationError",
    "RateLimitError",
    "ConversationNotFoundError",
    "ResourceNotFoundError",
    # Constants
    "API_VERSION",
    "API_PREFIX",
    "ResponseModeType",
    "DEFAULT_AI_MODEL",
    "MAX_PROMPT_LENGTH",
    "MAX_RESPONSE_LENGTH",
    "HTTP_OK",
    "HTTP_BAD_REQUEST",
    "HTTP_INTERNAL_ERROR",
    "ERROR_EMPTY_REQUEST",
    "ERROR_REQUEST_BODY_REQUIRED",
    "SUCCESS_REQUEST_PROCESSED",
    "SUCCESS_CHAT_PROCESSED",
    # Tasks
    "process_request_async",
    "generate_tts_batch",
    "cleanup_old_conversations",
    "log_analytics_event",
    "health_check",
]

# Domain information
__domain__ = "hermes"
__version__ = "1.0.0"
__description__ = "Hermes AI service domain - conversational AI with RAG and TTS"
