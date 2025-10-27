"""Hermes Domain Exceptions - Custom exception classes for error handling."""


class HermesError(Exception):
    """Base exception for all Hermes domain errors."""

    def __init__(self, message: str, code: str = "HERMES_ERROR", details: dict = None):
        """
        Initialize Hermes error.

        Args:
            message: Human-readable error message
            code: Machine-readable error code
            details: Additional error details
        """
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict:
        """Convert exception to dictionary for API responses."""
        return {"error": self.code, "message": self.message, "details": self.details}


class HermesServiceError(HermesError):
    """General service error for Hermes operations."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message=message, code="HERMES_SERVICE_ERROR", details=details)


class InvalidRequestError(HermesError):
    """Raised when a request is invalid or malformed."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message=message, code="INVALID_REQUEST", details=details)


class AIServiceError(HermesError):
    """Raised when AI service (Gemini) operations fail."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message=message, code="AI_SERVICE_ERROR", details=details)


class TTSServiceError(HermesError):
    """Raised when Text-to-Speech service operations fail."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message=message, code="TTS_SERVICE_ERROR", details=details)


class AuthenticationError(HermesError):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication required", details: dict = None):
        super().__init__(message=message, code="AUTHENTICATION_ERROR", details=details)


class RateLimitError(HermesError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", details: dict = None):
        super().__init__(message=message, code="RATE_LIMIT_EXCEEDED", details=details)


class ConversationNotFoundError(HermesError):
    """Raised when a conversation context is not found."""

    def __init__(self, user_id: str, details: dict = None):
        super().__init__(
            message=f"Conversation not found for user: {user_id}",
            code="CONVERSATION_NOT_FOUND",
            details=details or {"user_id": user_id},
        )


class ResourceNotFoundError(HermesError):
    """Raised when a requested resource is not found."""

    def __init__(self, resource_type: str, resource_id: str, details: dict = None):
        super().__init__(
            message=f"{resource_type} not found: {resource_id}",
            code="RESOURCE_NOT_FOUND",
            details=details
            or {"resource_type": resource_type, "resource_id": resource_id},
        )
