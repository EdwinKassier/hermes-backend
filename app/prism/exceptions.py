"""Prism Domain Exceptions - Following patterns from hermes/exceptions.py"""


class PrismException(Exception):
    """Base exception for Prism domain."""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class BotCreationError(PrismException):
    """Raised when Attendee bot creation fails."""

    def __init__(self, message: str = "Failed to create Attendee bot"):
        super().__init__(message, status_code=502)


class SessionNotFoundError(PrismException):
    """Raised when session ID doesn't exist."""

    def __init__(self, session_id: str):
        super().__init__(f"Session not found: {session_id}", status_code=404)


class AudioProcessingError(PrismException):
    """Raised when audio format conversion fails."""

    def __init__(self, message: str = "Audio processing failed"):
        super().__init__(message, status_code=500)


class InvalidMeetingURLError(PrismException):
    """Raised when meeting URL is invalid."""

    def __init__(self, message: str = "Invalid Google Meet URL"):
        super().__init__(message, status_code=400)


class WebSocketConnectionError(PrismException):
    """Raised when WebSocket connection fails."""

    def __init__(self, message: str = "WebSocket connection failed"):
        super().__init__(message, status_code=503)


class AttendeeAPIError(PrismException):
    """Raised when Attendee API returns an error."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(f"Attendee API error: {message}", status_code=status_code)
