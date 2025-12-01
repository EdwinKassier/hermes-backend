"""Hermes Request/Response Schemas - Pydantic validation for API endpoints."""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator

from .constants import MAX_PROMPT_LENGTH


# Request Schemas
class ProcessRequestSchema(BaseModel):
    """Schema for processing a user request (works for both query params and JSON body)."""

    request_text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_PROMPT_LENGTH,
        description="The text to process through the AI",
    )
    response_mode: str = Field(
        default="text", description="How to deliver the response (text or tts)"
    )
    persona: Optional[str] = Field(
        default="hermes", description="Which AI persona to use ('hermes' or 'prisma')"
    )
    legion_mode: Optional[bool] = Field(
        default=False, description="Whether to use legion processing mode"
    )

    @field_validator("request_text")
    @classmethod
    def validate_request_text(cls, v: str) -> str:
        """Validate and clean request text."""
        if not v or not v.strip():
            raise ValueError("Request text cannot be empty or whitespace only")
        return v.strip()

    @field_validator("response_mode")
    @classmethod
    def validate_response_mode(cls, v: str) -> str:
        """Validate response mode."""
        if v not in ["text", "tts"]:
            raise ValueError("Response mode must be 'text' or 'tts'")
        return v

    @field_validator("persona")
    @classmethod
    def validate_persona(cls, v: Optional[str]) -> str:
        """Validate persona."""
        if v is not None and v not in ["hermes", "prisma", "legion"]:
            raise ValueError("Persona must be 'hermes', 'prisma', or 'legion'")
        return v or "hermes"

    @field_validator("legion_mode", mode="before")
    @classmethod
    def validate_legion_mode(cls, v) -> bool:
        """Convert string booleans from query params to actual booleans."""
        if isinstance(v, str):
            v_lower = v.lower().strip()
            if v_lower in ("true", "1", "yes"):
                return True
            if v_lower in ("false", "0", "no", ""):
                return False
            # For any other string value, default to False
            return False
        if v is None:
            return False
        return bool(v)

    model_config = {
        "json_schema_extra": {
            "example": {
                "request_text": "Tell me about artificial intelligence",
                "response_mode": "text",
                "persona": "hermes",
                "legion_mode": False,
            }
        }
    }


class ChatMessageSchema(BaseModel):
    """Schema for chat message requests."""

    message: str = Field(
        ..., min_length=1, max_length=MAX_PROMPT_LENGTH, description="The chat message"
    )
    include_context: bool = Field(
        default=True, description="Whether to include conversation history"
    )
    persona: Optional[str] = Field(
        default="hermes", description="Which AI persona to use ('hermes' or 'prisma')"
    )
    legion_mode: Optional[bool] = Field(
        default=False, description="Whether to use legion processing mode"
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Validate and clean message."""
        if not v or not v.strip():
            raise ValueError("Message cannot be empty or whitespace only")
        return v.strip()

    @field_validator("persona")
    @classmethod
    def validate_persona(cls, v: Optional[str]) -> str:
        """Validate persona."""
        if v is not None and v not in ["hermes", "prisma", "legion"]:
            raise ValueError("Persona must be 'hermes', 'prisma', or 'legion'")
        return v or "hermes"

    @field_validator("legion_mode", mode="before")
    @classmethod
    def validate_legion_mode(cls, v) -> bool:
        """Convert string booleans from query params to actual booleans."""
        if isinstance(v, str):
            v_lower = v.lower().strip()
            if v_lower in ("true", "1", "yes"):
                return True
            if v_lower in ("false", "0", "no", ""):
                return False
            # For any other string value, default to False
            return False
        if v is None:
            return False
        return bool(v)

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "Hello, how can you help me today?",
                "include_context": True,
                "persona": "hermes",
                "legion_mode": False,
            }
        }
    }


# Response Schemas
class ProcessRequestResponseSchema(BaseModel):
    """Schema for process request response."""

    message: str = Field(..., description="The AI-generated response")
    response_mode: str = Field(..., description="Response delivery mode")
    wave_url: Optional[str] = Field(None, description="Audio URL if TTS mode")
    tts_provider: Optional[str] = Field(
        None, description="TTS provider (elevenlabs, google)"
    )
    user_id: str = Field(..., description="User identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "AI generated response",
                "response_mode": "text",
                "wave_url": None,
                "tts_provider": None,
                "user_id": "abc123",
                "timestamp": "2024-01-01T00:00:00Z",
                "metadata": {"model": "gemini-pro", "tokens": 150},
            }
        }
    }


class ChatResponseSchema(BaseModel):
    """Schema for chat response."""

    message: str = Field(..., description="The AI response")
    user_id: str = Field(..., description="User identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "I'm here to help! What would you like to know?",
                "user_id": "abc123",
                "timestamp": "2024-01-01T00:00:00Z",
                "metadata": {"model": "gemini-pro"},
            }
        }
    }


class HealthCheckResponseSchema(BaseModel):
    """Schema for health check response."""

    status: str = Field(..., description="Service health status")
    service: str = Field(..., description="Service name")
    message: str = Field(..., description="Status message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "service": "hermes",
                "message": "Hermes API is running",
                "timestamp": "2024-01-01T00:00:00Z",
            }
        }
    }


class ErrorResponseSchema(BaseModel):
    """Schema for error responses."""

    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional error details"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "INVALID_REQUEST",
                "message": "Request text cannot be empty",
                "details": {"field": "request_text"},
                "timestamp": "2024-01-01T00:00:00Z",
            }
        }
    }


class AuthResponseSchema(BaseModel):
    """Schema for authentication response."""

    message: str = Field(..., description="Authentication result message")
    authenticated: bool = Field(..., description="Whether authentication succeeded")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "Successful Auth",
                "authenticated": True,
                "timestamp": "2024-01-01T00:00:00Z",
            }
        }
    }


class FileListResponseSchema(BaseModel):
    """Schema for file list response."""

    files: list[str] = Field(..., description="List of files")
    count: int = Field(..., description="Number of files")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = {
        "json_schema_extra": {
            "example": {
                "files": ["file1.txt", "file2.txt"],
                "count": 2,
                "timestamp": "2024-01-01T00:00:00Z",
            }
        }
    }


# Note: ProcessRequestSchema above handles both query params and JSON body
# No need for a separate ProcessRequestQueryParams class
