"""Hermes Domain Models - Business entities and domain logic."""
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class ResponseMode(str, Enum):
    """Response mode enumeration."""
    TEXT = "text"
    TTS = "tts"


class UserIdentity(BaseModel):
    """User identity model with fingerprinting."""
    user_id: str = Field(..., description="Unique user identifier hash")
    ip_address: str = Field(..., description="User IP address")
    user_agent: str = Field(..., description="User agent string")
    accept_language: str = Field(default="en-US", description="Accept language header")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "user_id": "abc123hash",
                "ip_address": "192.168.1.1",
                "user_agent": "Mozilla/5.0...",
                "accept_language": "en-US",
                "timestamp": "2024-01-01T00:00:00Z"
            }
        }


class ConversationMessage(BaseModel):
    """Individual conversation message."""
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

    @field_validator('role')
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Validate role is one of the allowed values."""
        allowed_roles = ['user', 'assistant', 'system']
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of {allowed_roles}")
        return v

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "Hello, how are you?",
                "timestamp": "2024-01-01T00:00:00Z",
                "metadata": {"source": "web"}
            }
        }


class ConversationContext(BaseModel):
    """Conversation context for a user session."""
    user_id: str = Field(..., description="User identifier")
    messages: list[ConversationMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a message to the conversation."""
        message = ConversationMessage(role=role, content=content, metadata=metadata)
        self.messages.append(message)
        self.updated_at = datetime.utcnow()

    def get_recent_messages(self, limit: int = 10) -> list[ConversationMessage]:
        """Get the most recent messages."""
        return self.messages[-limit:]

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "user_id": "abc123",
                "messages": [],
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
                "metadata": {}
            }
        }


class GeminiResponse(BaseModel):
    """Response from Gemini AI service."""
    content: str = Field(..., description="Generated text response")
    user_id: str = Field(..., description="User who made the request")
    prompt: str = Field(..., description="Original user prompt")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_used: Optional[str] = Field(default=None, description="AI model identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "content": "AI generated response",
                "user_id": "abc123",
                "prompt": "User's question",
                "timestamp": "2024-01-01T00:00:00Z",
                "model_used": "gemini-pro",
                "metadata": {"tokens": 150}
            }
        }


class TTSResponse(BaseModel):
    """Response from Text-to-Speech service."""
    text: str = Field(..., description="Original text that was synthesized")
    audio_url: str = Field(..., description="Cloud storage URL for the audio file")
    duration_seconds: Optional[float] = Field(default=None, description="Audio duration")
    format: str = Field(default="wav", description="Audio format")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "text": "Hello world",
                "audio_url": "https://storage.googleapis.com/bucket/audio.wav",
                "duration_seconds": 2.5,
                "format": "wav",
                "timestamp": "2024-01-01T00:00:00Z"
            }
        }


class ProcessRequestResult(BaseModel):
    """Result from processing a user request."""
    message: str = Field(..., description="Response message")
    response_mode: ResponseMode = Field(..., description="Response delivery mode")
    audio_url: Optional[str] = Field(default=None, description="Audio URL if TTS mode")
    tts_provider: Optional[str] = Field(
        default=None,
        description="TTS provider (elevenlabs, google, chatterbox)"
    )
    user_id: str = Field(..., description="User identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "message": "AI response",
                "response_mode": "text",
                "audio_url": None,
                "tts_provider": None,
                "user_id": "abc123",
                "timestamp": "2024-01-01T00:00:00Z",
                "metadata": {}
            }
        }

