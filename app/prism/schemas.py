"""
Prism Domain Schemas - Request/response validation using Pydantic

Following patterns from hermes/schemas.py for consistency.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime
import re


class StartSessionRequest(BaseModel):
    """Request to start a new Prism session with a Google Meet URL."""
    meeting_url: str = Field(
        ...,
        description="Google Meet URL (https://meet.google.com/xxx-xxxx-xxx)"
    )

    @field_validator('meeting_url')
    @classmethod
    def validate_meeting_url(cls, v):
        """Validate Google Meet URL format."""
        # Google Meet URL pattern: https://meet.google.com/abc-defg-hij
        pattern = r'^https://meet\.google\.com/[a-z]{3}-[a-z]{4}-[a-z]{3}$'
        if not re.match(pattern, v):
            raise ValueError(
                'Invalid Google Meet URL format. '
                'Expected: https://meet.google.com/xxx-xxxx-xxx'
            )
        return v


class SessionStatusResponse(BaseModel):
    """Response with current session status (sent via user WebSocket)."""
    session_id: str
    status: str
    bot_state: Optional[str] = None
    bot_id: Optional[str] = None
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TranscriptMessage(BaseModel):
    """Transcript message from meeting (sent via user WebSocket)."""
    speaker: str
    text: str
    timestamp: datetime
    is_final: bool = True
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BotAudioMessage(BaseModel):
    """WebSocket message format for bot audio connection."""
    type: str  # "audio_chunk", "transcript", "status", "error", "ping", "pong"
    data: Optional[dict] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AttendeeWebhookPayload(BaseModel):
    """
    Webhook payload from Attendee API.
    
    Based on documentation examples:
    - bot.state_change: {"bot_id": "...", "state": "in_meeting"}
    - transcript.update: {"bot_id": "...", "transcript": {...}}
    """
    idempotency_key: str
    trigger: str
    bot_id: str
    bot_metadata: Optional[dict] = None
    data: dict
    timestamp: Optional[datetime] = None


class BotStateChangeData(BaseModel):
    """Data for bot.state_change webhook."""
    new_state: str  # "idle", "joining", "in_meeting", "leaving", "error", "ended"
    old_state: Optional[str] = None
    error_message: Optional[str] = None


class TranscriptUpdateData(BaseModel):
    """
    Data for transcript.update webhook from Attendee API.
    
    Actual format from Attendee:
    {
        "speaker_name": "John Doe",
        "speaker_uuid": "spaces/abc/devices/123",
        "speaker_user_uuid": null,
        "speaker_is_host": true,
        "timestamp_ms": 1760890193721,
        "duration_ms": 1219,
        "transcription": {"transcript": "Hello everyone"}
    }
    """
    speaker_name: str
    speaker_uuid: str
    speaker_user_uuid: Optional[str] = None
    speaker_is_host: bool
    timestamp_ms: int
    duration_ms: int
    transcription: dict  # Contains {"transcript": "text here"}
    
    @property
    def speaker(self) -> str:
        """Get speaker name (for backward compatibility)."""
        return self.speaker_name
    
    @property
    def text(self) -> str:
        """Extract transcript text."""
        return self.transcription.get("transcript", "")
    
    @property
    def is_final(self) -> bool:
        """Assume all transcripts from Attendee are final."""
        return True


class CreateBotRequest(BaseModel):
    """Request body for Attendee API POST /bots."""
    meeting_url: str
    bot_name: str = "Prism Voice Agent"
    transcription_settings: dict
    websocket_settings: dict
    webhook_url: Optional[str] = None


class CreateBotResponse(BaseModel):
    """Response from Attendee API POST /bots."""
    bot_id: str
    status: str
    meeting_url: str
    websocket_url: Optional[str] = None  # For audio streaming


class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: str
    message: str
    status_code: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

