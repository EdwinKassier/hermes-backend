"""
Prism Domain Schemas - Request/response validation using Pydantic

Following patterns from hermes/schemas.py for consistency.
"""
from pydantic import BaseModel, Field, field_validator, validator
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
        # Check it's a meet.google.com URL
        if not v.startswith('https://meet.google.com/'):
            raise ValueError(
                'URL must be a Google Meet link (https://meet.google.com/...)'
            )
        
        # Extract meeting code (everything after last /)
        meeting_code = v.split('/')[-1]
        
        # Validate meeting code: alphanumeric and hyphens only
        # Google Meet codes are typically 3 segments separated by hyphens
        if not re.match(r'^[a-z0-9-]+$', meeting_code):
            raise ValueError(
                'Meeting code contains invalid characters. '
                'Expected format: https://meet.google.com/xxx-xxxx-xxx'
            )
        
        # Minimum reasonable length check
        if len(meeting_code) < 7:
            raise ValueError('Meeting code too short')
        
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
    
    class Config:
        # Allow extra fields for future compatibility
        extra = "allow"
        
    @validator('bot_id')
    def validate_bot_id(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError('bot_id must be a non-empty string')
        if len(v) < 3:
            raise ValueError('bot_id must be at least 3 characters long')
        return v
    
    @validator('data')
    def validate_data(cls, v):
        if not isinstance(v, dict):
            raise ValueError('data must be a dictionary')
        return v
    
    @validator('trigger')
    def validate_trigger(cls, v):
        valid_triggers = ['bot.state_change', 'transcript.update', 'bot.error']
        if v not in valid_triggers:
            raise ValueError(f'trigger must be one of: {valid_triggers}')
        return v


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
    
    @validator('speaker_name')
    def validate_speaker_name(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError('speaker_name must be a non-empty string')
        return v
    
    @validator('timestamp_ms')
    def validate_timestamp_ms(cls, v):
        if not isinstance(v, int) or v <= 0:
            raise ValueError('timestamp_ms must be a positive integer')
        return v
    
    @validator('duration_ms')
    def validate_duration_ms(cls, v):
        if not isinstance(v, int) or v < 0:
            raise ValueError('duration_ms must be a non-negative integer')
        return v
    
    @validator('transcription')
    def validate_transcription(cls, v):
        if not isinstance(v, dict):
            raise ValueError('transcription must be a dictionary')
        if 'transcript' not in v:
            raise ValueError('transcription must contain a "transcript" field')
        return v
    
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

