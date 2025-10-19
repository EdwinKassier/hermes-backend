"""
Prism Domain Models - Session state and data structures

Following patterns from hermes/models.py for consistency.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from .constants import BotState, SessionStatus


@dataclass
class TranscriptEntry:
    """Represents a single transcript message from the meeting."""
    speaker: str
    text: str
    timestamp: datetime
    is_final: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "speaker": self.speaker,
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
            "is_final": self.is_final
        }


@dataclass
class AudioChunkIncoming:
    """Incoming audio chunk from Attendee bot (if needed for future features)."""
    data: bytes  # Base64-decoded PCM audio
    timestamp: datetime
    sequence: int = 0


@dataclass
class AudioChunkOutgoing:
    """Outgoing audio chunk to send to Attendee bot."""
    data: bytes  # Raw PCM audio (will be base64-encoded before sending)
    timestamp: datetime
    sequence: int = 0
    
    def to_base64(self) -> str:
        """Convert audio data to base64 for WebSocket transmission."""
        import base64
        return base64.b64encode(self.data).decode('utf-8')


@dataclass
class PrismSession:
    """
    Represents an active Prism session with Attendee bot.
    
    Lifecycle:
    1. User connects via WebSocket
    2. Session created (status=CREATED)
    3. Bot created via Attendee API (status=BOT_CREATING)
    4. Bot joins meeting (status=BOT_JOINING)
    5. Bot audio WebSocket connects (status=ACTIVE)
    6. Session runs until user disconnects or bot leaves
    7. Session closed (status=CLOSED)
    """
    session_id: str
    user_id: str
    meeting_url: str
    bot_id: Optional[str] = None
    bot_state: BotState = BotState.IDLE
    status: SessionStatus = SessionStatus.CREATED
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Connection tracking
    user_ws_connected: bool = False
    bot_ws_connected: bool = False
    
    # Conversation state
    transcript_history: List[TranscriptEntry] = field(default_factory=list)
    conversation_context: List[Dict[str, str]] = field(default_factory=list)  # For Gemini
    
    # Audio queue (for pending audio to send to bot)
    audio_queue: List[AudioChunkOutgoing] = field(default_factory=list)
    
    # Bot introduction tracking
    has_introduced: bool = False
    
    # Response generation lock (prevent concurrent responses)
    is_generating_response: bool = False
    
    # Metadata
    error_message: Optional[str] = None
    last_activity: datetime = field(default_factory=datetime.utcnow)
    
    def add_transcript(self, speaker: str, text: str, is_final: bool = True):
        """Add a new transcript entry and maintain history limit."""
        from .constants import MAX_CONVERSATION_HISTORY
        
        entry = TranscriptEntry(
            speaker=speaker,
            text=text,
            timestamp=datetime.utcnow(),
            is_final=is_final
        )
        self.transcript_history.append(entry)
        
        # Keep only last N entries
        if len(self.transcript_history) > MAX_CONVERSATION_HISTORY:
            self.transcript_history = self.transcript_history[-MAX_CONVERSATION_HISTORY:]
        
        self.last_activity = datetime.utcnow()
    
    def add_to_conversation_context(self, role: str, content: str):
        """Add message to Gemini conversation context."""
        from .constants import MAX_CONVERSATION_HISTORY
        
        self.conversation_context.append({
            "role": role,
            "content": content
        })
        
        # Keep only last N messages
        if len(self.conversation_context) > MAX_CONVERSATION_HISTORY:
            self.conversation_context = self.conversation_context[-MAX_CONVERSATION_HISTORY:]
    
    def update_status(self, status: SessionStatus, error: Optional[str] = None):
        """Update session status and timestamp."""
        self.status = status
        self.updated_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        if error:
            self.error_message = error
    
    def update_bot_state(self, state: BotState):
        """Update bot state from Attendee webhook."""
        self.bot_state = state
        self.updated_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        
        # Update session status based on bot state
        if state == BotState.JOINING:
            self.status = SessionStatus.BOT_JOINING
        elif state == BotState.IN_MEETING:
            self.status = SessionStatus.ACTIVE
        elif state == BotState.ERROR:
            self.status = SessionStatus.ERROR
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "meeting_url": self.meeting_url,
            "bot_id": self.bot_id,
            "bot_state": self.bot_state.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "user_ws_connected": self.user_ws_connected,
            "bot_ws_connected": self.bot_ws_connected,
            "transcript_count": len(self.transcript_history),
            "error_message": self.error_message
        }

