"""
Tests for Prism domain models.
Coverage: PrismSession, TranscriptEntry, AudioChunk
"""

from datetime import datetime

import pytest

from app.prism.constants import BotState, SessionStatus
from app.prism.models import (
    AudioChunkIncoming,
    AudioChunkOutgoing,
    PrismSession,
    TranscriptEntry,
)


@pytest.mark.unit
class TestTranscriptEntry:
    """Test transcript entry model"""

    def test_transcript_creation(self):
        """Test creating transcript entry"""
        entry = TranscriptEntry(
            speaker="John Doe",
            text="Hello everyone",
            timestamp=datetime.utcnow(),
            is_final=True,
        )
        assert entry.speaker == "John Doe"
        assert entry.text == "Hello everyone"
        assert entry.is_final is True

    def test_transcript_defaults(self):
        """Test default values"""
        entry = TranscriptEntry(
            speaker="Jane", text="Test", timestamp=datetime.utcnow()
        )
        assert entry.is_final is True

    def test_transcript_to_dict(self):
        """Test serialization to dict"""
        entry = TranscriptEntry(
            speaker="Jane", text="Test", timestamp=datetime.utcnow()
        )
        data = entry.to_dict()

        assert data["speaker"] == "Jane"
        assert data["text"] == "Test"
        assert "timestamp" in data
        assert data["is_final"] is True


@pytest.mark.unit
class TestAudioChunk:
    """Test audio chunk models"""

    def test_audio_chunk_outgoing(self):
        """Test outgoing audio chunk"""
        chunk = AudioChunkOutgoing(
            data=b"audio_data", timestamp=datetime.utcnow(), sequence=1
        )
        assert chunk.sequence == 1
        assert isinstance(chunk.data, bytes)
        assert len(chunk.data) == 10

    def test_audio_chunk_to_base64(self):
        """Test base64 encoding"""
        chunk = AudioChunkOutgoing(data=b"test", timestamp=datetime.utcnow())
        encoded = chunk.to_base64()
        assert isinstance(encoded, str)

        # Verify it's valid base64
        import base64

        decoded = base64.b64decode(encoded)
        assert decoded == b"test"

    def test_audio_chunk_incoming(self):
        """Test incoming audio chunk"""
        chunk = AudioChunkIncoming(
            data=b"incoming_audio", timestamp=datetime.utcnow(), sequence=5
        )
        assert chunk.sequence == 5
        assert chunk.data == b"incoming_audio"


@pytest.mark.unit
class TestPrismSession:
    """Test Prism session model"""

    def test_session_creation(self):
        """Test creating new session"""
        session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx",
        )
        assert session.status == SessionStatus.CREATED
        assert session.bot_state == BotState.IDLE
        assert len(session.transcript_history) == 0
        assert len(session.audio_queue) == 0

    def test_session_with_bot(self):
        """Test session with bot ID"""
        session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx",
            bot_id="bot_789",
        )
        assert session.bot_id == "bot_789"

    def test_add_transcript(self):
        """Test adding transcripts to session"""
        session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx",
        )

        session.add_transcript("Speaker 1", "Hello", is_final=True)
        session.add_transcript("Speaker 2", "Hi", is_final=True)

        assert len(session.transcript_history) == 2
        assert session.transcript_history[0].speaker == "Speaker 1"
        assert session.transcript_history[1].text == "Hi"

    def test_transcript_history_limit(self):
        """Test that transcript history respects MAX_CONVERSATION_HISTORY"""
        session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx",
        )

        # Add more than MAX_CONVERSATION_HISTORY transcripts (assume 50)
        for i in range(60):
            session.add_transcript(f"Speaker {i}", f"Message {i}")

        # Should keep only last 50
        assert len(session.transcript_history) <= 50
        # Last message should be Message 59
        assert "59" in session.transcript_history[-1].text

    def test_add_to_conversation_context(self):
        """Test adding to conversation context"""
        session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx",
        )

        session.add_to_conversation_context("user", "Hello")
        session.add_to_conversation_context("assistant", "Hi there")

        assert len(session.conversation_context) == 2
        assert session.conversation_context[0]["role"] == "user"
        assert session.conversation_context[1]["content"] == "Hi there"

    def test_update_status(self):
        """Test updating session status"""
        session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx",
        )

        initial_time = session.updated_at
        session.update_status(SessionStatus.ACTIVE)

        assert session.status == SessionStatus.ACTIVE
        assert session.updated_at >= initial_time

    def test_update_status_with_error(self):
        """Test updating status with error message"""
        session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx",
        )

        session.update_status(SessionStatus.ERROR, error="Test error")

        assert session.status == SessionStatus.ERROR
        assert session.error_message == "Test error"

    def test_update_bot_state(self):
        """Test updating bot state"""
        session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx",
        )

        session.update_bot_state(BotState.IN_MEETING)

        assert session.bot_state == BotState.IN_MEETING
        assert session.status == SessionStatus.ACTIVE

    def test_bot_state_updates_session_status(self):
        """Test that bot states correctly update session status"""
        session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx",
        )

        # Test various state transitions
        session.update_bot_state(BotState.JOINING)
        assert session.status == SessionStatus.BOT_JOINING

        session.update_bot_state(BotState.IN_MEETING)
        assert session.status == SessionStatus.ACTIVE

        session.update_bot_state(BotState.ERROR)
        assert session.status == SessionStatus.ERROR

    def test_session_to_dict(self):
        """Test serialization"""
        session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx",
            bot_id="bot_789",
        )
        session.add_transcript("Speaker", "Hello")

        data = session.to_dict()

        assert data["session_id"] == "sess_123"
        assert data["bot_id"] == "bot_789"
        assert data["transcript_count"] == 1
        assert data["status"] == SessionStatus.CREATED.value
        assert data["bot_state"] == BotState.IDLE.value

    def test_connection_tracking(self):
        """Test connection tracking fields"""
        session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx",
        )

        assert session.user_ws_connected is False
        assert session.bot_ws_connected is False

        session.user_ws_connected = True
        session.bot_ws_connected = True

        assert session.user_ws_connected is True
        assert session.bot_ws_connected is True

    def test_audio_queue(self):
        """Test audio queue management"""
        session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx",
        )

        # Add audio chunks
        for i in range(3):
            session.audio_queue.append(
                AudioChunkOutgoing(
                    data=b"data", timestamp=datetime.utcnow(), sequence=i
                )
            )

        assert len(session.audio_queue) == 3

    def test_has_introduced_flag(self):
        """Test bot introduction tracking"""
        session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx",
        )

        assert session.has_introduced is False

        session.has_introduced = True
        assert session.has_introduced is True

    def test_is_generating_response_lock(self):
        """Test response generation lock"""
        session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx",
        )

        assert session.is_generating_response is False

        session.is_generating_response = True
        assert session.is_generating_response is True


@pytest.mark.unit
class TestSessionConstants:
    """Test session constants"""

    def test_session_status_values(self):
        """Test SessionStatus enum values"""
        assert SessionStatus.CREATED.value == "created"
        assert SessionStatus.ACTIVE.value == "active"
        assert SessionStatus.CLOSED.value == "closed"
        assert SessionStatus.ERROR.value == "error"

    def test_bot_state_values(self):
        """Test BotState enum values"""
        assert BotState.IDLE.value == "idle"
        assert BotState.JOINING.value == "joining"
        assert BotState.IN_MEETING.value == "in_meeting"
        assert BotState.LEAVING.value == "leaving"
        assert BotState.ERROR.value == "error"
