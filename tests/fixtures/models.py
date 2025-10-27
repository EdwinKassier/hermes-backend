"""
Model fixtures for testing.
Provides reusable test data for complex domain models.
"""

from datetime import datetime

import pytest

from app.hermes.models import (
    GeminiResponse,
    ProcessRequestResult,
    ResponseMode,
    UserIdentity,
)
from app.prism.constants import BotState, SessionStatus
from app.prism.models import AudioChunkOutgoing, PrismSession, TranscriptEntry


@pytest.fixture
def sample_user_identity():
    """Reusable user identity fixture"""
    return UserIdentity(
        user_id="test_user_123",
        ip_address="192.168.1.100",
        user_agent="Mozilla/5.0 (Test Agent)",
        accept_language="en-US",
    )


@pytest.fixture
def sample_gemini_response():
    """Sample Gemini AI response"""
    return GeminiResponse(
        content="This is a test AI response with relevant information.",
        user_id="test_user_123",
        prompt="What is the test query?",
        model_used="gemini-2.5-flash",
        metadata={"tokens": 150},
    )


@pytest.fixture
def sample_process_result_text():
    """Sample process request result in text mode"""
    return ProcessRequestResult(
        message="Test AI response",
        response_mode=ResponseMode.TEXT,
        user_id="test_user_123",
        metadata={"model": "gemini-2.5-flash"},
    )


@pytest.fixture
def sample_process_result_tts():
    """Sample process request result in TTS mode"""
    return ProcessRequestResult(
        message="Test AI response",
        response_mode=ResponseMode.TTS,
        audio_url="https://storage.googleapis.com/test-bucket/audio.wav",
        tts_provider="elevenlabs",
        user_id="test_user_123",
        metadata={"model": "gemini-2.5-flash", "duration": 3.5},
    )


@pytest.fixture
def sample_prism_session():
    """Reusable Prism session with some history"""
    session = PrismSession(
        session_id="sess_test_123",
        user_id="user_test_456",
        meeting_url="https://meet.google.com/abc-defg-hij",
        bot_id="bot_test_789",
        status=SessionStatus.ACTIVE,
        bot_state=BotState.IN_MEETING,
    )

    # Add some transcript history
    session.add_transcript("Alice", "Hello everyone", is_final=True)
    session.add_transcript("Bob", "Hi Alice!", is_final=True)
    session.add_transcript("Alice", "How is everyone doing?", is_final=True)

    return session


@pytest.fixture
def sample_prism_session_new():
    """New Prism session without history"""
    return PrismSession(
        session_id="sess_new_123",
        user_id="user_new_456",
        meeting_url="https://meet.google.com/xyz-uvw-rst",
        status=SessionStatus.CREATED,
        bot_state=BotState.IDLE,
    )


@pytest.fixture
def sample_transcripts():
    """Collection of realistic transcripts for testing"""
    return [
        TranscriptEntry(
            speaker="Alice",
            text="Hello, can everyone hear me?",
            timestamp=datetime.utcnow(),
            is_final=True,
        ),
        TranscriptEntry(
            speaker="Bob",
            text="Yes, we can hear you clearly",
            timestamp=datetime.utcnow(),
            is_final=True,
        ),
        TranscriptEntry(
            speaker="Alice",
            text="Great! Let's start the meeting",
            timestamp=datetime.utcnow(),
            is_final=True,
        ),
        TranscriptEntry(
            speaker="Charlie",
            text="Can you share your screen?",
            timestamp=datetime.utcnow(),
            is_final=True,
        ),
    ]


@pytest.fixture
def sample_audio_chunk():
    """Sample audio chunk for testing"""
    return AudioChunkOutgoing(
        data=b"test_audio_data_chunk", timestamp=datetime.utcnow(), sequence=0
    )


@pytest.fixture
def sample_audio_chunks():
    """Collection of audio chunks for testing"""
    return [
        AudioChunkOutgoing(
            data=b"chunk_1_data", timestamp=datetime.utcnow(), sequence=0
        ),
        AudioChunkOutgoing(
            data=b"chunk_2_data", timestamp=datetime.utcnow(), sequence=1
        ),
        AudioChunkOutgoing(
            data=b"chunk_3_data", timestamp=datetime.utcnow(), sequence=2
        ),
    ]
