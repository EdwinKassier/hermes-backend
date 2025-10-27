"""
Custom assertion helpers for tests.
Provides domain-specific assertion utilities.
"""
from typing import List, Dict, Any
from app.prism.constants import SessionStatus, BotState


def assert_similarity_score_valid(score: float):
    """Assert that similarity score is in valid range [0, 1]"""
    assert isinstance(score, (int, float)), f"Score must be numeric, got {type(score)}"
    assert 0.0 <= score <= 1.0, f"Score must be in [0, 1], got {score}"


def assert_embedding_vector_valid(embedding: List[float], expected_dim: int = 1536):
    """Assert that embedding vector is valid"""
    assert isinstance(embedding, list), "Embedding must be a list"
    assert len(embedding) == expected_dim, \
        f"Expected {expected_dim} dimensions, got {len(embedding)}"
    assert all(isinstance(x, (int, float)) for x in embedding), \
        "All embedding values must be numeric"


def assert_session_state_valid(session):
    """Assert that session is in valid state"""
    assert session.session_id is not None, "Session must have session_id"
    assert session.user_id is not None, "Session must have user_id"
    assert session.status in SessionStatus, "Session must have valid status"
    assert session.bot_state in BotState, "Session must have valid bot_state"
    
    # Validate state consistency
    if session.status == SessionStatus.ACTIVE:
        assert session.bot_state == BotState.IN_MEETING, \
            "Active session must have bot in meeting"
    
    if session.bot_id is not None:
        assert session.status != SessionStatus.CREATED, \
            "Session with bot_id cannot be in CREATED state"


def assert_response_time_acceptable(duration: float, max_duration: float):
    """Assert that operation completed within acceptable time"""
    assert duration <= max_duration, \
        f"Operation took {duration:.2f}s, expected < {max_duration}s"


def assert_transcript_valid(transcript):
    """Assert that transcript entry is valid"""
    assert transcript.speaker is not None, "Transcript must have speaker"
    assert transcript.text is not None, "Transcript must have text"
    assert transcript.timestamp is not None, "Transcript must have timestamp"
    assert isinstance(transcript.is_final, bool), "is_final must be boolean"


def assert_audio_chunk_valid(chunk):
    """Assert that audio chunk is valid"""
    assert chunk.data is not None, "Audio chunk must have data"
    assert isinstance(chunk.data, bytes), "Audio data must be bytes"
    assert chunk.timestamp is not None, "Audio chunk must have timestamp"
    assert isinstance(chunk.sequence, int), "Sequence must be integer"
    assert chunk.sequence >= 0, "Sequence must be non-negative"


def assert_json_serializable(obj: Any):
    """Assert that object is JSON serializable"""
    import json
    try:
        json.dumps(obj)
    except (TypeError, ValueError) as e:
        raise AssertionError(f"Object is not JSON serializable: {e}")

