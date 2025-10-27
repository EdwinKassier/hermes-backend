"""
Tests for Prism service layer.
Tests session management, bot orchestration, and AI decision-making.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime
from app.prism.services import PrismService
from app.prism.models import PrismSession, AudioChunkOutgoing
from app.prism.constants import BotState, SessionStatus
from app.prism.exceptions import SessionNotFoundError, BotCreationError


@pytest.fixture
def prism_service():
    """Create PrismService with mocked dependencies"""
    with patch('app.prism.services.RedisSessionStore') as MockRedis, \
         patch('app.prism.services.AttendeeClient') as MockAttendee, \
         patch('app.prism.services.AudioProcessor') as MockAudio, \
         patch('app.prism.services.get_gemini_service') as MockGemini, \
         patch('app.prism.services.get_tts_service') as MockTTS:
        
        # Setup mock session store
        mock_store = Mock()
        mock_store.list_all_session_ids.return_value = []
        MockRedis.return_value = mock_store
        
        service = PrismService()
        service.session_store = mock_store
        service._gemini_service = Mock()
        service._tts_service = Mock()
        
        yield service


@pytest.mark.unit
class TestSessionManagement:
    """Test session lifecycle management"""
    
    def test_create_session(self, prism_service):
        """Test creating a new session"""
        session = prism_service.create_session(
            meeting_url="https://meet.google.com/abc-defg-hij",
            user_identifier="test_user"
        )
        
        assert session.meeting_url == "https://meet.google.com/abc-defg-hij"
        assert session.status == SessionStatus.CREATED
        assert session.user_id == "test_user"
        assert session.session_id is not None
        
        # Verify session was saved
        prism_service.session_store.save_session.assert_called_once()
    
    def test_create_session_generates_user_id(self, prism_service):
        """Test that session generates user ID if not provided"""
        session = prism_service.create_session(
            meeting_url="https://meet.google.com/abc-defg-hij"
        )
        
        assert session.user_id is not None
        assert len(session.user_id) > 0
    
    def test_get_session_success(self, prism_service):
        """Test getting existing session"""
        mock_session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx"
        )
        prism_service.session_store.get_session.return_value = mock_session
        
        session = prism_service.get_session("sess_123")
        
        assert session.session_id == "sess_123"
        prism_service.session_store.get_session.assert_called_once_with("sess_123")
    
    def test_get_session_not_found(self, prism_service):
        """Test getting non-existent session raises error"""
        prism_service.session_store.get_session.return_value = None
        
        with pytest.raises(SessionNotFoundError):
            prism_service.get_session("invalid_session_id")
    
    def test_get_session_by_bot_id(self, prism_service):
        """Test getting session by bot ID"""
        mock_session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx",
            bot_id="bot_789"
        )
        prism_service.session_store.get_session_by_bot_id.return_value = mock_session
        
        session = prism_service.get_session_by_bot_id("bot_789")
        
        assert session.bot_id == "bot_789"
    
    def test_close_session(self, prism_service):
        """Test closing a session"""
        mock_session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx",
            bot_id="bot_789"
        )
        prism_service.session_store.get_session.return_value = mock_session
        
        prism_service.close_session("sess_123")
        
        # Verify bot deletion was called
        prism_service.attendee_client.delete_bot.assert_called_once_with("bot_789")
        # Verify session was deleted from store
        prism_service.session_store.delete_session.assert_called_once_with("sess_123")
    
    def test_close_session_idempotent(self, prism_service):
        """Test that closing non-existent session is safe"""
        prism_service.session_store.get_session.side_effect = SessionNotFoundError("sess_123")
        
        # Should not raise error
        prism_service.close_session("sess_123")


@pytest.mark.unit
class TestBotManagement:
    """Test Attendee bot lifecycle"""
    
    def test_create_bot_success(self, prism_service):
        """Test successful bot creation"""
        mock_session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx"
        )
        prism_service.session_store.get_session.return_value = mock_session
        
        # Mock bot creation response
        mock_bot_response = Mock()
        mock_bot_response.bot_id = "bot_789"
        prism_service.attendee_client.create_bot.return_value = mock_bot_response
        
        bot_id = prism_service.create_bot(
            session_id="sess_123",
            webhook_base_url="https://example.com",
            websocket_base_url="wss://example.com"
        )
        
        assert bot_id == "bot_789"
        assert mock_session.bot_id == "bot_789"
        assert mock_session.status == SessionStatus.BOT_JOINING
        
        # Verify URLs were constructed correctly
        call_args = prism_service.attendee_client.create_bot.call_args
        assert "https://example.com/api/v1/prism/webhook" in str(call_args)
    
    def test_create_bot_failure(self, prism_service):
        """Test bot creation failure"""
        mock_session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx"
        )
        prism_service.session_store.get_session.return_value = mock_session
        
        # Mock bot creation failure
        prism_service.attendee_client.create_bot.side_effect = Exception("API Error")
        
        with pytest.raises(BotCreationError):
            prism_service.create_bot(
                session_id="sess_123",
                webhook_base_url="https://example.com",
                websocket_base_url="wss://example.com"
            )
        
        # Session should be marked as error
        assert mock_session.status == SessionStatus.ERROR
    
    def test_handle_bot_state_change(self, prism_service):
        """Test handling bot state changes"""
        mock_session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx",
            bot_id="bot_789"
        )
        prism_service.session_store.get_session_by_bot_id.return_value = mock_session
        
        prism_service.handle_bot_state_change(
            bot_id="bot_789",
            state="joined_recording"
        )
        
        assert mock_session.bot_state == BotState.IN_MEETING
        # Verify session was saved
        prism_service.session_store.save_session.assert_called()
    
    def test_map_attendee_state_to_bot_state(self, prism_service):
        """Test state mapping from Attendee API"""
        assert prism_service._map_attendee_state_to_bot_state("ready") == BotState.IDLE
        assert prism_service._map_attendee_state_to_bot_state("joining") == BotState.JOINING
        assert prism_service._map_attendee_state_to_bot_state("joined_recording") == BotState.IN_MEETING
        assert prism_service._map_attendee_state_to_bot_state("error") == BotState.ERROR


@pytest.mark.unit
class TestTranscriptProcessing:
    """Test transcript processing and AI decision-making"""
    
    def test_handle_transcript_final(self, prism_service):
        """Test handling final transcript"""
        mock_session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx"
        )
        prism_service.session_store.get_session.return_value = mock_session
        
        # Mock AI decision to not respond
        with patch.object(prism_service, '_should_respond', return_value=False):
            prism_service.handle_transcript(
                session_id="sess_123",
                speaker="John",
                text="Hello everyone",
                is_final=True
            )
        
        # Transcript should be added
        assert len(mock_session.transcript_history) == 1
        assert mock_session.transcript_history[0].text == "Hello everyone"
    
    def test_handle_transcript_skip_interim(self, prism_service):
        """Test that interim transcripts are skipped"""
        mock_session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx"
        )
        prism_service.session_store.get_session.return_value = mock_session
        
        prism_service.handle_transcript(
            session_id="sess_123",
            speaker="John",
            text="Hello...",
            is_final=False
        )
        
        # Interim transcript should not be added
        assert len(mock_session.transcript_history) == 0
    
    def test_handle_transcript_skip_bot_messages(self, prism_service):
        """Test that bot's own messages are skipped"""
        mock_session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx"
        )
        prism_service.session_store.get_session.return_value = mock_session
        
        prism_service.handle_transcript(
            session_id="sess_123",
            speaker="Prism",
            text="I can help with that",
            is_final=True
        )
        
        # Bot's message should not be added
        assert len(mock_session.transcript_history) == 0
    
    def test_handle_transcript_idempotency(self, prism_service):
        """Test webhook idempotency prevents duplicate processing"""
        mock_session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx"
        )
        prism_service.session_store.get_session.return_value = mock_session
        
        # Process same webhook twice with same idempotency key
        with patch.object(prism_service, '_should_respond', return_value=False):
            prism_service.handle_transcript(
                session_id="sess_123",
                speaker="John",
                text="Hello",
                is_final=True,
                idempotency_key="key_123"
            )
            
            prism_service.handle_transcript(
                session_id="sess_123",
                speaker="John",
                text="Hello",
                is_final=True,
                idempotency_key="key_123"
            )
        
        # Should only process once
        assert len(mock_session.transcript_history) == 1
    
    def test_should_respond_decision_yes(self, prism_service):
        """Test AI decision-making for responses - YES"""
        mock_session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx"
        )
        mock_session.add_transcript("User", "Hey Prism, can you help?")
        
        # Mock Gemini response
        prism_service.gemini_service.generate_gemini_response.return_value = "YES"
        
        should_respond = prism_service._should_respond(mock_session)
        assert should_respond is True
    
    def test_should_respond_decision_no(self, prism_service):
        """Test AI decision-making for responses - NO"""
        mock_session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx"
        )
        mock_session.add_transcript("User", "um, okay")
        
        # Mock Gemini response
        prism_service.gemini_service.generate_gemini_response.return_value = "NO"
        
        should_respond = prism_service._should_respond(mock_session)
        assert should_respond is False
    
    def test_should_skip_obvious_non_response(self, prism_service):
        """Test fast heuristic filter for obvious non-responses"""
        # Filler words - should skip
        assert prism_service._should_skip_obvious_non_response("um") is True
        assert prism_service._should_skip_obvious_non_response("uh") is True
        assert prism_service._should_skip_obvious_non_response("hmm") is True
        
        # Too short
        assert prism_service._should_skip_obvious_non_response("a") is True
        
        # Question - should NOT skip
        assert prism_service._should_skip_obvious_non_response("What time is it?") is False
        
        # Bot mention - should NOT skip
        assert prism_service._should_skip_obvious_non_response("Hey Prism") is False
        assert prism_service._should_skip_obvious_non_response("Can the bot help?") is False
    
    def test_prevents_concurrent_response_generation(self, prism_service):
        """Test that concurrent responses are prevented by lock"""
        mock_session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx"
        )
        mock_session.add_transcript("User", "Help me")
        mock_session.is_generating_response = True  # Lock is set
        
        prism_service.session_store.get_session.return_value = mock_session
        
        with patch.object(prism_service, '_should_respond', return_value=True):
            prism_service.handle_transcript(
                session_id="sess_123",
                speaker="User",
                text="Another question",
                is_final=True
            )
        
        # Should not call response generation due to lock
        prism_service.gemini_service.generate_gemini_response.assert_not_called()


@pytest.mark.unit
class TestWebSocketManagement:
    """Test WebSocket connection management"""
    
    def test_mark_user_connected(self, prism_service):
        """Test marking user WebSocket as connected"""
        mock_session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx"
        )
        prism_service.session_store.get_session.return_value = mock_session
        
        prism_service.mark_user_connected("sess_123")
        
        assert mock_session.user_ws_connected is True
        prism_service.session_store.save_session.assert_called()
    
    def test_mark_user_disconnected(self, prism_service):
        """Test marking user WebSocket as disconnected"""
        mock_session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx"
        )
        prism_service.session_store.get_session.return_value = mock_session
        
        with patch.object(prism_service, 'close_session'):
            prism_service.mark_user_disconnected("sess_123")
        
        assert mock_session.user_ws_connected is False
    
    def test_mark_bot_connected(self, prism_service):
        """Test marking bot WebSocket as connected"""
        mock_session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx"
        )
        prism_service.session_store.get_session.return_value = mock_session
        
        prism_service.mark_bot_connected("sess_123")
        
        assert mock_session.bot_ws_connected is True
    
    def test_get_pending_audio(self, prism_service):
        """Test retrieving pending audio chunks"""
        mock_session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx"
        )
        
        # Add some audio chunks
        for i in range(3):
            mock_session.audio_queue.append(
                AudioChunkOutgoing(
                    data=b"chunk_data",
                    timestamp=datetime.utcnow(),
                    sequence=i
                )
            )
        
        prism_service.session_store.get_session.return_value = mock_session
        
        pending = prism_service.get_pending_audio("sess_123")
        
        assert len(pending) == 3
        # Queue should be cleared
        assert len(mock_session.audio_queue) == 0


@pytest.mark.unit
class TestResourceMonitoring:
    """Test resource monitoring methods"""
    
    def test_active_sessions_count(self, prism_service):
        """Test getting active sessions count"""
        prism_service.session_store.list_all_session_ids.return_value = [
            "sess_1", "sess_2", "sess_3"
        ]
        
        count = prism_service.active_sessions_count
        assert count == 3
    
    def test_get_session_metrics(self, prism_service):
        """Test getting session metrics"""
        mock_session = PrismSession(
            session_id="sess_123",
            user_id="user_456",
            meeting_url="https://meet.google.com/xxx"
        )
        mock_session.add_transcript("Speaker", "Hello")
        
        prism_service.session_store.get_session.return_value = mock_session
        
        metrics = prism_service.get_session_metrics("sess_123")
        
        assert metrics['session_id'] == "sess_123"
        assert metrics['transcript_count'] == 1
        assert 'status' in metrics
        assert 'bot_state' in metrics

