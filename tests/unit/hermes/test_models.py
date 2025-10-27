"""
Tests for Hermes domain models.
Coverage: UserIdentity, ConversationContext, GeminiResponse, ProcessRequestResult
"""
import pytest
from datetime import datetime
import time
from app.hermes.models import (
    UserIdentity, ConversationContext, ConversationMessage,
    GeminiResponse, ProcessRequestResult, ResponseMode, TTSResponse
)


@pytest.mark.unit
class TestUserIdentity:
    """Test UserIdentity model validation and behavior"""
    
    def test_user_identity_creation(self):
        """Test creating a valid user identity"""
        identity = UserIdentity(
            user_id="test_user_123",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            accept_language="en-US"
        )
        assert identity.user_id == "test_user_123"
        assert identity.ip_address == "192.168.1.1"
        assert isinstance(identity.timestamp, datetime)
    
    def test_user_identity_defaults(self):
        """Test default values for optional fields"""
        identity = UserIdentity(
            user_id="test_user",
            ip_address="127.0.0.1",
            user_agent="test"
        )
        assert identity.accept_language == "en-US"
        assert identity.timestamp is not None
    
    def test_user_identity_pydantic_validation(self):
        """Test Pydantic validation"""
        # Missing required field
        with pytest.raises(Exception):  # Pydantic ValidationError
            UserIdentity(
                user_id="test",
                ip_address="127.0.0.1"
                # Missing user_agent
            )


@pytest.mark.unit
class TestConversationMessage:
    """Test conversation message model"""
    
    def test_message_creation(self):
        """Test creating a conversation message"""
        msg = ConversationMessage(
            role="user",
            content="Hello world"
        )
        assert msg.role == "user"
        assert msg.content == "Hello world"
        assert msg.timestamp is not None
    
    def test_message_role_validation(self):
        """Test that invalid roles are rejected"""
        with pytest.raises(ValueError, match="Role must be one of"):
            ConversationMessage(
                role="invalid_role",
                content="test"
            )
    
    def test_message_allowed_roles(self):
        """Test all allowed roles"""
        for role in ['user', 'assistant', 'system']:
            msg = ConversationMessage(role=role, content="Test")
            assert msg.role == role


@pytest.mark.unit
class TestConversationContext:
    """Test conversation context management"""
    
    def test_conversation_context_creation(self):
        """Test creating conversation context"""
        context = ConversationContext(user_id="user_123")
        assert context.user_id == "user_123"
        assert len(context.messages) == 0
        assert context.created_at is not None
    
    def test_add_message(self):
        """Test adding messages to conversation"""
        context = ConversationContext(user_id="user_123")
        context.add_message("user", "Hello")
        
        assert len(context.messages) == 1
        assert context.messages[0].role == "user"
        assert context.messages[0].content == "Hello"
    
    def test_add_multiple_messages(self):
        """Test adding multiple messages"""
        context = ConversationContext(user_id="user_123")
        context.add_message("user", "Hello")
        context.add_message("assistant", "Hi there!")
        context.add_message("user", "How are you?")
        
        assert len(context.messages) == 3
        assert context.messages[1].role == "assistant"
    
    def test_get_recent_messages(self):
        """Test retrieving limited recent messages"""
        context = ConversationContext(user_id="user_123")
        for i in range(15):
            context.add_message("user", f"Message {i}")
        
        recent = context.get_recent_messages(limit=5)
        assert len(recent) == 5
        assert recent[-1].content == "Message 14"
        assert recent[0].content == "Message 10"
    
    def test_get_recent_messages_less_than_limit(self):
        """Test get_recent_messages when fewer messages exist"""
        context = ConversationContext(user_id="user_123")
        context.add_message("user", "Message 1")
        context.add_message("user", "Message 2")
        
        recent = context.get_recent_messages(limit=10)
        assert len(recent) == 2
    
    def test_updated_timestamp(self):
        """Test that updated_at changes when adding messages"""
        context = ConversationContext(user_id="user_123")
        initial_time = context.updated_at
        
        time.sleep(0.01)  # Small delay
        context.add_message("user", "New message")
        
        assert context.updated_at > initial_time
    
    def test_add_message_with_metadata(self):
        """Test adding message with metadata"""
        context = ConversationContext(user_id="user_123")
        context.add_message("user", "Hello", metadata={"source": "test"})
        
        assert context.messages[0].metadata == {"source": "test"}


@pytest.mark.unit
class TestGeminiResponse:
    """Test Gemini response model"""
    
    def test_gemini_response_creation(self):
        """Test creating Gemini response"""
        response = GeminiResponse(
            content="AI generated response",
            user_id="user_123",
            prompt="Test prompt",
            model_used="gemini-2.5-flash"
        )
        assert response.content == "AI generated response"
        assert response.model_used == "gemini-2.5-flash"
    
    def test_gemini_response_optional_fields(self):
        """Test optional fields"""
        response = GeminiResponse(
            content="Response",
            user_id="user_123",
            prompt="Prompt"
        )
        assert response.model_used is None
        assert response.metadata == {}


@pytest.mark.unit
class TestProcessRequestResult:
    """Test process request result model"""
    
    def test_result_creation_text_mode(self):
        """Test creating result in text mode"""
        result = ProcessRequestResult(
            message="Test response",
            response_mode=ResponseMode.TEXT,
            user_id="user_123"
        )
        assert result.message == "Test response"
        assert result.response_mode == ResponseMode.TEXT
        assert result.audio_url is None
        assert result.tts_provider is None
    
    def test_result_creation_tts_mode(self):
        """Test creating result in TTS mode"""
        result = ProcessRequestResult(
            message="Test response",
            response_mode=ResponseMode.TTS,
            audio_url="https://storage.example.com/audio.wav",
            tts_provider="elevenlabs",
            user_id="user_123"
        )
        assert result.audio_url is not None
        assert result.tts_provider == "elevenlabs"
        assert result.response_mode == ResponseMode.TTS
    
    def test_result_with_metadata(self):
        """Test result with metadata"""
        result = ProcessRequestResult(
            message="Response",
            response_mode=ResponseMode.TEXT,
            user_id="user_123",
            metadata={"model": "gemini", "tokens": 100}
        )
        assert result.metadata["model"] == "gemini"
        assert result.metadata["tokens"] == 100
    
    def test_result_timestamp_auto_generated(self):
        """Test that timestamp is automatically generated"""
        result = ProcessRequestResult(
            message="Response",
            response_mode=ResponseMode.TEXT,
            user_id="user_123"
        )
        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)


@pytest.mark.unit
class TestResponseMode:
    """Test ResponseMode enum"""
    
    def test_response_mode_values(self):
        """Test response mode enum values"""
        assert ResponseMode.TEXT.value == "text"
        assert ResponseMode.TTS.value == "tts"
    
    def test_response_mode_from_string(self):
        """Test creating ResponseMode from string"""
        mode = ResponseMode("text")
        assert mode == ResponseMode.TEXT
        
        mode = ResponseMode("tts")
        assert mode == ResponseMode.TTS


@pytest.mark.unit
class TestTTSResponse:
    """Test TTS response model"""
    
    def test_tts_response_creation(self):
        """Test creating TTS response"""
        response = TTSResponse(
            text="Hello world",
            audio_url="https://storage.example.com/audio.wav",
            duration_seconds=2.5,
            format="wav"
        )
        assert response.text == "Hello world"
        assert response.duration_seconds == 2.5
        assert response.format == "wav"
    
    def test_tts_response_defaults(self):
        """Test default values"""
        response = TTSResponse(
            text="Test",
            audio_url="https://example.com/audio"
        )
        assert response.format == "wav"
        assert response.duration_seconds is None

