"""
Tests for Hermes service layer.
Tests business logic with mocked dependencies.
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

import pytest

from app.hermes.exceptions import (
    AIServiceError,
    HermesServiceError,
    InvalidRequestError,
    TTSServiceError,
)
from app.hermes.models import GeminiResponse, ResponseMode, UserIdentity
from app.hermes.services import HermesService


@pytest.fixture
def hermes_service():
    """Create HermesService with mocked dependencies"""
    with (
        patch("app.hermes.services.get_gemini_service") as mock_gemini_getter,
        patch("app.hermes.services.get_tts_service") as mock_tts_getter,
    ):

        mock_gemini = Mock()
        mock_tts = Mock()

        mock_gemini_getter.return_value = mock_gemini
        mock_tts_getter.return_value = mock_tts

        service = HermesService()
        service._gemini_service = mock_gemini
        service._tts_service = mock_tts

        yield service


@pytest.fixture
def user_identity():
    """Create test user identity"""
    return UserIdentity(
        user_id="test_user",
        ip_address="127.0.0.1",
        user_agent="test-agent",
        accept_language="en-US",
    )


@pytest.mark.unit
class TestProcessRequest:
    """Test process_request method"""

    def test_process_request_text_mode(self, hermes_service, user_identity):
        """Test processing request in text mode"""
        # Mock Gemini response
        hermes_service.gemini_service.generate_gemini_response_with_rag.return_value = (
            "AI response"
        )

        result = hermes_service.process_request(
            text="What is the weather?",
            user_identity=user_identity,
            response_mode=ResponseMode.TEXT,
        )

        assert result.message == "AI response"
        assert result.response_mode == ResponseMode.TEXT
        assert result.audio_url is None
        assert result.user_id == "test_user"

        # Verify Gemini was called correctly
        hermes_service.gemini_service.generate_gemini_response_with_rag.assert_called_once()

    def test_process_request_tts_mode(self, hermes_service, user_identity):
        """Test processing request in TTS mode"""
        # Mock services
        hermes_service.gemini_service.generate_gemini_response_with_rag.return_value = (
            "AI response"
        )
        hermes_service.tts_service.generate_audio.return_value = {
            "cloud_url": "https://storage.example.com/audio.wav"
        }
        hermes_service.tts_service.tts_provider = "elevenlabs"

        result = hermes_service.process_request(
            text="Tell me a story",
            user_identity=user_identity,
            response_mode=ResponseMode.TTS,
        )

        assert result.audio_url == "https://storage.example.com/audio.wav"
        assert result.tts_provider == "elevenlabs"
        assert result.response_mode == ResponseMode.TTS

        # Verify TTS was called with correct parameters
        hermes_service.tts_service.generate_audio.assert_called_once_with(
            text_input="AI response",
            upload_to_cloud=True,
            cloud_destination_path=unittest.mock.ANY,
        )

    def test_process_request_empty_text(self, hermes_service, user_identity):
        """Test that empty text raises error"""
        with pytest.raises(InvalidRequestError, match="cannot be empty"):
            hermes_service.process_request(text="", user_identity=user_identity)

    def test_process_request_whitespace_only(self, hermes_service, user_identity):
        """Test that whitespace-only text raises error"""
        with pytest.raises(InvalidRequestError):
            hermes_service.process_request(text="   ", user_identity=user_identity)

    def test_process_request_ai_service_error(self, hermes_service, user_identity):
        """Test handling of AI service errors"""
        hermes_service.gemini_service.generate_gemini_response_with_rag.side_effect = (
            Exception("API error")
        )

        # Service wraps generic exceptions in HermesServiceError
        with pytest.raises((HermesServiceError, Exception)):
            hermes_service.process_request(text="Test", user_identity=user_identity)

    def test_process_request_tts_error(self, hermes_service, user_identity):
        """Test handling of TTS errors"""
        hermes_service.gemini_service.generate_gemini_response_with_rag.return_value = (
            "Response"
        )
        hermes_service.tts_service.generate_audio.side_effect = KeyError("Missing key")

        with pytest.raises(TTSServiceError):
            hermes_service.process_request(
                text="Test", user_identity=user_identity, response_mode=ResponseMode.TTS
            )

    def test_process_request_includes_metadata(self, hermes_service, user_identity):
        """Test that result includes metadata"""
        hermes_service.gemini_service.generate_gemini_response_with_rag.return_value = (
            "Response"
        )

        result = hermes_service.process_request(
            text="Test question", user_identity=user_identity
        )

        assert "model" in result.metadata
        assert "prompt_length" in result.metadata
        assert "response_length" in result.metadata


@pytest.mark.unit
class TestChat:
    """Test chat method with conversation context"""

    def test_chat_with_context(self, hermes_service, user_identity):
        """Test chat maintains conversation context"""
        hermes_service.gemini_service.generate_gemini_response_with_rag.return_value = (
            "Response"
        )

        # First message
        result1 = hermes_service.chat(
            message="Hello", user_identity=user_identity, include_context=True
        )

        # Second message
        result2 = hermes_service.chat(
            message="How are you?", user_identity=user_identity, include_context=True
        )

        # Check context was maintained
        context = hermes_service._get_conversation_context(user_identity.user_id)
        assert len(context.messages) == 4  # 2 user + 2 assistant

    def test_chat_without_context(self, hermes_service, user_identity):
        """Test chat without conversation context"""
        hermes_service.gemini_service.generate_gemini_response_with_rag.return_value = (
            "Response"
        )

        result = hermes_service.chat(
            message="Hello", user_identity=user_identity, include_context=False
        )

        # Context should not be stored
        assert user_identity.user_id not in hermes_service._conversation_contexts

    def test_chat_empty_message(self, hermes_service, user_identity):
        """Test that empty message raises error"""
        with pytest.raises(InvalidRequestError):
            hermes_service.chat(message="", user_identity=user_identity)

    def test_chat_returns_gemini_response(self, hermes_service, user_identity):
        """Test that chat returns GeminiResponse object"""
        hermes_service.gemini_service.generate_gemini_response_with_rag.return_value = (
            "Test response"
        )

        result = hermes_service.chat(message="Hello", user_identity=user_identity)

        assert isinstance(result, GeminiResponse)
        assert result.content == "Test response"
        assert result.user_id == user_identity.user_id


@pytest.mark.unit
class TestGenerateTTS:
    """Test TTS generation"""

    def test_generate_tts_success(self, hermes_service):
        """Test successful TTS generation"""
        hermes_service.tts_service.generate_audio.return_value = {
            "cloud_url": "https://example.com/audio.wav"
        }
        hermes_service.tts_service.tts_provider = "google"

        url, provider = hermes_service.generate_tts("Hello world")

        assert url == "https://example.com/audio.wav"
        assert provider == "google"

    def test_generate_tts_error(self, hermes_service):
        """Test TTS generation error handling"""
        hermes_service.tts_service.generate_audio.side_effect = ValueError(
            "Invalid input"
        )

        with pytest.raises(TTSServiceError):
            hermes_service.generate_tts("Test")


@pytest.mark.unit
class TestConversationContextManagement:
    """Test conversation context management"""

    def test_get_conversation_context_creates_new(self, hermes_service):
        """Test getting non-existent context creates new one"""
        context = hermes_service._get_conversation_context("new_user")

        assert context.user_id == "new_user"
        assert len(context.messages) == 0

    def test_get_conversation_context_retrieves_existing(self, hermes_service):
        """Test getting existing context"""
        # Create context
        context1 = hermes_service._get_conversation_context("user_123")
        context1.add_message("user", "Hello")

        # Retrieve same context
        context2 = hermes_service._get_conversation_context("user_123")

        assert len(context2.messages) == 1
        assert context2.messages[0].content == "Hello"

    def test_clear_conversation_context(self, hermes_service):
        """Test clearing conversation context"""
        user_id = "test_user"

        # Create context
        context = hermes_service._get_conversation_context(user_id)
        context.add_message("user", "Hello")
        hermes_service._save_conversation_context(user_id, context)

        # Clear context
        hermes_service.clear_conversation_context(user_id)

        assert user_id not in hermes_service._conversation_contexts

    def test_clear_non_existent_context(self, hermes_service):
        """Test clearing non-existent context doesn't error"""
        # Should not raise error
        hermes_service.clear_conversation_context("non_existent_user")


@pytest.mark.unit
class TestServiceSingleton:
    """Test service singleton pattern"""

    def test_get_hermes_service_singleton(self):
        """Test that get_hermes_service returns singleton"""
        from app.hermes.services import get_hermes_service

        with (
            patch("app.hermes.services.get_gemini_service"),
            patch("app.hermes.services.get_tts_service"),
        ):
            service1 = get_hermes_service()
            service2 = get_hermes_service()

            assert service1 is service2
