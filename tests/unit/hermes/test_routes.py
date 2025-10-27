"""
Tests for Hermes HTTP routes.
Tests request handling, validation, and response formatting.
"""

import json
from unittest.mock import Mock, patch

import pytest
from flask import Flask

from app.hermes.models import GeminiResponse, ProcessRequestResult, ResponseMode
from app.hermes.routes import hermes


@pytest.fixture
def app():
    """Create Flask test app"""
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test-secret-key"
    app.register_blueprint(hermes, url_prefix="/api/v1/hermes")
    return app


@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()


@pytest.mark.unit
class TestProcessRequestRoute:
    """Test /process_request endpoint"""

    @patch("app.hermes.routes.get_hermes_service")
    @patch("app.hermes.routes.IdentityService.get_identity_fingerprint")
    def test_process_request_success(self, mock_identity, mock_service_getter, client):
        """Test successful request processing"""
        # Mock identity
        mock_identity.return_value = {
            "user_id": "test_user",
            "ip_address": "127.0.0.1",
            "user_agent": "test",
            "accept_language": "en-US",
        }

        # Mock service
        mock_service = Mock()
        mock_service_getter.return_value = mock_service

        mock_result = ProcessRequestResult(
            message="AI response",
            response_mode=ResponseMode.TEXT,
            user_id="test_user",
            metadata={},
        )
        mock_service.process_request.return_value = mock_result

        # Make request
        response = client.get(
            "/api/v1/hermes/process_request?request_text=Hello&response_mode=text"
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["message"] == "AI response"
        assert data["response_mode"] == "text"

    @patch("app.hermes.routes.get_hermes_service")
    @patch("app.hermes.routes.IdentityService.get_identity_fingerprint")
    def test_process_request_tts_mode(self, mock_identity, mock_service_getter, client):
        """Test request processing in TTS mode"""
        mock_identity.return_value = {
            "user_id": "test_user",
            "ip_address": "127.0.0.1",
            "user_agent": "test",
            "accept_language": "en-US",
        }

        mock_service = Mock()
        mock_service_getter.return_value = mock_service

        mock_result = ProcessRequestResult(
            message="AI response",
            response_mode=ResponseMode.TTS,
            audio_url="https://storage.example.com/audio.wav",
            tts_provider="elevenlabs",
            user_id="test_user",
            metadata={},
        )
        mock_service.process_request.return_value = mock_result

        response = client.get(
            "/api/v1/hermes/process_request?request_text=Hello&response_mode=tts"
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["wave_url"] == "https://storage.example.com/audio.wav"
        assert data["tts_provider"] == "elevenlabs"

    @patch("app.hermes.routes.IdentityService.get_identity_fingerprint")
    def test_process_request_missing_text(self, mock_identity, client):
        """Test request with missing text parameter"""
        mock_identity.return_value = {
            "user_id": "test_user",
            "ip_address": "127.0.0.1",
            "user_agent": "test",
            "accept_language": "en-US",
        }

        response = client.get("/api/v1/hermes/process_request")

        # Should either handle gracefully or return error
        # Depends on validation implementation
        assert response.status_code in [200, 400]


@pytest.mark.unit
class TestChatRoute:
    """Test /chat endpoint"""

    @patch("app.hermes.routes.get_hermes_service")
    @patch("app.hermes.routes.IdentityService.get_identity_fingerprint")
    def test_chat_success(self, mock_identity, mock_service_getter, client):
        """Test successful chat request"""
        mock_identity.return_value = {
            "user_id": "test_user",
            "ip_address": "127.0.0.1",
            "user_agent": "test",
            "accept_language": "en-US",
        }

        mock_service = Mock()
        mock_service_getter.return_value = mock_service

        mock_response = GeminiResponse(
            content="Chat response",
            user_id="test_user",
            prompt="Hello",
            model_used="gemini-pro",
        )
        mock_service.chat.return_value = mock_response

        response = client.post(
            "/api/v1/hermes/chat", json={"message": "Hello", "include_context": True}
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["message"] == "Chat response"

    def test_chat_missing_body(self, client):
        """Test chat with missing request body"""
        response = client.post("/api/v1/hermes/chat")
        # Flask returns 500 for unsupported media type/missing JSON
        assert response.status_code in [400, 500]

    @patch("app.hermes.routes.get_hermes_service")
    @patch("app.hermes.routes.IdentityService.get_identity_fingerprint")
    def test_chat_without_context(self, mock_identity, mock_service_getter, client):
        """Test chat without context"""
        mock_identity.return_value = {
            "user_id": "test_user",
            "ip_address": "127.0.0.1",
            "user_agent": "test",
            "accept_language": "en-US",
        }

        mock_service = Mock()
        mock_service_getter.return_value = mock_service

        mock_response = GeminiResponse(
            content="Response",
            user_id="test_user",
            prompt="Test",
            model_used="gemini-pro",
        )
        mock_service.chat.return_value = mock_response

        response = client.post(
            "/api/v1/hermes/chat", json={"message": "Test", "include_context": False}
        )

        assert response.status_code == 200


@pytest.mark.unit
class TestHealthCheck:
    """Test health check endpoint"""

    def test_health_check(self, client):
        """Test health check returns 200"""
        response = client.get("/api/v1/hermes/health")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"
        assert data["service"] == "hermes"


@pytest.mark.unit
class TestClearContextRoute:
    """Test clear context endpoint"""

    @patch("app.hermes.routes.get_hermes_service")
    @patch("app.hermes.routes.IdentityService.get_identity_fingerprint")
    def test_clear_context(self, mock_identity, mock_service_getter, client):
        """Test clearing conversation context"""
        mock_identity.return_value = {
            "user_id": "test_user",
            "ip_address": "127.0.0.1",
            "user_agent": "test",
            "accept_language": "en-US",
        }

        mock_service = Mock()
        mock_service_getter.return_value = mock_service

        response = client.post("/api/v1/hermes/clear-context")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "user_id" in data

        # Verify service method was called
        mock_service.clear_conversation_context.assert_called_once()


@pytest.mark.unit
class TestFilesRoute:
    """Test files listing endpoint"""

    def test_files_endpoint(self, client):
        """Test files listing"""
        with patch("os.listdir", return_value=["file1.txt", "file2.txt"]):
            response = client.get("/api/v1/hermes/files")

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["count"] == 2
            assert "files" in data


@pytest.mark.unit
class TestErrorHandlers:
    """Test error handler endpoints"""

    @patch("app.hermes.routes.get_hermes_service")
    @patch("app.hermes.routes.IdentityService.get_identity_fingerprint")
    def test_hermes_error_handler(self, mock_identity, mock_service_getter, client):
        """Test HermesError handler"""
        from app.hermes.exceptions import InvalidRequestError

        mock_identity.return_value = {
            "user_id": "test_user",
            "ip_address": "127.0.0.1",
            "user_agent": "test",
            "accept_language": "en-US",
        }

        mock_service = Mock()
        mock_service_getter.return_value = mock_service
        mock_service.process_request.side_effect = InvalidRequestError("Test error")

        response = client.get("/api/v1/hermes/process_request?request_text=test")

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
