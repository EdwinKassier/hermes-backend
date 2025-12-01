from unittest.mock import MagicMock, patch

import pytest

from app.hermes.models import ResponseMode, UserIdentity
from app.hermes.schemas import ChatMessageSchema, ProcessRequestSchema
from app.hermes.services import HermesService


class TestLegionPersona:
    """Test suite for Legion persona implementation."""

    def test_schema_validation_allows_legion(self):
        """Test that schemas accept 'legion' as a valid persona."""
        # Test ProcessRequestSchema
        req = ProcessRequestSchema(request_text="test", persona="legion")
        assert req.persona == "legion"

        # Test ChatMessageSchema
        chat = ChatMessageSchema(message="test", persona="legion")
        assert chat.persona == "legion"

    def test_schema_validation_rejects_invalid(self):
        """Test that schemas reject invalid personas."""
        with pytest.raises(ValueError):
            ProcessRequestSchema(request_text="test", persona="invalid_persona")

    @patch("app.hermes.legion.graph_service.LegionGraphService")
    @patch("app.hermes.services.asyncio.run")
    def test_process_request_enforces_legion_persona(
        self, mock_asyncio_run, MockLegionService
    ):
        """Test that process_request enforces 'legion' persona when legion_mode is True."""
        # Setup mocks
        mock_legion_instance = MockLegionService.return_value
        mock_asyncio_run.return_value = MagicMock()

        service = HermesService()
        user_identity = UserIdentity(
            user_id="test_user", ip_address="127.0.0.1", user_agent="test_agent"
        )

        # Call with legion_mode=True and persona="hermes" (should be overridden)
        service.process_request(
            text="test request",
            user_identity=user_identity,
            response_mode=ResponseMode.TEXT,
            persona="hermes",
            legion_mode=True,
        )

        # Verify LegionGraphService was called with persona="legion"
        mock_legion_instance.process_request.assert_called_once()
        call_args = mock_legion_instance.process_request.call_args[1]
        assert call_args["persona"] == "legion"
        assert call_args["text"] == "test request"

    @patch("app.hermes.legion.graph_service.LegionGraphService")
    @patch("app.hermes.services.asyncio.run")
    def test_chat_enforces_legion_persona(self, mock_asyncio_run, MockLegionService):
        """Test that chat enforces 'legion' persona when legion_mode is True."""
        # Setup mocks
        mock_legion_instance = MockLegionService.return_value
        mock_asyncio_run.return_value = MagicMock()

        service = HermesService()
        user_identity = UserIdentity(
            user_id="test_user", ip_address="127.0.0.1", user_agent="test_agent"
        )

        # Call with legion_mode=True and persona="hermes" (should be overridden)
        service.chat(
            message="test message",
            user_identity=user_identity,
            persona="hermes",
            legion_mode=True,
        )

        # Verify LegionGraphService was called with persona="legion"
        mock_legion_instance.chat.assert_called_once()
        call_args = mock_legion_instance.chat.call_args[1]
        assert call_args["persona"] == "legion"
        assert call_args["message"] == "test message"
