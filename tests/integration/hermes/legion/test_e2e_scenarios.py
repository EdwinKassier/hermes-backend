"""End-to-end scenario tests for Legion sub-agent orchestration."""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from app.hermes.legion.service import LegionService
from app.hermes.models import ResponseMode, UserIdentity

pytestmark = pytest.mark.skip(
    reason="Tests depend on removed LegionService class. Needs rewrite for LegionGraphService."
)


@pytest.fixture
def legion_service_with_db():
    """Create LegionService with temporary database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        db_path = tmp_db.name

    try:
        with (
            patch("app.hermes.legion.service.get_gemini_service") as mock_gemini_getter,
            patch("app.hermes.legion.service.get_tts_service") as mock_tts_getter,
        ):
            from app.shared.utils.conversation_state import ConversationState

            mock_gemini = Mock()
            conversation_state = ConversationState(db_path=db_path)
            mock_gemini.conversation_state = conversation_state
            mock_gemini.generate_gemini_response_with_rag = Mock(
                return_value="AI Response"
            )
            mock_gemini.generate_gemini_response = Mock(return_value="AI Response")
            mock_gemini_getter.return_value = mock_gemini
            mock_tts_getter.return_value = Mock()

            service = LegionService()
            service.orchestration_state = ConversationState(db_path=db_path)
            service._gemini_service = mock_gemini
            service._tts_service = mock_tts_getter.return_value

            yield service
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


@pytest.fixture
def user_identity():
    """Create test user identity."""
    return UserIdentity(
        user_id="test_user",
        ip_address="127.0.0.1",
        user_agent="test-agent",
        accept_language="en-US",
    )


@pytest.mark.integration
@pytest.mark.slow
class TestE2EScenarios:
    """End-to-end scenario tests."""

    @patch("app.hermes.legion.orchestrator.get_gemini_service")
    def test_complete_research_scenario(
        self, mock_orch_gemini, legion_service_with_db, user_identity
    ):
        """Test complete research scenario from start to finish."""
        import json

        # Mock task identification
        def gemini_side_effect(*args, **kwargs):
            prompt = str(kwargs.get("prompt", ""))
            if "identify if it requires" in prompt:
                return "research"
            elif "Extract" in prompt:
                return json.dumps({"time_period": "2024", "depth": "comprehensive"})
            elif "answering" in prompt.lower():
                return "ANSWERING"
            return "AI Response"

        mock_orch_gemini.return_value.generate_gemini_response.side_effect = (
            gemini_side_effect
        )

        # Step 1: User requests research
        response1 = legion_service_with_db.chat(
            "Research quantum computing advances", user_identity
        )

        # Should identify as research task and ask for info
        assert response1 is not None

        # Step 2: User provides information
        response2 = legion_service_with_db.chat(
            "2024, comprehensive research", user_identity
        )

        # Should extract info and either ask for more or execute
        assert response2 is not None

    @patch("app.hermes.legion.orchestrator.get_gemini_service")
    def test_cancellation_scenario(
        self, mock_orch_gemini, legion_service_with_db, user_identity
    ):
        """Test cancellation scenario."""
        # Mock task identification
        mock_orch_gemini.return_value.generate_gemini_response.return_value = "research"

        # Step 1: Request task
        response1 = legion_service_with_db.chat(
            "Research quantum computing", user_identity
        )

        # Step 2: Cancel
        response2 = legion_service_with_db.chat("never mind", user_identity)

        # Should acknowledge cancellation
        assert response2 is not None
        assert (
            "cancel" in response2.content.lower()
            or "never mind" in response2.content.lower()
        )

    def test_process_request_with_legion_mode(
        self, legion_service_with_db, user_identity
    ):
        """Test process_request endpoint with legion mode."""
        with patch.object(
            legion_service_with_db, "_generate_general_response"
        ) as mock_general:
            from app.hermes.models import GeminiResponse

            mock_general.return_value = GeminiResponse(
                content="Response",
                user_id="test_user",
                prompt="Test",
                model_used="gemini-pro",
            )

            result = legion_service_with_db.process_request(
                "Test message",
                user_identity,
                response_mode=ResponseMode.TEXT,
                persona="hermes",
            )

            assert result.message == "Response"
            assert result.metadata.get("legion_mode") is True

    def test_backward_compatibility(self, legion_service_with_db, user_identity):
        """Test that system works without legion mode (backward compatibility)."""
        # This test verifies that the system doesn't break existing functionality
        # when legion_mode is False (tested in regular HermesService tests)
        # Here we just verify LegionService can handle general conversations
        with patch.object(
            legion_service_with_db, "_generate_general_response"
        ) as mock_general:
            from app.hermes.models import GeminiResponse

            mock_general.return_value = GeminiResponse(
                content="General response",
                user_id="test_user",
                prompt="Hello",
                model_used="gemini-pro",
            )

            response = legion_service_with_db.chat("Hello", user_identity)

            assert response.content == "General response"
            assert response.metadata.get("legion_mode") is True
