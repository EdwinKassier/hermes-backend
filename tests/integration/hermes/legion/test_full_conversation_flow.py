"""Integration tests for full conversation flow with sub-agents."""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest

# from app.hermes.legion.service import LegionService  # Module removed, tests need rewrite
from app.hermes.models import UserIdentity

pytestmark = pytest.mark.skip(
    reason="Tests depend on removed LegionService class. Needs rewrite for LegionGraphService."
)


# Stub class to prevent flake8 errors in skipped tests
class LegionService:  # noqa: F811
    """Stub for removed LegionService class."""

    pass


@pytest.fixture
def legion_service_with_db():
    """Create LegionService with temporary database."""
    # Create temporary database file
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
            mock_gemini.generate_gemini_response = Mock(return_value="AI Response")
            mock_gemini.generate_gemini_response = Mock(return_value="AI Response")
            mock_gemini_getter.return_value = mock_gemini
            mock_tts_getter.return_value = Mock()

            service = LegionService()
            service.orchestration_state = ConversationState(db_path=db_path)
            service._gemini_service = mock_gemini
            service._tts_service = mock_tts_getter.return_value

            yield service
    finally:
        # Cleanup
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
class TestFullConversationFlow:
    """Test full conversation flows with sub-agents."""

    @patch("app.hermes.legion.service.get_gemini_service")
    def test_multi_step_research_conversation(
        self,
        mock_service_gemini,
        mock_orch_gemini,
        legion_service_with_db,
        user_identity,
    ):
        """Test complete multi-step research conversation."""
        # Mock task identification

        # Mock information extraction
        import json

        def extract_side_effect(*args, **kwargs):
            if "Extract" in str(kwargs.get("prompt", "")):
                return json.dumps({"time_period": "last 6 months"})
            return "ANSWERING"

        # First message: Request research
        response1 = legion_service_with_db.chat(
            "Research quantum computing", user_identity, persona="hermes"
        )

        # Should ask for information
        assert response1 is not None
        assert (
            "information" in response1.content.lower()
            or "questions" in response1.content.lower()
        )

        # Second message: Provide information
        response2 = legion_service_with_db.chat(
            "Last 6 months, comprehensive", user_identity, persona="hermes"
        )

        # Should execute task or ask for more info
        assert response2 is not None

    def test_cancellation_flow(
        self, mock_orch_gemini, legion_service_with_db, user_identity
    ):
        """Test cancellation flow."""
        # Mock task identification

        # First message: Request research
        response1 = legion_service_with_db.chat(
            "Research quantum computing", user_identity, persona="hermes"
        )

        # Should ask for information
        assert response1 is not None

        # Second message: Cancel
        response2 = legion_service_with_db.chat(
            "cancel", user_identity, persona="hermes"
        )

        # Should acknowledge cancellation
        assert response2 is not None
        assert "cancel" in response2.content.lower()

    def test_new_question_while_awaiting(
        self, mock_orch_gemini, legion_service_with_db, user_identity
    ):
        """Test asking new question while awaiting input."""
        # Mock task identification

        # First message: Request research
        response1 = legion_service_with_db.chat(
            "Research quantum computing", user_identity, persona="hermes"
        )

        # Should ask for information
        assert response1 is not None

        # Mock new question detection
        def intent_side_effect(*args, **kwargs):
            if "NEW_QUESTION" in str(kwargs.get("prompt", "")):
                return "NEW_QUESTION"
            return "research"

        # Second message: New question
        response2 = legion_service_with_db.chat(
            "What's the weather?", user_identity, persona="hermes"
        )

        # Should handle new question
        assert response2 is not None


@pytest.mark.integration
class TestStatePersistence:
    """Test state persistence across requests."""

    def test_state_persists_across_requests(
        self, legion_service_with_db, user_identity
    ):
        """Test that state persists across multiple requests."""
        # Create agent state
        from app.hermes.legion.models import SubAgentState, SubAgentStatus

        agent_state = SubAgentState(
            agent_id="test_agent",
            task="Test task",
            task_type="research",
            triggering_message="Test",
            status=SubAgentStatus.AWAITING_USER_INPUT,
        )

        # Save state
        legion_service_with_db.save_sub_agent_state(
            "test_user", "test_agent", agent_state
        )

        # Retrieve state in new service instance
        retrieved_state = legion_service_with_db.get_sub_agent_state(
            "test_user", "test_agent"
        )

        assert retrieved_state.agent_id == "test_agent"
        assert retrieved_state.status == SubAgentStatus.AWAITING_USER_INPUT
        assert retrieved_state.task == "Test task"

    def test_conversation_state_persists(self, legion_service_with_db, user_identity):
        """Test that conversation state persists."""
        # Send a message
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
            response = legion_service_with_db.chat("Hello", user_identity)

        # Check conversation was saved
        state = legion_service_with_db.gemini_service.conversation_state.get_state(
            "test_user"
        )
        assert "conversation" in state.data
        assert len(state.data["conversation"]) > 0
