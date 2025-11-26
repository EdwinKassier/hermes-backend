"""Unit tests for LegionGraphService."""

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest

from app.hermes.legion.graph_service import LegionGraphService
from app.hermes.legion.state.graph_state import OrchestratorState
from app.hermes.models import GeminiResponse, ResponseMode, UserIdentity


@pytest.fixture
def legion_graph_service():
    """Create LegionGraphService with mocked dependencies."""
    with (
        patch(
            "app.hermes.legion.graph_service.get_gemini_service"
        ) as mock_gemini_getter,
        patch("app.hermes.legion.graph_service.get_tts_service") as mock_tts_getter,
        patch(
            "app.hermes.legion.graph_service.get_orchestration_graph"
        ) as mock_get_graph,
    ):
        mock_gemini = Mock()
        mock_gemini_getter.return_value = mock_gemini
        mock_tts_getter.return_value = Mock()

        mock_graph = Mock()
        mock_get_graph.return_value = mock_graph

        service = LegionGraphService(checkpoint_db_path=":memory:")
        service._gemini_service = mock_gemini
        service._tts_service = mock_tts_getter.return_value
        service._graph = mock_graph

        yield service


@pytest.fixture
def user_identity():
    """Create test user identity."""
    return UserIdentity(
        user_id="test_user",
        ip_address="127.0.0.1",
        user_agent="test-agent",
        accept_language="en-US",
    )


@pytest.mark.unit
class TestLegionGraphService:
    """Test LegionGraphService methods."""

    def test_process_request_success(self, legion_graph_service, user_identity):
        """Test process_request success flow."""
        # Setup mock graph response - graph.stream() returns an iterator
        mock_result = {
            "messages": [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response"},
            ],
            "decision_rationale": [],
            "metadata": {"parallel_execution_metrics": {}},
        }
        # stream() yields chunks, so we return an iterator
        legion_graph_service.graph.stream.return_value = iter([mock_result])

        # Execute
        result = legion_graph_service.process_request(
            "Test message", user_identity, response_mode=ResponseMode.TEXT
        )

        # Verify
        assert result.message == "Response"
        assert result.user_id == "test_user"
        assert result.metadata["legion_mode"] is True
        assert result.metadata["langgraph_enabled"] is True

        # Verify graph stream called
        legion_graph_service.graph.stream.assert_called_once()

    def test_chat_success(self, legion_graph_service, user_identity):
        """Test chat success flow."""
        # Setup mock graph response
        mock_result = {
            "messages": [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response"},
            ],
            "decision_rationale": [],
            "metadata": {},
        }
        legion_graph_service.graph.stream.return_value = iter([mock_result])

        # Execute
        response = legion_graph_service.chat("Test message", user_identity)

        # Verify
        assert isinstance(response, GeminiResponse)
        assert response.content == "Response"
        assert response.user_id == "test_user"
        assert response.metadata["legion_mode"] is True

    def test_orchestration_rationale_generation(
        self, legion_graph_service, user_identity
    ):
        """Test generation of orchestration rationale."""
        # Setup mock graph response with rationale
        mock_rationale = [
            {
                "analysis": {"multi_agent_task_detected": True, "subtasks_count": 2},
                "decisions": {"action": "legion_orchestration"},
            }
        ]
        mock_result = {
            "messages": [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response"},
            ],
            "decision_rationale": mock_rationale,
            "agents_used": ["agent1", "agent2"],
            "tools_used": ["tool1"],
            "metadata": {},
        }
        legion_graph_service.graph.stream.return_value = iter([mock_result])

        # Execute
        result = legion_graph_service.process_request("Test message", user_identity)

        # Verify rationale in metadata
        assert "orchestration_rationale" in result.metadata
        rationale = result.metadata["orchestration_rationale"]
        assert rationale["execution_mode"] == "parallel"
        assert len(rationale["agents"]) == 2
        assert "Multi-agent parallel execution" in rationale["orchestration_structure"]

    def test_error_handling(self, legion_graph_service, user_identity):
        """Test error handling during graph execution."""
        # Setup mock graph to raise exception
        legion_graph_service.graph.invoke.side_effect = Exception("Graph error")

        # Execute and verify raises AIServiceError
        from app.hermes.exceptions import AIServiceError

        with pytest.raises(AIServiceError):
            legion_graph_service.process_request("Test message", user_identity)
