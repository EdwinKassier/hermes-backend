"""Unit tests for Legion interrupt functionality."""

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest

from app.hermes.legion.graph_service import LegionGraphService
from app.hermes.legion.state.graph_state import OrchestratorState
from app.hermes.models import GeminiResponse, UserIdentity


@pytest.fixture
def user_identity():
    """Create test user identity."""
    return UserIdentity(
        user_id="test_user_interrupt",
        ip_address="127.0.0.1",
        user_agent="test-agent",
        accept_language="en-US",
    )


@pytest.mark.unit
class TestLegionInterrupts:
    """Test Legion interrupt and resume functionality."""

    def test_graph_interrupts_on_legion_orchestration(self, user_identity):
        """Test that graph interrupts when hitting legion_orchestrator node."""
        with (
            patch(
                "app.hermes.legion.graph_service.get_gemini_service"
            ) as mock_gemini_getter,
            patch("app.hermes.legion.graph_service.get_tts_service") as mock_tts_getter,
            patch(
                "app.hermes.legion.graph_service.get_orchestration_graph"
            ) as mock_get_graph,
        ):
            # Setup mocks
            mock_gemini_getter.return_value = Mock()
            mock_tts_getter.return_value = Mock()

            # Mock graph that simulates interrupt
            mock_graph = Mock()

            # Simulate stream response with interrupt
            def mock_stream(inputs, config, stream_mode):
                yield {
                    "__interrupt__": {
                        "type": "worker_plan_review",
                        "strategy": "council",
                        "workers": [
                            {"worker_id": "worker_1", "role": "researcher"},
                            {"worker_id": "worker_2", "role": "analyst"},
                        ],
                        "worker_count": 2,
                        "message": "I've planned 2 workers using council strategy.",
                        "actions": ["approve", "modify", "cancel"],
                    }
                }

            mock_graph.stream.side_effect = mock_stream
            mock_get_graph.return_value = mock_graph

            # Create service
            service = LegionGraphService(checkpoint_db_path=":memory:")
            service._graph = mock_graph

            # Execute - should hit interrupt
            response = service.chat(
                "Complex multi-agent task", user_identity, persona="hermes"
            )

            # Verify interrupt detected
            assert response.content == ""  # Empty during interrupt
            assert response.metadata["interrupted"] is True
            assert "interrupt_data" in response.metadata
            assert response.metadata["interrupt_data"]["type"] == "worker_plan_review"
            assert response.metadata["interrupt_data"]["worker_count"] == 2
            assert "thread_id" in response.metadata

    def test_resume_after_interrupt_approval(self, user_identity):
        """Test resuming execution after user approves interrupt."""
        with (
            patch(
                "app.hermes.legion.graph_service.get_gemini_service"
            ) as mock_gemini_getter,
            patch("app.hermes.legion.graph_service.get_tts_service") as mock_tts_getter,
            patch(
                "app.hermes.legion.graph_service.get_orchestration_graph"
            ) as mock_get_graph,
        ):
            # Setup mocks
            mock_gemini_getter.return_value = Mock()
            mock_tts_getter.return_value = Mock()

            mock_graph = Mock()

            # Simulate stream response after resume (completed execution)
            def mock_stream_after_resume(inputs, config, stream_mode):
                # Return completed state
                yield {
                    "messages": [
                        {"role": "user", "content": "Task"},
                        {"role": "assistant", "content": "Task completed successfully"},
                    ],
                    "decision_rationale": [],
                    "metadata": {},
                }

            mock_graph.stream.side_effect = mock_stream_after_resume
            mock_get_graph.return_value = mock_graph

            # Create service
            service = LegionGraphService(checkpoint_db_path=":memory:")
            service._graph = mock_graph

            # Resume with approval
            response = service.resume(
                user_identity, resume_value={"action": "approve"}, persona="hermes"
            )

            # Verify execution continued
            assert response.content == "Task completed successfully"
            assert response.metadata["resumed"] is True
            assert response.metadata["interrupted"] is False
            assert response.metadata["legion_mode"] is True

    def test_resume_with_modification(self, user_identity):
        """Test resuming with modified worker plan."""
        with (
            patch(
                "app.hermes.legion.graph_service.get_gemini_service"
            ) as mock_gemini_getter,
            patch("app.hermes.legion.graph_service.get_tts_service") as mock_tts_getter,
            patch(
                "app.hermes.legion.graph_service.get_orchestration_graph"
            ) as mock_get_graph,
        ):
            mock_gemini_getter.return_value = Mock()
            mock_tts_getter.return_value = Mock()

            mock_graph = Mock()

            def mock_stream_modified(inputs, config, stream_mode):
                # Verify modified workers were passed
                if isinstance(inputs, dict) and inputs.get("action") == "modify":
                    assert len(inputs.get("workers", [])) == 1  # User reduced workers

                yield {
                    "messages": [
                        {"role": "user", "content": "Task"},
                        {"role": "assistant", "content": "Executed with modified plan"},
                    ],
                    "decision_rationale": [],
                    "metadata": {},
                }

            mock_graph.stream.side_effect = mock_stream_modified
            mock_get_graph.return_value = mock_graph

            service = LegionGraphService(checkpoint_db_path=":memory:")
            service._graph = mock_graph

            # Resume with modified workers
            response = service.resume(
                user_identity,
                resume_value={
                    "action": "modify",
                    "workers": [
                        {"worker_id": "worker_1", "role": "researcher"}
                    ],  # Reduced from 2 to 1
                },
                persona="hermes",
            )

            assert response.content == "Executed with modified plan"
            assert response.metadata["resumed"] is True

    def test_cancellation_during_interrupt(self, user_identity):
        """Test cancelling execution during interrupt."""
        with (
            patch(
                "app.hermes.legion.graph_service.get_gemini_service"
            ) as mock_gemini_getter,
            patch("app.hermes.legion.graph_service.get_tts_service") as mock_tts_getter,
            patch(
                "app.hermes.legion.graph_service.get_orchestration_graph"
            ) as mock_get_graph,
        ):
            mock_gemini_getter.return_value = Mock()
            mock_tts_getter.return_value = Mock()

            mock_graph = Mock()

            def mock_stream_cancelled(inputs, config, stream_mode):
                # Simulate legion_orchestrator handling cancel
                if isinstance(inputs, dict) and inputs.get("action") == "cancel":
                    yield {
                        "messages": [
                            {"role": "user", "content": "Task"},
                            {
                                "role": "assistant",
                                "content": "Worker plan cancelled. How else can I help?",
                            },
                        ],
                        "next_action": "complete",
                        "metadata": {"legion_cancelled": True},
                    }

            mock_graph.stream.side_effect = mock_stream_cancelled
            mock_get_graph.return_value = mock_graph

            service = LegionGraphService(checkpoint_db_path=":memory:")
            service._graph = mock_graph

            # Resume with cancel
            response = service.resume(
                user_identity, resume_value={"action": "cancel"}, persona="hermes"
            )

            assert "cancelled" in response.content.lower()
            assert response.metadata["resumed"] is True
