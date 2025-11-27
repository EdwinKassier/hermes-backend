"""Unit tests for Legion interrupt functionality."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

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
@pytest.mark.asyncio
class TestLegionInterrupts:
    """Test Legion interrupt and resume functionality."""

    async def test_graph_interrupts_on_legion_orchestration(self, user_identity):
        """Test that graph interrupts when hitting legion_orchestrator node."""
        with (
            patch("app.hermes.legion.graph_service.TTSService") as mock_tts_class,
            patch(
                "app.hermes.legion.graph_service.get_orchestration_graph"
            ) as mock_get_graph,
        ):
            # Setup mocks
            mock_tts_class.return_value = Mock()

            # Mock graph that simulates interrupt
            mock_graph = Mock()

            # Simulate stream response with interrupt
            async def mock_stream(inputs, config, stream_mode):
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

            mock_graph.astream.side_effect = mock_stream
            mock_get_graph.return_value = mock_graph

            # Create service
            service = LegionGraphService(checkpoint_db_path=":memory:")
            service._persistence = Mock()

            # Setup async context manager for get_checkpointer
            mock_cm = MagicMock()
            mock_cm.__aenter__ = AsyncMock(return_value=Mock())
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            service._persistence.get_checkpointer.return_value = mock_cm

            # Execute - should hit interrupt
            response = await service.chat(
                "Complex multi-agent task", user_identity, persona="hermes"
            )

            # Verify interrupt detected
            assert response.content == "I need your approval to proceed."
            assert response.metadata["legion_mode"] is True
            assert response.metadata["interrupted"] is True
            assert "interrupt_data" in response.metadata
            assert response.metadata["interrupt_data"]["type"] == "worker_plan_review"
            assert response.metadata["interrupt_data"]["worker_count"] == 2
            assert "thread_id" in response.metadata

    async def test_resume_after_interrupt_approval(self, user_identity):
        """Test resuming execution after user approves interrupt."""
        with (
            patch("app.hermes.legion.graph_service.TTSService") as mock_tts_class,
            patch(
                "app.hermes.legion.graph_service.get_orchestration_graph"
            ) as mock_get_graph,
        ):
            # Setup mocks
            mock_tts_class.return_value = Mock()

            mock_graph = Mock()

            # Simulate stream response after resume (completed execution)
            async def mock_stream_after_resume(inputs, config, stream_mode):
                # Return completed state
                yield {
                    "messages": [
                        {"role": "user", "content": "Task"},
                        {"role": "assistant", "content": "Task completed successfully"},
                    ],
                    "decision_rationale": [],
                    "metadata": {},
                }

            mock_graph.astream.side_effect = mock_stream_after_resume
            mock_get_graph.return_value = mock_graph

            # Create service
            service = LegionGraphService(checkpoint_db_path=":memory:")
            service._persistence = Mock()

            # Setup async context manager for get_checkpointer
            mock_cm = MagicMock()
            mock_cm.__aenter__ = AsyncMock(return_value=Mock())
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            service._persistence.get_checkpointer.return_value = mock_cm

            # Resume with approval
            response = await service.resume_execution(
                user_identity, resume_value={"action": "approve"}, persona="hermes"
            )

            # Verify execution continued
            assert response.content == "Task completed successfully"
            assert response.metadata["legion_mode"] is True
            assert response.metadata["interrupted"] is False

    async def test_resume_with_modification(self, user_identity):
        """Test resuming with modified worker plan."""
        with (
            patch("app.hermes.legion.graph_service.TTSService") as mock_tts_class,
            patch(
                "app.hermes.legion.graph_service.get_orchestration_graph"
            ) as mock_get_graph,
        ):
            mock_tts_class.return_value = Mock()

            mock_graph = Mock()

            async def mock_stream_modified(inputs, config, stream_mode):
                yield {
                    "messages": [
                        {"role": "user", "content": "Task"},
                        {"role": "assistant", "content": "Executed with modified plan"},
                    ],
                    "decision_rationale": [],
                    "metadata": {},
                }

            mock_graph.astream.side_effect = mock_stream_modified
            mock_get_graph.return_value = mock_graph

            service = LegionGraphService(checkpoint_db_path=":memory:")
            service._persistence = Mock()

            # Setup async context manager for get_checkpointer
            mock_cm = MagicMock()
            mock_cm.__aenter__ = AsyncMock(return_value=Mock())
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            service._persistence.get_checkpointer.return_value = mock_cm

            # Resume with modified workers
            response = await service.resume_execution(
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

    async def test_cancellation_during_interrupt(self, user_identity):
        """Test cancelling execution during interrupt."""
        with (
            patch("app.hermes.legion.graph_service.TTSService") as mock_tts_class,
            patch(
                "app.hermes.legion.graph_service.get_orchestration_graph"
            ) as mock_get_graph,
        ):
            mock_tts_class.return_value = Mock()

            mock_graph = Mock()

            async def mock_stream_cancelled(inputs, config, stream_mode):
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

            mock_graph.astream.side_effect = mock_stream_cancelled
            mock_get_graph.return_value = mock_graph

            service = LegionGraphService(checkpoint_db_path=":memory:")
            service._persistence = Mock()

            # Setup async context manager for get_checkpointer
            mock_cm = MagicMock()
            mock_cm.__aenter__ = AsyncMock(return_value=Mock())
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            service._persistence.get_checkpointer.return_value = mock_cm

            # Resume with cancel
            response = await service.resume_execution(
                user_identity, resume_value={"action": "cancel"}, persona="hermes"
            )

            assert "cancelled" in response.content.lower()
