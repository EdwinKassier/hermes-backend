"""Unit tests for Legion interrupt functionality."""

import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from app.hermes.legion.graph_service import LegionGraphService
from app.hermes.legion.state.graph_state import OrchestratorState
from app.hermes.models import GeminiResponse, UserIdentity


@pytest.fixture
def legion_graph_service():
    """Create LegionGraphService with mocked dependencies."""
    with (
        patch("app.hermes.legion.graph_service.TTSService") as MockTTSService,
        patch(
            "app.hermes.legion.graph_service.get_orchestration_graph"
        ) as mock_get_graph,
        patch(
            "app.hermes.legion.nodes.legion_orchestrator.get_strategy_registry"
        ) as mock_get_registry,
        patch(
            "app.hermes.legion.utils.conversation_memory.get_async_llm_service"
        ) as mock_memory_llm,
        patch(
            "app.hermes.legion.intelligence.routing_service.get_async_llm_service"
        ) as mock_get_llm,
        patch(
            "app.hermes.legion.parallel.task_decomposer.get_gemini_service"
        ) as mock_decomposer_service,
        patch(
            "app.hermes.legion.nodes.graph_nodes.get_gemini_service"
        ) as mock_nodes_service,
        patch(
            "app.hermes.legion.nodes.graph_nodes.get_async_llm_service"
        ) as mock_nodes_async_service,
        patch(
            "app.hermes.legion.orchestrator.get_gemini_service"
        ) as mock_orchestrator_service,
        patch(
            "app.hermes.legion.agents.research_agent.get_gemini_service"
        ) as mock_research_service,
        patch(
            "app.hermes.legion.agents.code_agent.get_gemini_service"
        ) as mock_code_service,
        patch(
            "app.hermes.legion.agents.analysis_agent.get_gemini_service"
        ) as mock_analysis_service,
        patch(
            "app.hermes.legion.agents.data_agent.get_gemini_service"
        ) as mock_data_service,
    ):
        mock_tts_instance = Mock()
        MockTTSService.return_value = mock_tts_instance

        mock_graph = Mock()
        mock_get_graph.return_value = mock_graph

        mock_llm_service = AsyncMock()
        mock_get_llm.return_value = mock_llm_service
        mock_nodes_async_service.return_value = mock_llm_service
        mock_memory_llm.return_value = mock_llm_service

        mock_gemini = Mock()
        mock_decomposer_service.return_value = mock_gemini
        mock_nodes_service.return_value = mock_gemini
        mock_orchestrator_service.return_value = mock_gemini
        mock_research_service.return_value = mock_gemini
        mock_code_service.return_value = mock_gemini
        mock_analysis_service.return_value = mock_gemini
        mock_data_service.return_value = mock_gemini

        service = LegionGraphService(checkpoint_db_path=":memory:")

        # Mock persistence
        mock_persistence = Mock()
        mock_checkpointer = AsyncMock()
        mock_checkpointer.__aenter__.return_value = Mock()
        mock_checkpointer.__aexit__.return_value = None
        mock_persistence.get_checkpointer.return_value = mock_checkpointer
        service._persistence = mock_persistence

        service._graph = mock_graph

        yield service, mock_graph


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

    async def test_graph_interrupts_on_legion_orchestration(
        self, user_identity, legion_graph_service
    ):
        """Test that graph interrupts when hitting legion_orchestrator node."""
        service, mock_graph = legion_graph_service

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

    async def test_resume_after_interrupt_approval(
        self, user_identity, legion_graph_service
    ):
        """Test resuming execution after user approves interrupt."""
        service, mock_graph = legion_graph_service

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

        # Resume with approval
        response = await service.resume_execution(
            user_identity, resume_value={"action": "approve"}, persona="hermes"
        )

        # Verify execution continued
        assert response.content == "Task completed successfully"
        assert response.metadata["legion_mode"] is True
        assert response.metadata["status"]["interrupted"] is False

    async def test_resume_with_modification(self, user_identity, legion_graph_service):
        """Test resuming with modified worker plan."""
        service, mock_graph = legion_graph_service

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

    async def test_cancellation_during_interrupt(
        self, user_identity, legion_graph_service
    ):
        """Test cancelling execution during interrupt."""
        service, mock_graph = legion_graph_service

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

        # Resume with cancel
        response = await service.resume_execution(
            user_identity, resume_value={"action": "cancel"}, persona="hermes"
        )

        assert "cancelled" in response.content.lower()
