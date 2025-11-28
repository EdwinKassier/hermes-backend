"""Unit tests for LegionGraphService."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from app.hermes.legion.graph_service import LegionGraphService
from app.hermes.legion.state.graph_state import OrchestratorState
from app.hermes.models import GeminiResponse, ResponseMode, UserIdentity


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

        # Mock LLM service to avoid LangChain dependency
        mock_llm_service = AsyncMock()
        mock_get_llm.return_value = mock_llm_service
        mock_nodes_async_service.return_value = mock_llm_service
        mock_memory_llm.return_value = mock_llm_service

        # Mock Gemini service
        mock_gemini = Mock()
        mock_decomposer_service.return_value = mock_gemini
        mock_nodes_service.return_value = mock_gemini
        mock_orchestrator_service.return_value = mock_gemini
        mock_research_service.return_value = mock_gemini
        mock_code_service.return_value = mock_gemini
        mock_analysis_service.return_value = mock_gemini
        mock_data_service.return_value = mock_gemini

        service = LegionGraphService(checkpoint_db_path=":memory:")

        # Mock persistence to support async context manager
        mock_persistence = Mock()
        mock_checkpointer = AsyncMock()
        mock_checkpointer.__aenter__.return_value = Mock()
        mock_checkpointer.__aexit__.return_value = None
        mock_persistence.get_checkpointer.return_value = mock_checkpointer
        service._persistence = mock_persistence

        service._graph = mock_graph  # Still setting it, though maybe unused

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

    async def test_process_request_success(self, legion_graph_service, user_identity):
        """Test process_request success flow."""
        # Setup mock graph response
        mock_result = {
            "messages": [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response"},
            ],
            "decision_rationale": [],
            "metadata": {"parallel_execution_metrics": {}},
        }

        # Helper to create async iterator
        async def async_iter(items):
            for item in items:
                yield item

        # We need to access the mock_graph that get_orchestration_graph returns
        # Since we don't have direct access to the mock object created in fixture easily inside the test method
        # unless we inspect the service or the patch.
        # But we set service._graph = mock_graph in fixture, and get_orchestration_graph returns mock_graph.
        # So service._graph IS the mock graph.

        legion_graph_service._graph.astream = Mock(
            return_value=async_iter([mock_result])
        )

        # Execute
        result = await legion_graph_service.process_request(
            "Test message", user_identity, response_mode=ResponseMode.TEXT
        )

        # Verify
        assert result.message == "Response"
        assert result.user_id == "test_user"
        assert result.metadata["legion_mode"] is True
        assert result.metadata["langgraph_enabled"] is True

        # Verify graph astream called
        legion_graph_service._graph.astream.assert_called_once()

    async def test_chat_success(self, legion_graph_service, user_identity):
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

        async def async_iter(items):
            for item in items:
                yield item

        legion_graph_service._graph.astream = Mock(
            return_value=async_iter([mock_result])
        )

        # Execute
        response = await legion_graph_service.chat("Test message", user_identity)

        # Verify
        assert isinstance(response, GeminiResponse)
        assert response.content == "Response"
        assert response.user_id == "test_user"
        assert response.metadata["legion_mode"] is True

    async def test_orchestration_rationale_generation(
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
            "legion_strategy": "parallel",  # Add strategy to trigger parallel mode
            "metadata": {
                "agents_used": ["agent1", "agent2"],
                "tools_used": ["tool1"],
            },
        }

        async def async_iter(items):
            for item in items:
                yield item

        legion_graph_service._graph.astream = Mock(
            return_value=async_iter([mock_result])
        )

        # Execute
        result = await legion_graph_service.process_request(
            "Test message", user_identity
        )

        # Verify rationale in metadata - now in 'rationale' block
        assert "rationale" in result.metadata
        rationale = result.metadata["rationale"]
        assert "Multi-agent parallel execution" in rationale["summary"]

        # Verify execution block
        assert "execution" in result.metadata
        execution = result.metadata["execution"]
        assert execution["mode"] == "parallel"

    async def test_error_handling(self, legion_graph_service, user_identity):
        """Test error handling during graph execution."""
        # Setup mock graph to raise exception
        legion_graph_service._graph.astream = Mock(side_effect=Exception("Graph error"))

        # Execute and verify raises AIServiceError
        from app.hermes.exceptions import AIServiceError

        with pytest.raises(AIServiceError):
            await legion_graph_service.process_request("Test message", user_identity)
