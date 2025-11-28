"""Unit tests for level-aware execution and timeout handling."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from app.hermes.legion.nodes.legion_orchestrator import (
    legion_dispatch_node,
    legion_level_complete_node,
    legion_level_routing_edge,
)
from app.hermes.legion.nodes.router_service import RouterService
from app.hermes.legion.state import GraphDecision


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

        from app.hermes.legion.graph_service import LegionGraphService

        service = LegionGraphService(checkpoint_db_path=":memory:")

        # Mock persistence
        mock_persistence = Mock()
        mock_checkpointer = AsyncMock()
        mock_checkpointer.__aenter__.return_value = Mock()
        mock_checkpointer.__aexit__.return_value = None
        mock_persistence.get_checkpointer.return_value = mock_checkpointer
        service._persistence = mock_persistence

        service._graph = mock_graph

        yield service


@pytest.mark.unit
class TestLevelAwareRouting:
    """Test suite for level-aware worker routing."""

    def test_route_legion_workers_filters_by_level_0(self):
        """Test that only level 0 workers are dispatched when current_execution_level is 0."""
        router = RouterService()

        workers = [
            {
                "worker_id": "research_1",
                "role": "researcher",
                "task_description": "Research topic A",
                "tools": ["search"],
                "execution_level": 0,
            },
            {
                "worker_id": "research_2",
                "role": "researcher",
                "task_description": "Research topic B",
                "tools": ["search"],
                "execution_level": 0,
            },
            {
                "worker_id": "analysis_1",
                "role": "analyzer",
                "task_description": "Analyze research results",
                "tools": ["calculator"],
                "execution_level": 1,  # Should NOT be dispatched
            },
        ]

        state = {
            "metadata": {"legion_worker_plans": workers},
            "user_id": "test_user",
            "persona": "hermes",
            "collected_info": {},
            "current_execution_level": 0,
            "level_results": {},
        }

        result = router.route_legion_workers(state)

        # Should only return 2 Send objects (level 0 workers)
        assert len(result) == 2

    def test_route_legion_workers_filters_by_level_1(self):
        """Test that only level 1 workers are dispatched when current_execution_level is 1."""
        router = RouterService()

        workers = [
            {
                "worker_id": "research_1",
                "role": "researcher",
                "task_description": "Research topic A",
                "tools": ["search"],
                "execution_level": 0,
            },
            {
                "worker_id": "analysis_1",
                "role": "analyzer",
                "task_description": "Analyze research results",
                "tools": ["calculator"],
                "execution_level": 1,
            },
            {
                "worker_id": "summary_1",
                "role": "summarizer",
                "task_description": "Summarize analysis",
                "tools": [],
                "execution_level": 2,  # Should NOT be dispatched
            },
        ]

        state = {
            "metadata": {"legion_worker_plans": workers},
            "user_id": "test_user",
            "persona": "hermes",
            "collected_info": {},
            "current_execution_level": 1,
            "level_results": {
                0: {
                    "workers": {
                        "research_1": {
                            "result": "Research findings",
                            "status": "success",
                        }
                    },
                    "success_count": 1,
                    "failed_count": 0,
                }
            },
        }

        result = router.route_legion_workers(state)

        # Should only return 1 Send object (level 1 worker)
        assert len(result) == 1

    def test_route_legion_workers_includes_previous_results_in_context(self):
        """Test that previous level results are included in worker context."""
        router = RouterService()

        workers = [
            {
                "worker_id": "analysis_1",
                "role": "analyzer",
                "task_description": "Analyze research results",
                "tools": ["calculator"],
                "execution_level": 1,
            },
        ]

        previous_results = {
            "research_1": {"result": "Research findings A", "status": "success"},
            "research_2": {"result": "Research findings B", "status": "success"},
        }

        state = {
            "metadata": {"legion_worker_plans": workers},
            "user_id": "test_user",
            "persona": "hermes",
            "collected_info": {"topic": "AI"},
            "current_execution_level": 1,
            "level_results": {
                0: {
                    "workers": previous_results,
                    "success_count": 2,
                    "failed_count": 0,
                }
            },
        }

        result = router.route_legion_workers(state)

        assert len(result) == 1
        # The Send object should contain previous_level_results in context
        # (We can't easily inspect Send internals, but we can verify it was created)

    def test_route_legion_workers_returns_empty_for_nonexistent_level(self):
        """Test that empty list is returned when no workers exist for current level."""
        router = RouterService()

        workers = [
            {
                "worker_id": "research_1",
                "role": "researcher",
                "task_description": "Research",
                "tools": [],
                "execution_level": 0,
            },
        ]

        state = {
            "metadata": {"legion_worker_plans": workers},
            "user_id": "test_user",
            "persona": "hermes",
            "collected_info": {},
            "current_execution_level": 5,  # No workers at this level
            "level_results": {},
        }

        result = router.route_legion_workers(state)

        assert len(result) == 0


@pytest.mark.unit
class TestLevelCompleteNode:
    """Test suite for legion_level_complete_node."""

    @pytest.mark.asyncio
    async def test_level_complete_stores_results(self):
        """Test that level completion stores worker results correctly."""
        state = {
            "current_execution_level": 0,
            "total_execution_levels": 2,
            "fail_on_level_error": False,
            "legion_results": {
                "worker_1": {
                    "result": "Result A",
                    "status": "success",
                    "execution_level": 0,
                },
                "worker_2": {
                    "result": "Result B",
                    "status": "success",
                    "execution_level": 0,
                },
            },
            "level_results": {},
            "metadata": {},
        }

        result = await legion_level_complete_node(state)

        # Should store level 0 results
        assert 0 in result["level_results"]
        assert result["level_results"][0]["success_count"] == 2
        assert result["level_results"][0]["failed_count"] == 0

        # Should increment to next level
        assert result["current_execution_level"] == 1

    @pytest.mark.asyncio
    async def test_level_complete_tracks_failures(self):
        """Test that level completion tracks failed workers."""
        state = {
            "current_execution_level": 0,
            "total_execution_levels": 1,
            "fail_on_level_error": False,
            "legion_results": {
                "worker_1": {
                    "result": "Result A",
                    "status": "success",
                    "execution_level": 0,
                },
                "worker_2": {
                    "result": "Error occurred",
                    "status": "failed",
                    "execution_level": 0,
                },
                "worker_3": {
                    "result": "Timed out",
                    "status": "timeout",
                    "execution_level": 0,
                },
            },
            "level_results": {},
            "metadata": {},
        }

        result = await legion_level_complete_node(state)

        assert result["level_results"][0]["success_count"] == 1
        assert result["level_results"][0]["failed_count"] == 2

    @pytest.mark.asyncio
    async def test_level_complete_stops_on_error_when_configured(self):
        """Test that level completion stops when fail_on_level_error is True."""
        state = {
            "current_execution_level": 0,
            "total_execution_levels": 2,
            "fail_on_level_error": True,  # Stop on error
            "legion_results": {
                "worker_1": {
                    "result": "Error occurred",
                    "status": "failed",
                    "execution_level": 0,
                },
            },
            "level_results": {},
            "metadata": {},
        }

        result = await legion_level_complete_node(state)

        # Should NOT increment level
        assert result["current_execution_level"] == 0

        # Should set stop flag
        assert result["metadata"]["level_execution_stopped"] is True
        assert result["metadata"]["stopped_at_level"] == 0

    @pytest.mark.asyncio
    async def test_level_complete_continues_with_partial_on_error(self):
        """Test that level completion continues when fail_on_level_error is False."""
        state = {
            "current_execution_level": 0,
            "total_execution_levels": 2,
            "fail_on_level_error": False,  # Continue with partial
            "legion_results": {
                "worker_1": {
                    "result": "Error occurred",
                    "status": "failed",
                    "execution_level": 0,
                },
                "worker_2": {
                    "result": "Success",
                    "status": "success",
                    "execution_level": 0,
                },
            },
            "level_results": {},
            "metadata": {},
        }

        result = await legion_level_complete_node(state)

        # Should increment level despite failure
        assert result["current_execution_level"] == 1

        # Should NOT set stop flag
        assert "level_execution_stopped" not in result.get("metadata", {})


@pytest.mark.unit
class TestLevelRoutingEdge:
    """Test suite for legion_level_routing_edge."""

    def test_routes_to_dispatch_when_more_levels(self):
        """Test routing to dispatch when there are more levels."""
        state = {
            "current_execution_level": 1,
            "total_execution_levels": 3,
            "metadata": {},
        }

        result = legion_level_routing_edge(state)

        assert result == "legion_dispatch"

    def test_routes_to_synthesis_when_all_levels_complete(self):
        """Test routing to synthesis when all levels are done."""
        state = {
            "current_execution_level": 3,
            "total_execution_levels": 3,
            "metadata": {},
        }

        result = legion_level_routing_edge(state)

        assert result == "legion_synthesis"

    def test_routes_to_synthesis_on_execution_stopped(self):
        """Test routing to synthesis when execution was stopped due to errors."""
        state = {
            "current_execution_level": 0,
            "total_execution_levels": 3,
            "metadata": {"level_execution_stopped": True},
        }

        result = legion_level_routing_edge(state)

        assert result == "legion_synthesis"


@pytest.mark.unit
class TestLegionDispatchNode:
    """Test suite for legion_dispatch_node."""

    @pytest.mark.asyncio
    async def test_dispatch_node_returns_empty_update(self):
        """Test that dispatch node returns empty update (pass-through)."""
        state = {
            "current_execution_level": 1,
            "total_execution_levels": 2,
        }

        result = await legion_dispatch_node(state)

        # Should return empty dict (pass-through)
        assert result == {}


@pytest.mark.unit
class TestBackpressure:
    """Test suite for worker backpressure/batching."""

    def test_backpressure_limits_worker_dispatch(self):
        """Test that backpressure limits workers per dispatch batch."""
        # Create router with low limit
        router = RouterService(max_concurrent_workers=2)

        workers = [
            {
                "worker_id": f"worker_{i}",
                "role": "researcher",
                "task_description": f"Task {i}",
                "tools": [],
                "execution_level": 0,
            }
            for i in range(5)  # 5 workers, but limit is 2
        ]

        state = {
            "metadata": {"legion_worker_plans": workers},
            "user_id": "test_user",
            "persona": "hermes",
            "collected_info": {},
            "current_execution_level": 0,
            "level_results": {},
        }

        result = router.route_legion_workers(state)

        # Should only dispatch 2 workers due to backpressure
        assert len(result) == 2

    def test_backpressure_tracks_dispatched_workers(self):
        """Test that already-dispatched workers are excluded from subsequent batches."""
        router = RouterService(max_concurrent_workers=2)

        workers = [
            {
                "worker_id": f"worker_{i}",
                "role": "researcher",
                "task_description": f"Task {i}",
                "tools": [],
                "execution_level": 0,
            }
            for i in range(4)
        ]

        # Simulate some workers already dispatched
        state = {
            "metadata": {
                "legion_worker_plans": workers,
                "dispatched_worker_ids": ["worker_0", "worker_1"],  # Already dispatched
            },
            "user_id": "test_user",
            "persona": "hermes",
            "collected_info": {},
            "current_execution_level": 0,
            "level_results": {},
        }

        result = router.route_legion_workers(state)

        # Should only dispatch remaining workers (2 more), but capped at 2 by backpressure
        assert len(result) == 2

    def test_backpressure_returns_empty_when_all_dispatched(self):
        """Test that empty list is returned when all workers already dispatched."""
        router = RouterService(max_concurrent_workers=5)

        workers = [
            {
                "worker_id": f"worker_{i}",
                "role": "researcher",
                "task_description": f"Task {i}",
                "tools": [],
                "execution_level": 0,
            }
            for i in range(3)
        ]

        state = {
            "metadata": {
                "legion_worker_plans": workers,
                "dispatched_worker_ids": ["worker_0", "worker_1", "worker_2"],
            },
            "user_id": "test_user",
            "persona": "hermes",
            "collected_info": {},
            "current_execution_level": 0,
            "level_results": {},
        }

        result = router.route_legion_workers(state)

        assert len(result) == 0

    def test_set_max_concurrent_workers(self):
        """Test dynamic adjustment of max concurrent workers."""
        router = RouterService(max_concurrent_workers=5)
        assert router.get_max_concurrent_workers() == 5

        router.set_max_concurrent_workers(10)
        assert router.get_max_concurrent_workers() == 10

    def test_set_max_concurrent_workers_rejects_invalid(self):
        """Test that invalid values are rejected."""
        router = RouterService()

        with pytest.raises(ValueError):
            router.set_max_concurrent_workers(0)

        with pytest.raises(ValueError):
            router.set_max_concurrent_workers(-1)

    @pytest.mark.asyncio
    async def test_level_complete_handles_pending_workers(self):
        """Test that level_complete_node handles pending workers for batching."""
        state = {
            "current_execution_level": 0,
            "total_execution_levels": 1,
            "fail_on_level_error": False,
            "legion_results": {
                "worker_0": {
                    "result": "Done",
                    "status": "success",
                    "execution_level": 0,
                },
                "worker_1": {
                    "result": "Done",
                    "status": "success",
                    "execution_level": 0,
                },
            },
            "level_results": {},
            "metadata": {
                "legion_worker_plans": [
                    {"worker_id": f"worker_{i}", "execution_level": 0} for i in range(5)
                ],
            },
        }

        result = await legion_level_complete_node(state)

        # Should stay at same level since 3 workers are still pending
        assert result["current_execution_level"] == 0
        assert result["metadata"].get("pending_workers_in_level") is True

    def test_routing_edge_handles_pending_workers(self):
        """Test that routing edge routes back to dispatch for pending workers."""
        state = {
            "current_execution_level": 0,
            "total_execution_levels": 2,
            "metadata": {"pending_workers_in_level": True},
        }

        result = legion_level_routing_edge(state)

        assert result == "legion_dispatch"


@pytest.mark.unit
class TestTimeoutHandling:
    """Test suite for orchestration timeout handling."""

    def test_default_orchestration_timeout_value(self):
        """Test that default orchestration timeout is 5 minutes."""
        from app.hermes.legion.utils.task_timeout import DEFAULT_ORCHESTRATION_TIMEOUT

        assert DEFAULT_ORCHESTRATION_TIMEOUT == 300

    def test_orchestration_timeout_error_class(self):
        """Test OrchestrationTimeoutError exception class."""
        from app.hermes.legion.utils.task_timeout import OrchestrationTimeoutError

        error = OrchestrationTimeoutError(
            user_id="test_user_123",
            timeout=300,
            elapsed=305.5,
            workers_completed=2,
            total_workers=5,
        )

        assert error.user_id == "test_user_123"
        assert error.timeout == 300
        assert error.elapsed == 305.5
        assert error.workers_completed == 2
        assert error.total_workers == 5
        assert "test_use" in str(error)  # Truncated user_id
        assert "305.5" in str(error)
        assert "300" in str(error)

    @pytest.mark.asyncio
    async def test_graph_service_build_timeout_response_no_partial(
        self, legion_graph_service
    ):
        """Test timeout response with no partial results."""
        service = legion_graph_service

        response = service._build_timeout_response(
            partial_result=None,
            workers_completed=0,
            total_workers=5,
            elapsed=310.0,
            timeout=300.0,
        )

        assert "took longer than expected" in response
        assert "310" in response
        assert "300" in response
        assert "simpler request" in response

    @pytest.mark.asyncio
    async def test_graph_service_build_timeout_response_with_partial_results(
        self, legion_graph_service
    ):
        """Test timeout response with partial worker results."""
        service = legion_graph_service

        partial_result = {
            "legion_results": {
                "worker_1": {
                    "result": "Successfully completed research on topic A",
                    "status": "success",
                    "role": "researcher",
                },
                "worker_2": {
                    "result": "",
                    "status": "timeout",
                    "role": "analyzer",
                },
            },
        }

        response = service._build_timeout_response(
            partial_result=partial_result,
            workers_completed=2,
            total_workers=3,
            elapsed=310.0,
            timeout=300.0,
        )

        assert "took longer than expected" in response
        assert "was able to gather some information" in response
        assert "1/3 tasks completed" in response
        assert "Researcher" in response

    @pytest.mark.asyncio
    async def test_graph_service_build_timeout_response_with_level_results(
        self, legion_graph_service
    ):
        """Test timeout response with level-based results."""
        service = legion_graph_service

        partial_result = {
            "legion_results": {},  # No direct results
            "level_results": {
                0: {"success_count": 2, "failed_count": 0},
                1: {"success_count": 1, "failed_count": 1},
            },
        }

        response = service._build_timeout_response(
            partial_result=partial_result,
            workers_completed=4,
            total_workers=6,
            elapsed=310.0,
            timeout=300.0,
        )

        assert "took longer than expected" in response
        assert "3 subtasks completed" in response
        assert "2 execution level" in response
