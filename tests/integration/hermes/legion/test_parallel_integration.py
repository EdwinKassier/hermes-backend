"""Integration tests for Parallel multi-agent flow."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from app.hermes.legion.nodes.legion_orchestrator import (
    LegionWorkerState,
    legion_orchestrator_node,
    legion_synthesis_node,
    legion_worker_node,
)
from app.hermes.legion.utils.resilience import CircuitBreaker


@pytest.fixture
def mock_gemini_service():
    """Mock GeminiService for all LLM calls."""
    service = MagicMock()

    # Patch all locations where get_gemini_service is used
    with (
        patch(
            "app.hermes.legion.nodes.legion_orchestrator.get_gemini_service",
            return_value=service,
        ),
        patch(
            "app.hermes.legion.parallel.task_decomposer.get_gemini_service",
            return_value=service,
        ),
        patch(
            "app.hermes.legion.utils.tool_allocator.get_gemini_service",
            return_value=service,
        ),
        patch(
            "app.hermes.legion.agents.factory.get_gemini_service", return_value=service
        ),
        patch(
            "app.hermes.legion.parallel.result_synthesizer.get_gemini_service",
            return_value=service,
        ),
    ):
        yield service


@pytest.fixture
def mock_circuit_breaker():
    """Mock CircuitBreaker to avoid actual resilience logic in tests."""
    with patch("app.hermes.legion.utils.resilience.get_llm_circuit_breaker") as mock:
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=1)
        mock.return_value = cb
        yield cb


async def async_return(val):
    return val


async def async_raise(ex):
    raise ex


@pytest.mark.asyncio
async def test_parallel_flow_integration(mock_gemini_service, mock_circuit_breaker):
    """Test end-to-end parallel flow with 2 agents."""

    # Configure mocks
    mock_gemini_service.generate_gemini_response.side_effect = [
        # 1. Task decomposition response (ParallelStrategy)
        """{
            "is_multi_agent": true,
            "subtasks": [
                {
                    "agent_type": "research",
                    "description": "Research historical context",
                    "depends_on": []
                },
                {
                    "agent_type": "analysis",
                    "description": "Analyze current trends",
                    "depends_on": []
                }
            ]
        }""",
        # 2. Tool allocation for agent 1 (orchestrator)
        """{"selected_tools": []}""",
        # 3. Tool allocation for agent 2 (orchestrator)
        """{"selected_tools": []}""",
        # 4. Tool allocation for agent 1 (executor)
        """{"selected_tools": []}""",
        # 5. Tool allocation for agent 2 (executor)
        """{"selected_tools": []}""",
        # 6. Synthesis response
        "Synthesized final response combining both agent outputs",
    ]

    # Mock agent execution
    with patch(
        "app.hermes.legion.agents.factory.AgentFactory.create_agent"
    ) as mock_create_agent:
        mock_agent = MagicMock()
        mock_agent.execute_task_async = MagicMock(
            side_effect=[
                async_return("Research agent response: Historical context analysis"),
                async_return("Analysis agent response: Current trend analysis"),
            ]
        )
        mock_create_agent.return_value = mock_agent

        # 1. Legion Orchestrator (Parallel Strategy)
        state = {
            "messages": [
                {
                    "content": "Analyze the historical context and current trends",
                    "role": "user",
                }
            ],
            "user_id": "test_user",
            "persona": "hermes",
            "legion_strategy": "parallel",
            "metadata": {},
            "collected_info": {},
        }

        # Run orchestrator
        orchestrator_result = await legion_orchestrator_node(state)

        # Verify decomposition
        assert orchestrator_result["legion_strategy"] == "parallel"
        workers = orchestrator_result["metadata"]["legion_worker_plans"]
        assert len(workers) == 2
        assert workers[0]["role"] == "research"
        assert workers[1]["role"] == "analysis"

        # 2. Execute Workers
        worker_results = {}

        for worker in workers:
            worker_state = LegionWorkerState(
                worker_id=worker["worker_id"],
                role=worker["role"],
                task_description=worker["task_description"],
                tools=worker["tools"],
                user_id=state["user_id"],
                persona=state["persona"],
                context=state.get("collected_info", {}),
            )

            result = await legion_worker_node(worker_state)
            worker_results.update(result["legion_results"])

        # Verify execution
        assert len(worker_results) == 2

        # 3. Synthesis
        state["legion_results"] = worker_results
        synthesis_result = await legion_synthesis_node(state)

        # Verify synthesis
        assert (
            "Synthesized final response" in synthesis_result["messages"][-1]["content"]
        )


@pytest.mark.asyncio
async def test_parallel_flow_with_partial_failure(
    mock_gemini_service, mock_circuit_breaker
):
    """Test parallel flow handling when one agent fails."""

    # Configure mocks - agent 2 will fail
    mock_gemini_service.generate_gemini_response.side_effect = [
        # Task decomposition
        """{
            "is_multi_agent": true,
            "subtasks": [
                {"agent_type": "research", "description": "Task 1", "depends_on": []},
                {"agent_type": "analysis", "description": "Task 2", "depends_on": []}
            ]
        }""",
        # Tool allocations (orchestrator)
        """{"selected_tools": []}""",
        """{"selected_tools": []}""",
        # Tool allocations (executor)
        """{"selected_tools": []}""",
        """{"selected_tools": []}""",
        # Synthesis
        "Partial synthesis with available results",
    ]

    # Mock agent execution with failure
    with patch(
        "app.hermes.legion.agents.factory.AgentFactory.create_agent"
    ) as mock_create_agent:
        mock_agent = MagicMock()
        mock_agent.execute_task_async = MagicMock(
            side_effect=[
                async_return("Agent 1 success response"),
                async_raise(Exception("Agent 2 execution failed")),
            ]
        )
        mock_create_agent.return_value = mock_agent

        state = {
            "messages": [{"content": "Test query", "role": "user"}],
            "user_id": "test_user",
            "persona": "hermes",
            "legion_strategy": "parallel",
            "metadata": {},
            "collected_info": {},
        }

        # Orchestrate
        orchestrator_result = await legion_orchestrator_node(state)
        workers = orchestrator_result["metadata"]["legion_worker_plans"]

        # Execute
        worker_results = {}
        for worker in workers:
            worker_state = LegionWorkerState(
                worker_id=worker["worker_id"],
                role=worker["role"],
                task_description=worker["task_description"],
                tools=worker["tools"],
                user_id=state["user_id"],
                persona=state["persona"],
                context=state.get("collected_info", {}),
            )

            result = await legion_worker_node(worker_state)
            worker_results.update(result["legion_results"])

        # Verify partial results
        # Should have 2 results (1 success, 1 failure)
        assert len(worker_results) == 2

        # Verify at least one succeeded
        successful = [
            r for r in worker_results.values() if r.get("status") == "success"
        ]
        failed = [r for r in worker_results.values() if r.get("status") == "failed"]
        assert len(successful) >= 1
        assert len(failed) >= 1

        # Synthesis should still work with partial results
        state["legion_results"] = worker_results
        synthesis_result = await legion_synthesis_node(state)
        assert "messages" in synthesis_result
