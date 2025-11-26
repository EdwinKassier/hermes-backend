"""Integration tests for Legion Council flow."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from app.hermes.legion.nodes.legion_orchestrator import (
    LegionWorkerState,
    legion_orchestrator_node,
    legion_synthesis_node,
    legion_worker_node,
)
from app.hermes.legion.utils.resilience import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
)


@pytest.fixture
def mock_gemini_service():
    service = MagicMock()

    # Patch all locations where get_gemini_service is used
    with (
        patch(
            "app.hermes.legion.nodes.legion_orchestrator.get_gemini_service",
            return_value=service,
        ),
        patch(
            "app.hermes.legion.strategies.council.get_gemini_service",
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
    # Patch the source definition since it's imported locally in functions
    with patch("app.hermes.legion.utils.resilience.get_llm_circuit_breaker") as mock:
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=1)
        mock.return_value = cb
        yield cb


async def async_return(val):
    return val


async def async_raise(ex):
    raise ex


@pytest.mark.asyncio
async def test_council_flow_integration(mock_gemini_service, mock_circuit_breaker):
    """Test full council flow from creation to execution."""
    # Setup mocks
    mock_gemini_service.generate_gemini_response.side_effect = [
        # 1. Persona generation response (CouncilStrategy)
        """{
            "personas": [
                {"name": "p1", "description": "d1", "perspective": "q1"},
                {"name": "p2", "description": "d2", "perspective": "q2"},
                {"name": "p3", "description": "d3", "perspective": "q3"}
            ]
        }""",
        # 2. Tool allocation response (legion_worker_node -> ToolAllocator) - for worker 1
        """{ "selected_tools": [] }""",
        # 3. Tool allocation response - for worker 2
        """{ "selected_tools": [] }""",
        # 4. Tool allocation response - for worker 3
        """{ "selected_tools": [] }""",
        # 5. Council synthesis
        "Final synthesis response",
    ]

    # Mock RAG response for agents (if they use RAG, or execute_task)
    # Note: legion_worker_node calls agent.execute_task_async
    # We need to mock the agent execution or the LLM call inside it.
    # Assuming AgentFactory creates agents that use GeminiService.

    # We'll mock the agent execution to return a string directly to simplify
    with patch(
        "app.hermes.legion.agents.factory.AgentFactory.create_agent"
    ) as mock_create_agent:
        mock_agent = MagicMock()
        mock_agent.execute_task_async = MagicMock(
            side_effect=[
                async_return("Response from p1"),
                async_return("Response from p2"),
                async_return("Response from p3"),
            ]
        )
        mock_create_agent.return_value = mock_agent

        # 1. Execute Legion Orchestrator (Council Strategy)
        state = {
            "messages": [{"content": "Test question", "role": "user"}],
            "user_id": "test_user",
            "persona": "hermes",
            "legion_strategy": "council",
            "metadata": {},
            "collected_info": {},
        }

        # Run orchestrator
        orchestrator_result = await legion_orchestrator_node(state)

        # Verify workers generated
        assert orchestrator_result["legion_strategy"] == "council"
        workers = orchestrator_result["metadata"]["legion_worker_plans"]
        assert len(workers) == 3
        assert workers[0]["role"] == "p1"

        # 2. Execute Workers (Simulate parallel execution)
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
            # Merge results manually as LangGraph would
            worker_results.update(result["legion_results"])

        # Verify worker results
        assert len(worker_results) == 3
        assert worker_results["council_p1"]["result"] == "Response from p1"

        # 3. Synthesis
        state["legion_results"] = worker_results
        synthesis_result = await legion_synthesis_node(state)

        # Verify synthesis
        assert synthesis_result["next_action"] == "complete"
        assert synthesis_result["messages"][-1]["content"] == "Final synthesis response"


def test_circuit_breaker_behavior():
    """Test circuit breaker opens after failures."""
    cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)
    func = MagicMock()
    func.side_effect = Exception("Fail")

    # 1. First failure
    with pytest.raises(Exception):
        cb.call(func)
    assert cb.state == CircuitState.CLOSED

    # 2. Second failure (threshold reached)
    with pytest.raises(Exception):
        cb.call(func)
    assert cb.state == CircuitState.OPEN

    # 3. Circuit open - fail fast
    with pytest.raises(CircuitBreakerError):
        cb.call(func)

    # 4. Wait for timeout
    import time

    time.sleep(0.2)

    # 5. Half open
    assert cb.state == CircuitState.HALF_OPEN

    # 6. Success closes circuit
    func.side_effect = None
    cb.call(func)
    assert cb.state == CircuitState.CLOSED
