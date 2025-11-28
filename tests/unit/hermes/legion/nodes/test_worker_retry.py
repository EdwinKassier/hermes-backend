"""Tests for Legion worker retry logic."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.hermes.legion.nodes.legion_orchestrator import (
    LegionWorkerState,
    legion_worker_node,
)


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, should_fail=False, fail_count=0):
        self.should_fail = should_fail
        self.fail_count = fail_count
        self.call_count = 0

    async def execute_task_async(self, state):
        self.call_count += 1
        if self.should_fail and self.call_count <= self.fail_count:
            raise RuntimeError(f"Simulated failure {self.call_count}")
        return "Task completed successfully"


@pytest.fixture
def worker_state():
    """Create a basic worker state for testing."""
    return LegionWorkerState(
        worker_id="test_worker_1",
        role="researcher",
        task_description="Research the topic",
        tools=["web_search"],
        user_id="test_user",
        persona="hermes",
        context={},
        execution_level=0,
        max_retries=2,
        retry_delay_seconds=0.01,  # Short delay for tests
    )


class TestWorkerRetryLogic:
    """Tests for worker retry behavior."""

    @pytest.mark.asyncio
    async def test_successful_execution_no_retry(self, worker_state):
        """Successful execution completes without retry."""
        mock_agent = MockAgent(should_fail=False)

        with patch(
            "app.hermes.legion.utils.tool_registry.get_tool_registry"
        ) as mock_registry:
            mock_registry.return_value.get_tools.return_value = []

            with patch(
                "app.hermes.legion.nodes.legion_orchestrator.AgentFactory.create_agent"
            ) as mock_factory:
                mock_factory.return_value = mock_agent

                result = await legion_worker_node(worker_state)

        assert result["legion_results"]["test_worker_1"]["status"] == "success"
        assert result["legion_results"]["test_worker_1"]["attempts"] == 1
        assert mock_agent.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_transient_failure(self, worker_state):
        """Worker retries on transient failure."""
        # Fail first attempt, succeed on second
        mock_agent = MockAgent(should_fail=True, fail_count=1)

        with patch(
            "app.hermes.legion.utils.tool_registry.get_tool_registry"
        ) as mock_registry:
            mock_registry.return_value.get_tools.return_value = []

            with patch(
                "app.hermes.legion.nodes.legion_orchestrator.AgentFactory.create_agent"
            ) as mock_factory:
                mock_factory.return_value = mock_agent

                result = await legion_worker_node(worker_state)

        assert result["legion_results"]["test_worker_1"]["status"] == "success"
        assert result["legion_results"]["test_worker_1"]["attempts"] == 2
        assert mock_agent.call_count == 2

    @pytest.mark.asyncio
    async def test_fails_after_max_retries(self, worker_state):
        """Worker fails after exhausting retries."""
        # Always fail
        mock_agent = MockAgent(should_fail=True, fail_count=999)

        with patch(
            "app.hermes.legion.utils.tool_registry.get_tool_registry"
        ) as mock_registry:
            mock_registry.return_value.get_tools.return_value = []

            with patch(
                "app.hermes.legion.nodes.legion_orchestrator.AgentFactory.create_agent"
            ) as mock_factory:
                mock_factory.return_value = mock_agent

                result = await legion_worker_node(worker_state)

        # max_retries=2 means 3 total attempts (initial + 2 retries)
        assert result["legion_results"]["test_worker_1"]["status"] == "failed"
        assert result["legion_results"]["test_worker_1"]["attempts"] == 3
        assert mock_agent.call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_timeout(self, worker_state):
        """Timeouts are not retried."""
        from app.hermes.legion.utils.task_timeout import TaskTimeoutError

        with patch(
            "app.hermes.legion.utils.tool_registry.get_tool_registry"
        ) as mock_registry:
            mock_registry.return_value.get_tools.return_value = []

            with patch(
                "app.hermes.legion.utils.task_timeout.with_timeout"
            ) as mock_timeout:
                mock_timeout.side_effect = TaskTimeoutError("test_worker_1", 30.0, 30.0)

                with patch(
                    "app.hermes.legion.nodes.legion_orchestrator.AgentFactory.create_agent"
                ):
                    result = await legion_worker_node(worker_state)

        assert result["legion_results"]["test_worker_1"]["status"] == "timeout"
        # Only 1 attempt for timeout (no retry)
        assert result["legion_results"]["test_worker_1"]["attempts"] == 1

    @pytest.mark.asyncio
    async def test_custom_retry_config(self, worker_state):
        """Custom retry configuration is respected."""
        worker_state["max_retries"] = 5
        worker_state["retry_delay_seconds"] = 0.001

        # Fail 4 times, succeed on 5th
        mock_agent = MockAgent(should_fail=True, fail_count=4)

        with patch(
            "app.hermes.legion.utils.tool_registry.get_tool_registry"
        ) as mock_registry:
            mock_registry.return_value.get_tools.return_value = []

            with patch(
                "app.hermes.legion.nodes.legion_orchestrator.AgentFactory.create_agent"
            ) as mock_factory:
                mock_factory.return_value = mock_agent

                result = await legion_worker_node(worker_state)

        assert result["legion_results"]["test_worker_1"]["status"] == "success"
        assert result["legion_results"]["test_worker_1"]["attempts"] == 5

    @pytest.mark.asyncio
    async def test_zero_retries(self, worker_state):
        """Zero retries means only one attempt."""
        worker_state["max_retries"] = 0

        mock_agent = MockAgent(should_fail=True, fail_count=999)

        with patch(
            "app.hermes.legion.utils.tool_registry.get_tool_registry"
        ) as mock_registry:
            mock_registry.return_value.get_tools.return_value = []

            with patch(
                "app.hermes.legion.nodes.legion_orchestrator.AgentFactory.create_agent"
            ) as mock_factory:
                mock_factory.return_value = mock_agent

                result = await legion_worker_node(worker_state)

        assert result["legion_results"]["test_worker_1"]["status"] == "failed"
        assert result["legion_results"]["test_worker_1"]["attempts"] == 1
        assert mock_agent.call_count == 1

    @pytest.mark.asyncio
    async def test_exponential_backoff(self, worker_state):
        """Verify exponential backoff timing."""
        worker_state["max_retries"] = 2
        worker_state["retry_delay_seconds"] = 0.1

        mock_agent = MockAgent(should_fail=True, fail_count=2)
        sleep_calls = []

        original_sleep = asyncio.sleep

        async def mock_sleep(delay):
            sleep_calls.append(delay)
            await original_sleep(0.001)  # Minimal actual delay

        with patch(
            "app.hermes.legion.utils.tool_registry.get_tool_registry"
        ) as mock_registry:
            mock_registry.return_value.get_tools.return_value = []

            with patch(
                "app.hermes.legion.nodes.legion_orchestrator.AgentFactory.create_agent"
            ) as mock_factory:
                mock_factory.return_value = mock_agent

                with patch("asyncio.sleep", mock_sleep):
                    result = await legion_worker_node(worker_state)

        # Should succeed on 3rd attempt
        assert result["legion_results"]["test_worker_1"]["status"] == "success"

        # Verify exponential backoff: 0.1 * 2^0 = 0.1, 0.1 * 2^1 = 0.2
        assert len(sleep_calls) == 2
        assert sleep_calls[0] == pytest.approx(0.1, rel=0.1)
        assert sleep_calls[1] == pytest.approx(0.2, rel=0.1)

    @pytest.mark.asyncio
    async def test_uses_tool_registry(self, worker_state):
        """Worker uses ToolRegistry for tool lookup."""
        worker_state["tools"] = ["web_search", "database_query"]

        mock_tools = [MagicMock(name="web_search"), MagicMock(name="database_query")]
        mock_agent = MockAgent(should_fail=False)

        with patch(
            "app.hermes.legion.utils.tool_registry.get_tool_registry"
        ) as mock_registry:
            mock_registry.return_value.get_tools.return_value = mock_tools

            with patch(
                "app.hermes.legion.nodes.legion_orchestrator.AgentFactory.create_agent"
            ) as mock_factory:
                mock_factory.return_value = mock_agent

                result = await legion_worker_node(worker_state)

        # Verify registry was called with correct tool names
        mock_registry.return_value.get_tools.assert_called_once_with(
            ["web_search", "database_query"]
        )

    @pytest.mark.asyncio
    async def test_result_includes_execution_level(self, worker_state):
        """Result includes execution level from state."""
        worker_state["execution_level"] = 2

        mock_agent = MockAgent(should_fail=False)

        with patch(
            "app.hermes.legion.utils.tool_registry.get_tool_registry"
        ) as mock_registry:
            mock_registry.return_value.get_tools.return_value = []

            with patch(
                "app.hermes.legion.nodes.legion_orchestrator.AgentFactory.create_agent"
            ) as mock_factory:
                mock_factory.return_value = mock_agent

                result = await legion_worker_node(worker_state)

        assert result["legion_results"]["test_worker_1"]["execution_level"] == 2
