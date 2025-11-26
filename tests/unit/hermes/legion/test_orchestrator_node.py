from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest

from app.hermes.legion.nodes.graph_nodes import orchestrator_node
from app.hermes.legion.state.graph_state import (
    GraphDecision,
    OrchestratorState,
    TaskStatus,
)


@pytest.fixture
def mock_state():
    return {
        "messages": [{"role": "user", "content": "test message"}],
        "decision_rationale": [],
        "task_ledger": {},
        "current_agent_id": None,
        "current_task_id": None,
    }


@pytest.mark.asyncio
@patch("app.hermes.legion.nodes.graph_nodes.IntentDetector")
async def test_cancellation(MockIntentDetector, mock_state):
    # Setup
    mock_detector = MockIntentDetector.return_value
    mock_detector.is_cancellation_intent.return_value = True

    # Execute
    result = await orchestrator_node(mock_state)

    # Verify
    assert result["next_action"] == GraphDecision.COMPLETE.value
    assert result["messages"][-1]["metadata"]["cancelled"] is True
    assert (
        "User requested cancellation"
        in result["decision_rationale"][-1]["reasoning"]["action"]
    )


@pytest.mark.asyncio
@patch("app.hermes.legion.nodes.graph_nodes.IntentDetector")
async def test_active_agent(MockIntentDetector, mock_state):
    # Setup
    MockIntentDetector.return_value.is_cancellation_intent.return_value = False
    mock_state["current_agent_id"] = "agent_123"

    # Execute
    result = await orchestrator_node(mock_state)

    # Verify
    assert result["next_action"] == GraphDecision.GATHER_INFO.value
    assert result["decision_rationale"][-1]["analysis"]["has_active_agent"] is True


@pytest.mark.asyncio
@patch("app.hermes.legion.nodes.graph_nodes.IntentDetector")
@patch("app.hermes.legion.parallel.task_decomposer.ParallelTaskDecomposer")
async def test_multi_agent(MockDecomposer, MockIntentDetector, mock_state):
    # Setup
    MockIntentDetector.return_value.is_cancellation_intent.return_value = False
    MockDecomposer.return_value.is_multi_agent_task.return_value = True

    # Execute
    result = await orchestrator_node(mock_state)

    # Verify
    assert result["next_action"] == "legion_orchestrate"
    assert (
        result["decision_rationale"][-1]["analysis"]["multi_agent_task_detected"]
        is True
    )


@pytest.mark.asyncio
@patch("app.hermes.legion.nodes.graph_nodes.IntentDetector")
@patch("app.hermes.legion.parallel.task_decomposer.ParallelTaskDecomposer")
@patch("app.hermes.legion.nodes.graph_nodes.TaskIdentifier")
async def test_general_conversation(
    MockTaskIdentifier, MockDecomposer, MockIntentDetector, mock_state
):
    # Setup
    MockIntentDetector.return_value.is_cancellation_intent.return_value = False
    MockDecomposer.return_value.is_multi_agent_task.return_value = False
    MockTaskIdentifier.return_value.identify_task_type.return_value = None

    # Execute
    result = await orchestrator_node(mock_state)

    # Verify
    assert result["next_action"] == GraphDecision.COMPLETE.value
    assert result["decision_rationale"][-1]["decisions"]["agent_needed"] is False


@pytest.mark.asyncio
@patch("app.hermes.legion.nodes.graph_nodes.IntentDetector")
@patch("app.hermes.legion.parallel.task_decomposer.ParallelTaskDecomposer")
@patch("app.hermes.legion.nodes.graph_nodes.TaskIdentifier")
@patch("app.hermes.legion.nodes.graph_nodes.ToolAllocator")
@patch("app.hermes.legion.nodes.graph_nodes.AgentFactory")
async def test_new_task_creation(
    MockAgentFactory,
    MockToolAllocator,
    MockTaskIdentifier,
    MockDecomposer,
    MockIntentDetector,
    mock_state,
):
    # Setup
    MockIntentDetector.return_value.is_cancellation_intent.return_value = False
    MockDecomposer.return_value.is_multi_agent_task.return_value = False
    MockTaskIdentifier.return_value.identify_task_type.return_value = "research"

    mock_agent = Mock()
    mock_agent.task_types = ["research"]
    mock_agent.identify_required_info.return_value = {}  # No extra info needed

    mock_agent_info = Mock()
    mock_agent_info.agent_id = "new_agent_1"
    mock_agent_info.agent_type = "research_agent"

    MockAgentFactory.create_agent_from_task.return_value = (mock_agent, mock_agent_info)

    # Execute
    result = await orchestrator_node(mock_state)

    # Verify
    assert result["next_action"] == GraphDecision.EXECUTE_AGENT.value
    assert result["current_agent_id"] == "new_agent_1"
    assert result["decision_rationale"][-1]["decisions"]["agent_created"] is True
