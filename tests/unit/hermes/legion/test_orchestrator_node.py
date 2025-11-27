"""Unit tests for orchestrator node."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from app.hermes.legion.intelligence.routing_intelligence import (
    ConversationPhase,
    RoutingAction,
    RoutingDecision,
)
from app.hermes.legion.nodes.graph_nodes import orchestrator_node
from app.hermes.legion.state.graph_state import GraphDecision


@pytest.fixture
def mock_state():
    """Create basic mock state."""
    return {
        "messages": [{"role": "user", "content": "test message"}],
        "decision_rationale": [],
        "task_ledger": {},
        "current_agent_id": None,
        "current_task_id": None,
        "metadata": {},
    }


@pytest.mark.asyncio
@patch("app.hermes.legion.intelligence.routing_service.RoutingIntelligence")
async def test_simple_response(MockRoutingIntelligence, mock_state):
    """Test simple response routing."""
    # Setup routing decision
    mock_routing = MockRoutingIntelligence.return_value
    mock_decision = RoutingDecision(
        action=RoutingAction.SIMPLE_RESPONSE,
        reasoning="Simple greeting",
        confidence=0.9,
        requires_agents=False,
        conversation_type="casual",
        complexity_estimate=0.1,
        user_goal="greet",
        conversation_phase=ConversationPhase.INITIATING,
    )
    mock_routing.analyze = AsyncMock(return_value=mock_decision)

    # Execute
    result = await orchestrator_node(mock_state)

    # Verify
    assert result["next_action"] == GraphDecision.COMPLETE.value
    assert result["decision_rationale"][-1]["decisions"]["action"] == "simple_response"


@pytest.mark.asyncio
@patch("app.hermes.legion.intelligence.routing_service.RoutingIntelligence")
async def test_gather_info(MockRoutingIntelligence, mock_state):
    """Test gather info routing."""
    # Setup routing decision
    mock_routing = MockRoutingIntelligence.return_value
    mock_decision = RoutingDecision(
        action=RoutingAction.GATHER_INFO,
        reasoning="Need more details",
        confidence=0.8,
        requires_agents=False,
        conversation_type="clarification",
        complexity_estimate=0.2,
        user_goal="clarify",
        conversation_phase=ConversationPhase.GATHERING_INFO,
    )
    mock_routing.analyze = AsyncMock(return_value=mock_decision)

    # Execute
    result = await orchestrator_node(mock_state)

    # Verify
    assert result["next_action"] == GraphDecision.GATHER_INFO.value
    assert result["decision_rationale"][-1]["decisions"]["action"] == "gather_info"


@pytest.mark.asyncio
@patch("app.hermes.legion.intelligence.routing_service.RoutingIntelligence")
@patch("app.hermes.legion.parallel.task_decomposer.ParallelTaskDecomposer")
async def test_multi_agent_orchestration(
    MockDecomposer, MockRoutingIntelligence, mock_state
):
    """Test multi-agent orchestration routing."""
    # Setup routing decision
    mock_routing = MockRoutingIntelligence.return_value
    mock_decision = RoutingDecision(
        action=RoutingAction.ORCHESTRATE,
        reasoning="Complex task",
        confidence=0.9,
        requires_agents=True,
        conversation_type="complex_task",
        complexity_estimate=0.8,
        user_goal="research_and_analyze",
        conversation_phase=ConversationPhase.INITIATING,
    )
    mock_routing.analyze = AsyncMock(return_value=mock_decision)

    # Setup decomposer to confirm multi-agent
    MockDecomposer.return_value.is_multi_agent_task.return_value = True

    # Execute
    result = await orchestrator_node(mock_state)

    # Verify
    assert result["next_action"] == "legion_orchestrate"
    assert result["decision_rationale"][-1]["decisions"]["action"] == "orchestrate"


@pytest.mark.asyncio
@patch("app.hermes.legion.intelligence.routing_service.RoutingIntelligence")
@patch("app.hermes.legion.parallel.task_decomposer.ParallelTaskDecomposer")
@patch("app.hermes.legion.nodes.graph_nodes.TaskIdentifier")
@patch("app.hermes.legion.nodes.graph_nodes.AgentFactory")
async def test_single_agent_orchestration(
    MockAgentFactory,
    MockTaskIdentifier,
    MockDecomposer,
    MockRoutingIntelligence,
    mock_state,
):
    """Test single agent orchestration routing."""
    # Setup routing decision
    mock_routing = MockRoutingIntelligence.return_value
    mock_decision = RoutingDecision(
        action=RoutingAction.ORCHESTRATE,
        reasoning="Research task",
        confidence=0.9,
        requires_agents=True,
        conversation_type="research",
        complexity_estimate=0.5,
        user_goal="research",
        conversation_phase=ConversationPhase.INITIATING,
    )
    mock_routing.analyze = AsyncMock(return_value=mock_decision)

    # Setup decomposer to deny multi-agent
    MockDecomposer.return_value.is_multi_agent_task.return_value = False

    # Setup task identifier
    MockTaskIdentifier.return_value.identify_task_type.return_value = "research"

    # Setup agent factory
    mock_agent = Mock()
    mock_agent.task_types = ["research"]
    mock_agent.identify_required_info.return_value = {}
    mock_agent_info = Mock()
    mock_agent_info.agent_id = "agent_1"

    MockAgentFactory.create_agent_from_task.return_value = (mock_agent, mock_agent_info)

    # Execute
    result = await orchestrator_node(mock_state)

    # Verify
    assert result["next_action"] == GraphDecision.EXECUTE_AGENT.value
    assert result["current_agent_id"] == "agent_1"


@pytest.mark.asyncio
@patch("app.hermes.legion.intelligence.routing_service.RoutingIntelligence")
async def test_topic_change_cancellation(MockRoutingIntelligence, mock_state):
    """Test topic change causing cancellation."""
    # Setup state with active task
    mock_state["current_task_id"] = "task_1"

    # Setup routing decision with topic change
    mock_routing = MockRoutingIntelligence.return_value
    mock_decision = RoutingDecision(
        action=RoutingAction.SIMPLE_RESPONSE,
        reasoning="Changing topic",
        confidence=0.9,
        requires_agents=False,
        conversation_type="topic_change",
        complexity_estimate=0.1,
        user_goal="change_topic",
        conversation_phase=ConversationPhase.TOPIC_SHIFTING,
        topic_change_detected=True,
        should_abandon_current_task=True,
        previous_topic_description="Old topic",
        new_topic_description="New topic",
    )
    mock_routing.analyze = AsyncMock(return_value=mock_decision)

    # Execute
    result = await orchestrator_node(mock_state)

    # Verify
    assert result["next_action"] == GraphDecision.REPLAN.value
    assert result["messages"][-1]["metadata"]["topic_change"] is True
    assert "Topic change" in result["decision_rationale"][-1]["reasoning"]["action"]
