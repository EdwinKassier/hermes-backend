import asyncio
import json
from unittest.mock import MagicMock, Mock, patch

import pytest

from app.hermes.legion.graph_service import LegionGraphService
from app.hermes.legion.nodes.orchestration_graph import get_orchestration_graph
from app.hermes.legion.state.graph_state import GraphDecision


@pytest.fixture(autouse=True)
def reset_singleton():
    from app.hermes.legion.nodes import orchestration_graph

    orchestration_graph._orchestration_graph = None
    yield
    orchestration_graph._orchestration_graph = None


@pytest.mark.asyncio
@patch("app.shared.utils.service_loader.get_llm_service")
async def test_full_system_flow(mock_get_llm):
    """Test full system flow end-to-end."""
    # Setup mocks
    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm

    # Mock LLM responses for different stages
    mock_llm.generate_response.side_effect = [
        # Planner response
        json.dumps({"plan": "Test plan", "agents": ["researcher", "analyst"]}),
        # Researcher response
        "Research results: Some interesting facts.",
        # Analyst response
        "Analysis results: A summary of the facts.",
        # Final response
        "Here is the final answer based on the research and analysis.",
    ]

    # Get the graph directly for testing
    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = MemorySaver()
    graph = get_orchestration_graph(checkpointer=checkpointer)

    # --- Single Turn: Complex query ---
    config = {"configurable": {"thread_id": "e2e-system-test"}}
    initial_state = {
        "messages": [
            {"role": "user", "content": "Tell me about the latest AI advancements."}
        ],
        "user_id": "test_user_system",
        "persona": "hermes",
    }

    # Run graph (async)
    result = await graph.ainvoke(initial_state, config=config)

    # Verify we got a response
    assert "messages" in result
    assert len(result["messages"]) > 0

    # Verify the last message is from assistant
    last_message = result["messages"][-1]
    assert last_message["role"] == "assistant"
    assert len(last_message["content"]) > 0
    assert "final answer" in last_message["content"].lower()

    # Verify state was saved (checkpointer works)
    state_snapshot = await graph.aget_state(config)
    assert state_snapshot is not None
    assert len(state_snapshot.values["messages"]) > 0
    assert (
        state_snapshot.values["current_task"]
        == "Here is the final answer based on the research and analysis."
    )
    assert state_snapshot.values["plan"] == "Test plan"
    assert state_snapshot.values["agents"] == ["researcher", "analyst"]
    assert (
        state_snapshot.values["research_results"]
        == "Research results: Some interesting facts."
    )
    assert (
        state_snapshot.values["analysis_results"]
        == "Analysis results: A summary of the facts."
    )


@patch("app.shared.utils.service_loader.get_gemini_service")
async def test_full_conversation_flow(mock_service):
    # Setup mocks
    mock_gemini = Mock()
    mock_service.return_value = mock_gemini

    # Configure all_tools to be a list
    mock_tool = Mock()
    mock_tool.name = "web_search"
    mock_tool.description = "Search the web"
    mock_gemini.all_tools = [mock_tool]

    # Mock responses for a simple flow
    mock_gemini.generate_gemini_response.side_effect = [
        # 1. Task Identification -> "general" (no agent needed)
        "general",
        # 2. General Response Generation
        "Hello! How can I help you today?",
    ]

    # Get the graph directly for testing
    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = MemorySaver()
    graph = get_orchestration_graph(checkpointer=checkpointer)

    # --- Single Turn: Simple greeting ---
    config = {"configurable": {"thread_id": "e2e-test"}}
    initial_state = {
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "test_user",
        "persona": "hermes",
    }

    # Run graph (async)
    result = await graph.ainvoke(initial_state, config=config)

    # Verify we got a response
    assert "messages" in result
    assert len(result["messages"]) > 0

    # Verify the last message is from assistant
    last_message = result["messages"][-1]
    assert last_message["role"] == "assistant"
    assert len(last_message["content"]) > 0

    # Verify state was saved (checkpointer works)
    state_snapshot = await graph.aget_state(config)
    assert state_snapshot is not None
    assert len(state_snapshot.values["messages"]) > 0
