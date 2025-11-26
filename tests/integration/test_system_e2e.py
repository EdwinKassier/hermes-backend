from unittest.mock import Mock, patch

import pytest

from app.hermes.legion.graph_service import LegionGraphService
from app.hermes.legion.state.graph_state import GraphDecision


@pytest.fixture(autouse=True)
def reset_singleton():
    from app.hermes.legion.nodes import orchestration_graph

    orchestration_graph._orchestration_graph = None
    yield
    orchestration_graph._orchestration_graph = None


@patch("app.hermes.legion.graph_service.get_gemini_service")
@patch("app.hermes.legion.nodes.graph_nodes.get_gemini_service")
@patch("app.hermes.legion.agents.research_agent.get_gemini_service")
@patch("app.hermes.legion.utils.tool_allocator.get_gemini_service")
@patch("app.hermes.legion.orchestrator.get_gemini_service")
def test_full_conversation_flow(
    mock_orch_service,
    mock_allocator_service,
    mock_agent_service,
    mock_node_service,
    mock_service,
):
    # Setup mocks
    mock_gemini = Mock()
    mock_service.return_value = mock_gemini
    mock_node_service.return_value = mock_gemini
    mock_agent_service.return_value = mock_gemini
    mock_allocator_service.return_value = mock_gemini
    mock_orch_service.return_value = mock_gemini

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

    # Initialize service
    service = LegionGraphService(checkpoint_db_path=":memory:")
    graph = service.graph

    # --- Single Turn: Simple greeting ---
    config = {"configurable": {"thread_id": "e2e-test"}}
    initial_state = {
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "test_user",
        "persona": "hermes",
    }

    # Run graph
    result = graph.invoke(initial_state, config=config)

    # Verify we got a response
    assert "messages" in result
    assert len(result["messages"]) > 0

    # Verify the last message is from assistant
    last_message = result["messages"][-1]
    assert last_message["role"] == "assistant"
    assert len(last_message["content"]) > 0

    # Verify state was saved (checkpointer works)
    state_snapshot = graph.get_state(config)
    assert state_snapshot is not None
    assert len(state_snapshot.values["messages"]) > 0
