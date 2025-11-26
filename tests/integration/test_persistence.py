import os
import sqlite3

import pytest

from app.hermes.legion.graph_service import LegionGraphService
from app.hermes.legion.state.graph_state import Message


@pytest.fixture(autouse=True)
def reset_singleton():
    from app.hermes.legion.nodes import orchestration_graph

    orchestration_graph._orchestration_graph = None
    yield
    orchestration_graph._orchestration_graph = None


@pytest.fixture
def temp_db_path(tmp_path):
    db_file = tmp_path / "test_conversations.db"
    return str(db_file)


def test_persistence_across_restarts(temp_db_path):
    # 1. Start service and add a message
    service1 = LegionGraphService(checkpoint_db_path=temp_db_path)

    # Manually inject a message into the state (simulating a conversation)
    # Since we can't easily run the full graph without mocking Gemini,
    # we'll use the checkpointer directly to save state if possible,
    # or just rely on the fact that graph initialization creates the tables.

    # Actually, let's try to run a simple interaction if possible, or just check if the graph uses the DB.
    # A better test is to check if the checkpointer is correctly initialized with the DB connection.

    graph1 = service1.graph

    # Verify checkpointer is SqliteSaver
    from langgraph.checkpoint.sqlite import SqliteSaver

    assert isinstance(graph1.checkpointer, SqliteSaver)

    # Verify DB file exists
    assert os.path.exists(temp_db_path)

    # 2. Simulate "restart" by creating new service instance with same DB
    service2 = LegionGraphService(checkpoint_db_path=temp_db_path)
    graph2 = service2.graph

    # Verify it's using the same DB
    assert isinstance(graph2.checkpointer, SqliteSaver)

    # 3. Verify we can write and read from the shared DB

    # We need to invoke the graph to trigger the checkpointer to save state/schema
    # We'll patch the orchestrator_node to return a simple state and end
    from unittest.mock import patch

    from app.hermes.legion.state.graph_state import GraphDecision

    with patch(
        "app.hermes.legion.nodes.orchestration_graph.orchestrator_node"
    ) as mock_node:
        mock_node.return_value = {
            "messages": [],
            "next_action": GraphDecision.COMPLETE.value,
        }

        # Invoke graph
        config = {"configurable": {"thread_id": "test-thread"}}
        initial_state = {
            "messages": [{"role": "user", "content": "hi"}],
            "user_id": "test_user",
            "persona": "hermes",
        }
        graph1.invoke(initial_state, config=config)

    # Verify DB file exists and has size > 0
    assert os.path.exists(temp_db_path)

    # Check if we can connect to it
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    # Check if tables exist (checkpoints)
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints';"
    )
    assert cursor.fetchone() is not None

    # Verify we can read the checkpoint from the second service instance
    # (which shares the DB)
    checkpoint = graph2.get_state(config)
    assert checkpoint is not None
    # We should have some state
    assert len(checkpoint.values) > 0

    conn.close()


def test_memory_fallback():
    # Verify fallback to MemorySaver
    service = LegionGraphService(checkpoint_db_path=":memory:")
    graph = service.graph

    from langgraph.checkpoint.memory import MemorySaver

    assert isinstance(graph.checkpointer, MemorySaver)
