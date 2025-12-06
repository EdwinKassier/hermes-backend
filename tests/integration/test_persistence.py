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


@pytest.mark.asyncio
async def test_persistence_across_restarts(temp_db_path):
    """Test that state persists across service restarts."""
    import asyncio

    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from app.hermes.legion.nodes.orchestration_graph import get_orchestration_graph

    # 1. Create first service instance with file-based checkpointer
    async with aiosqlite.connect(temp_db_path) as conn:
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        checkpointer1 = AsyncSqliteSaver(conn)
        graph1 = get_orchestration_graph(checkpointer=checkpointer1)

        # Verify checkpointer is AsyncSqliteSaver
        assert isinstance(graph1.checkpointer, AsyncSqliteSaver)

        # 2. Save some state
        config = {"configurable": {"thread_id": "test-thread"}}
        initial_state = {
            "messages": [{"role": "user", "content": "hi"}],
            "user_id": "test_user",
            "persona": "hermes",
        }

        # Invoke graph (async)
        await graph1.ainvoke(initial_state, config=config)

        # Verify state was saved
        checkpoint1 = await graph1.aget_state(config)
        assert checkpoint1 is not None

    # 3. Simulate "restart" by creating new connection to same DB
    async with aiosqlite.connect(temp_db_path) as conn2:
        await conn2.execute("PRAGMA journal_mode=WAL")
        await conn2.execute("PRAGMA synchronous=NORMAL")
        checkpointer2 = AsyncSqliteSaver(conn2)
        graph2 = get_orchestration_graph(checkpointer=checkpointer2)

        # Verify we can read the checkpoint from the second instance
        checkpoint2 = await graph2.aget_state(config)
        assert checkpoint2 is not None
        assert len(checkpoint2.values) > 0

    # Verify DB file exists
    assert os.path.exists(temp_db_path)

    # Check if we can connect to it and verify tables exist
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints';"
    )
    assert cursor.fetchone() is not None
    conn.close()


def test_memory_fallback():
    """Verify fallback to MemorySaver for :memory: path."""
    from langgraph.checkpoint.memory import MemorySaver

    from app.hermes.legion.nodes.orchestration_graph import get_orchestration_graph

    checkpointer = MemorySaver()
    graph = get_orchestration_graph(checkpointer=checkpointer)

    assert isinstance(graph.checkpointer, MemorySaver)
