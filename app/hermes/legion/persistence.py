import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

logger = logging.getLogger(__name__)


class LegionPersistence:
    """
    Manages async persistence for Legion graph state.

    Wraps AsyncSqliteSaver to provide connection management and
    context managers for clean resource handling.
    """

    def __init__(self, db_path: str = "legion_state.db"):
        """
        Initialize persistence manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None
        self._saver: Optional[AsyncSqliteSaver] = None

    @asynccontextmanager
    async def get_checkpointer(self) -> AsyncIterator[AsyncSqliteSaver]:
        """
        Async context manager to get a configured checkpointer.

        Yields:
            AsyncSqliteSaver instance ready for use
        """
        if self.db_path == ":memory:":
            from langgraph.checkpoint.memory import MemorySaver

            yield MemorySaver()
            return

        try:
            async with aiosqlite.connect(self.db_path) as conn:
                # Enable WAL mode for better concurrency
                await conn.execute("PRAGMA journal_mode=WAL")
                await conn.execute("PRAGMA synchronous=NORMAL")

                saver = AsyncSqliteSaver(conn)

                # Initialize tables if needed (AsyncSqliteSaver does this, but good to be explicit if needed)
                # Note: AsyncSqliteSaver.from_conn_string is a factory, but we pass conn directly to constructor

                yield saver

        except Exception as e:
            logger.error(f"Failed to initialize async persistence: {e}")
            # Fallback to memory saver in case of DB error
            from langgraph.checkpoint.memory import MemorySaver

            logger.warning("Falling back to MemorySaver due to persistence error")
            yield MemorySaver()

    async def cleanup(self):
        """Cleanup resources (if any persistent connections are kept)."""
        pass
