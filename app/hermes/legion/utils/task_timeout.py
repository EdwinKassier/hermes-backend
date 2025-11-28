"""
Task Timeout Management for Worker Execution.

This module provides utilities for managing timeouts on long-running tasks
to prevent runaway execution and ensure system responsiveness.

Usage:
    # Simple timeout wrapper
    result = await with_timeout(my_coroutine(), timeout=30, task_id="my_task")

    # Convenience function
    result = await run_with_timeout(my_coroutine(), timeout=30)
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

# Type variable for generic function return type
T = TypeVar("T")

# Default timeout settings
DEFAULT_WORKER_TIMEOUT = 120  # 2 minutes per worker
DEFAULT_LLM_CALL_TIMEOUT = 60  # 1 minute per LLM call
DEFAULT_ORCHESTRATION_TIMEOUT = 300  # 5 minutes for entire orchestration


class TaskTimeoutError(Exception):
    """Raised when a task exceeds its timeout."""

    def __init__(self, task_id: str, timeout: float, elapsed: float):
        self.task_id = task_id
        self.timeout = timeout
        self.elapsed = elapsed
        super().__init__(
            f"Task '{task_id}' timed out after {elapsed:.1f}s (limit: {timeout}s)"
        )


class OrchestrationTimeoutError(Exception):
    """Raised when the entire orchestration exceeds its timeout."""

    def __init__(
        self,
        user_id: str,
        timeout: float,
        elapsed: float,
        workers_completed: int = 0,
        total_workers: int = 0,
    ):
        self.user_id = user_id
        self.timeout = timeout
        self.elapsed = elapsed
        self.workers_completed = workers_completed
        self.total_workers = total_workers
        super().__init__(
            f"Orchestration for user '{user_id[:8]}...' timed out after {elapsed:.1f}s "
            f"(limit: {timeout}s). Completed {workers_completed}/{total_workers} workers."
        )


async def with_timeout(
    coroutine,
    timeout: float,
    task_id: str = "unknown",
    default_value: Any = None,
    raise_on_timeout: bool = True,
) -> Any:
    """
    Execute a coroutine with a timeout.

    Args:
        coroutine: The async function to execute
        timeout: Timeout in seconds
        task_id: Identifier for logging
        default_value: Value to return on timeout (if not raising)
        raise_on_timeout: Whether to raise TaskTimeoutError

    Returns:
        Result of the coroutine or default_value on timeout

    Raises:
        TaskTimeoutError: If timeout exceeded and raise_on_timeout is True
    """
    start = datetime.utcnow()

    try:
        result = await asyncio.wait_for(coroutine, timeout=timeout)
        elapsed = (datetime.utcnow() - start).total_seconds()
        logger.debug(
            "Task '%s' completed in %.2fs (limit: %.0fs)", task_id, elapsed, timeout
        )
        return result

    except asyncio.TimeoutError as exc:
        elapsed = (datetime.utcnow() - start).total_seconds()
        logger.warning(
            "Task '%s' timed out after %.2fs (limit: %.0fs)", task_id, elapsed, timeout
        )

        if raise_on_timeout:
            raise TaskTimeoutError(task_id, timeout, elapsed) from exc

        return default_value


# Convenience function for simple timeout wrapping
async def run_with_timeout(
    coro, timeout: float = DEFAULT_WORKER_TIMEOUT, task_id: str = "task"
) -> Any:
    """
    Simple convenience function to run a coroutine with timeout.

    Args:
        coro: Coroutine to execute
        timeout: Timeout in seconds
        task_id: Identifier for logging

    Returns:
        Result of the coroutine

    Raises:
        TaskTimeoutError: If timeout exceeded
    """
    return await with_timeout(coro, timeout, task_id, raise_on_timeout=True)
