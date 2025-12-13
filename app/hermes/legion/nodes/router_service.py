"""
Consolidated routing service for LangGraph orchestration.

Consolidates routing logic from multiple functions into a single,
maintainable service class.
"""

import logging
from typing import List, Literal, Optional

from langgraph.types import Send

from ..state import GraphDecision, OrchestratorState

logger = logging.getLogger(__name__)

# Default backpressure configuration
DEFAULT_MAX_CONCURRENT_WORKERS = 5  # Maximum workers per dispatch batch
DEFAULT_WORKER_BATCH_SIZE = 5  # Workers to dispatch in each batch


class RouterService:
    """
    Centralized routing service for all graph routing decisions.

    Consolidates routing logic from:
    - route_state() - Main orchestrator routing
    - should_continue() - Conversation continuation logic
    - route_from_info_gathering() - Information gathering routing
    - legion_routing_edge() - Legion worker dispatch

    Features:
    - Backpressure support via configurable max concurrent workers
    - Batch dispatching to prevent resource exhaustion
    """

    # Routing constants
    ROUTE_LEGION_ORCHESTRATE = "legion_orchestrate"
    ROUTE_SYNTHESIZE = "synthesize"

    def __init__(
        self,
        max_concurrent_workers: int = DEFAULT_MAX_CONCURRENT_WORKERS,
    ):
        """
        Initialize RouterService with backpressure configuration.

        Args:
            max_concurrent_workers: Maximum workers to dispatch in a single batch.
                                   Prevents resource exhaustion with large task decompositions.
        """
        self.max_concurrent_workers = max_concurrent_workers

    def route_from_orchestrator(self, state: OrchestratorState) -> str:
        """
        Route from orchestrator node based on next_action.

        Replaces: route_state()

        Args:
            state: Current orchestrator state

        Returns:
            Name of next node to execute
        """
        next_action = state.get("next_action", "")

        logger.debug(f"Routing from orchestrator: next_action={next_action}")

        # Define routing map for all possible actions
        route_map = {
            # Legion routes
            self.ROUTE_LEGION_ORCHESTRATE: "legion_orchestrator",
            self.ROUTE_SYNTHESIZE: "legion_synthesis",
            # Standard routes (using GraphDecision enum)
            GraphDecision.GATHER_INFO.value: "information_gathering",
            GraphDecision.EXECUTE_AGENT.value: "agent_executor",
            GraphDecision.REPLAN.value: "orchestrator",  # Loop back for topic changes/replanning
            GraphDecision.COMPLETE.value: "general_response",
            GraphDecision.ERROR.value: "error_handler",
        }

        # Get route from map
        route = route_map.get(next_action)

        if route is None:
            # Fallback for legacy routes
            if "council" in next_action or "parallel" in next_action:
                logger.info(
                    f"Legacy route '{next_action}' detected, redirecting to legion_orchestrator"
                )
                return "legion_orchestrator"

            logger.warning(
                f"Unknown next_action: '{next_action}', defaulting to general_response"
            )
            return "general_response"

        logger.info(f"Routing to: {route}")
        return route

    def should_continue(
        self, state: OrchestratorState
    ) -> Literal["continue", "replan", "end"]:
        """
        Determine if conversation should continue or end.

        Replaces: should_continue()

        This enables dynamic conversation flow where the graph can:
        - Pause and wait for user response (e.g., after asking a question)
        - Continue to orchestrator for topic changes or follow-ups
        - End the conversation when explicitly complete

        Args:
            state: Current orchestrator state

        Returns:
            "continue" to wait for user, "replan" to re-evaluate, or "end" to terminate
        """
        # Check explicit completion flag
        if state.get("conversation_complete"):
            logger.info("Conversation marked as complete")
            return "end"

        # Check if awaiting user input
        if state.get("awaiting_user_response"):
            logger.info("Awaiting user response - pausing conversation")
            return "continue"

        # Check if we're in an error state
        if state.get("next_action") == GraphDecision.ERROR.value:
            logger.info("Error state - ending conversation")
            return "end"

        # Default: allow continuation
        logger.info("Allowing conversation continuation")
        return "continue"

    def route_from_info_gathering(self, state: OrchestratorState) -> str:
        """
        Route from information gathering node.

        Replaces: route_from_info_gathering()

        Args:
            state: Current orchestrator state

        Returns:
            Next node to execute
        """
        next_action = state.get("next_action", "")

        logger.debug(f"Routing from info gathering: next_action={next_action}")

        if next_action == GraphDecision.GATHER_INFO.value:
            # Still need more info, end turn to wait for user
            logger.info("Still gathering information, pausing for user input")
            return "gather_info"
        elif next_action == GraphDecision.EXECUTE_AGENT.value:
            # Info collected, execute agent
            logger.info("Information gathered, proceeding to agent execution")
            return "execute_agent"
        else:
            # Unexpected state, end to be safe
            logger.warning(
                f"Unexpected action from info gathering: {next_action}, ending turn"
            )
            return "end"

    def route_legion_workers(self, state: OrchestratorState) -> List[Send]:
        """
        Route to Legion workers using Send API, filtered by current execution level.

        Replaces: legion_routing_edge()

        This creates dynamic worker nodes based on the workers in state,
        but ONLY for workers at the current execution level. This enables
        dependency-aware sequential execution between levels while maintaining
        parallel execution within each level.

        Implements backpressure by limiting concurrent workers per batch.
        If more workers are needed than the limit, they are tracked in metadata
        for subsequent dispatch cycles.

        Args:
            state: Current orchestrator state with legion_worker_plans in metadata

        Returns:
            List of Send objects for dynamic worker dispatch (current level only,
            limited by max_concurrent_workers for backpressure)
        """
        from .legion_orchestrator import LegionWorkerState

        workers = state["metadata"].get("legion_worker_plans", [])
        current_level = state.get("current_execution_level", 0)

        if not workers:
            logger.warning("No legion_worker_plans found in state metadata")
            return []

        # Filter workers to only those at the current execution level
        level_workers = [
            w for w in workers if w.get("execution_level", 0) == current_level
        ]

        if not level_workers:
            logger.warning(
                f"No workers found for execution level {current_level}. "
                f"Total workers: {len(workers)}"
            )
            return []

        # Check for already dispatched workers in this level (for batching)
        dispatched_worker_ids = set(
            state.get("metadata", {}).get("dispatched_worker_ids", [])
        )

        # Filter out already dispatched workers
        pending_workers = [
            w for w in level_workers if w["worker_id"] not in dispatched_worker_ids
        ]

        if not pending_workers:
            logger.debug(f"All workers for level {current_level} already dispatched")
            return []

        # Apply backpressure: limit batch size
        batch_workers = pending_workers[: self.max_concurrent_workers]
        remaining_count = len(pending_workers) - len(batch_workers)

        if remaining_count > 0:
            logger.info(
                f"Backpressure: Dispatching {len(batch_workers)} of {len(pending_workers)} "
                f"pending workers for level {current_level}. "
                f"{remaining_count} workers queued for next batch."
            )
        else:
            logger.info(
                f"Dispatching {len(batch_workers)} Legion workers for level {current_level} "
                f"(total workers across all levels: {len(workers)})"
            )

        # Build context with previous level results
        level_results = state.get("level_results", {})
        previous_results = {}
        for level_idx in range(current_level):
            level_data = level_results.get(level_idx, {})
            for worker_id, result_data in level_data.get("workers", {}).items():
                previous_results[worker_id] = result_data.get("result", "")

        # Create Send object for each worker in this batch
        sends = []
        for worker in batch_workers:
            # Merge previous results into context
            context = {
                **state.get("collected_info", {}),
                "previous_level_results": previous_results,
            }

            worker_state = LegionWorkerState(
                worker_id=worker["worker_id"],
                role=worker["role"],
                task_description=worker["task_description"],
                tools=worker["tools"],
                user_id=state["user_id"],
                persona=worker.get("persona", state.get("persona", "legion")),
                context=context,
                execution_level=worker.get("execution_level", 0),
                max_retries=worker.get("max_retries", 2),
                retry_delay_seconds=worker.get("retry_delay_seconds", 1.0),
            )
            sends.append(Send("legion_worker", worker_state))

        return sends

    def get_max_concurrent_workers(self) -> int:
        """Get the current max concurrent workers limit."""
        return self.max_concurrent_workers

    def set_max_concurrent_workers(self, limit: int) -> None:
        """
        Set the max concurrent workers limit.

        Args:
            limit: New limit (must be >= 1)
        """
        if limit < 1:
            raise ValueError("max_concurrent_workers must be at least 1")
        self.max_concurrent_workers = limit
        logger.info(f"Max concurrent workers updated to {limit}")
