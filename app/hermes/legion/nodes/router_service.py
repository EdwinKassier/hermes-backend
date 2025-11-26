"""
Consolidated routing service for LangGraph orchestration.

Consolidates routing logic from multiple functions into a single,
maintainable service class.
"""

import logging
from typing import List, Literal

from langgraph.types import Send

from ..state import GraphDecision, OrchestratorState

logger = logging.getLogger(__name__)


class RouterService:
    """
    Centralized routing service for all graph routing decisions.

    Consolidates routing logic from:
    - route_state() - Main orchestrator routing
    - should_continue() - Conversation continuation logic
    - route_from_info_gathering() - Information gathering routing
    - legion_routing_edge() - Legion worker dispatch
    """

    # Routing constants
    ROUTE_LEGION_ORCHESTRATE = "legion_orchestrate"
    ROUTE_SYNTHESIZE = "synthesize"

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
        Route to Legion workers using Send API.

        Replaces: legion_routing_edge()

        This creates dynamic worker nodes based on the workers in state.
        Each worker becomes a separate graph node that can execute in parallel.

        Args:
            state: Current orchestrator state with legion_worker_plans in metadata

        Returns:
            List of Send objects for dynamic worker dispatch
        """
        from .legion_orchestrator import LegionWorkerState

        workers = state["metadata"].get("legion_worker_plans", [])

        if not workers:
            logger.warning("No legion_worker_plans found in state metadata")
            return []

        logger.info(f"Dispatching {len(workers)} Legion workers")

        # Create Send object for each worker
        # Send API enables dynamic parallelism
        sends = []
        for worker in workers:
            worker_state = LegionWorkerState(
                worker_id=worker["worker_id"],
                role=worker["role"],
                task_description=worker["task_description"],
                tools=worker["tools"],
                user_id=state["user_id"],
                persona=state["persona"],
                context=state.get("collected_info", {}),
            )
            sends.append(Send("legion_worker", worker_state))

        return sends
