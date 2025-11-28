"""LangGraph-based orchestration workflow."""

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from ..state import GraphDecision, OrchestratorState
from .graph_nodes import (
    agent_executor_node,
    error_handler_node,
    general_response_node,
    information_gathering_node,
    orchestrator_node,
)
from .legion_orchestrator import (
    legion_dispatch_node,
    legion_level_complete_node,
    legion_level_routing_edge,
    legion_orchestrator_node,
    legion_routing_edge,
    legion_synthesis_node,
    legion_worker_node,
)
from .router_service import RouterService

logger = logging.getLogger(__name__)


# Routing decision constants (kept for backwards compatibility)
ROUTE_LEGION_ORCHESTRATE = "legion_orchestrate"
ROUTE_SYNTHESIZE = "synthesize"


# Create singleton RouterService instance
_router_service = RouterService()


def route_state(state: OrchestratorState) -> str:
    """
    DEPRECATED: Use RouterService.route_from_orchestrator() instead.

    Kept for backwards compatibility during migration.
    """
    return _router_service.route_from_orchestrator(state)


def route_from_info_gathering(state: OrchestratorState) -> str:
    """
    Route from information gathering node.

    Wrapper for RouterService.route_from_info_gathering().
    """
    return _router_service.route_from_info_gathering(state)


def should_continue(state: OrchestratorState) -> str:
    """

    Args:
        state: Current orchestrator state

    Returns:
        "continue" to wait for user, "replan" to re-evaluate, or "end" to terminate
    """
    # Check explicit flags
    if state.get("conversation_complete"):
        logger.info("Conversation marked as complete")
        return "end"

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


def create_orchestration_graph(checkpointer=None, interrupt_before=None) -> StateGraph:
    """
    Create the LangGraph-based orchestration workflow.

    This implements the Magentic Orchestrator pattern with:
    - Centralized orchestrator node for decision-making
    - Dynamic agent creation and tool allocation
    - Multi-turn information gathering
    - Task ledger for tracking
    - State persistence via checkpointing
    - Human-in-the-loop interrupts for approval workflows

    Args:
        checkpointer: An optional LangGraph checkpointer instance. If None, a MemorySaver will be used.
        interrupt_before: Optional list of node names to interrupt before executing.
                         Common values: ["legion_orchestrator", "agent_executor"]

    Returns:
        Compiled StateGraph ready for execution
    """
    # Create the graph
    workflow = StateGraph(OrchestratorState)

    # Add nodes to graph
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("information_gathering", information_gathering_node)
    workflow.add_node("agent_executor", agent_executor_node)
    workflow.add_node("general_response", general_response_node)
    workflow.add_node("error_handler", error_handler_node)

    # Add unified Legion nodes
    workflow.add_node("legion_orchestrator", legion_orchestrator_node)
    workflow.add_node("legion_dispatch", legion_dispatch_node)
    workflow.add_node("legion_worker", legion_worker_node)
    workflow.add_node("legion_level_complete", legion_level_complete_node)
    workflow.add_node("legion_synthesis", legion_synthesis_node)

    # Set entry point
    workflow.set_entry_point("orchestrator")

    # Add edges using unified routing function

    # From orchestrator, route based on decision
    workflow.add_conditional_edges(
        "orchestrator",
        route_state,
        {
            "legion_orchestrator": "legion_orchestrator",
            "information_gathering": "information_gathering",
            "agent_executor": "agent_executor",
            "general_response": "general_response",
            "error_handler": "error_handler",
        },
    )

    # Legion Orchestrator -> Dynamic Workers (using Send API)
    # This dispatches workers for level 0
    workflow.add_conditional_edges(
        "legion_orchestrator", legion_routing_edge, ["legion_worker"]
    )

    # Legion Dispatch -> Dynamic Workers (for subsequent levels)
    # This is the loop entry point for levels > 0
    workflow.add_conditional_edges(
        "legion_dispatch", legion_routing_edge, ["legion_worker"]
    )

    # Legion Workers -> Level Complete (process results and determine next step)
    workflow.add_edge("legion_worker", "legion_level_complete")

    # Level Complete -> Either dispatch next level or proceed to synthesis
    workflow.add_conditional_edges(
        "legion_level_complete",
        legion_level_routing_edge,
        {
            "legion_dispatch": "legion_dispatch",  # More levels to process
            "legion_synthesis": "legion_synthesis",  # All levels complete
        },
    )

    # Synthesis -> End
    workflow.add_edge("legion_synthesis", END)

    # Add conditional edges from info gathering
    workflow.add_conditional_edges(
        "information_gathering",
        route_from_info_gathering,
        {
            "gather_info": END,  # End turn, wait for user response
            "execute_agent": "agent_executor",
            "end": END,
        },
    )

    # From agent executor -> general response (after execution)
    workflow.add_edge("agent_executor", "general_response")

    # Terminal nodes -> Conditional continuation or END
    # This allows multi-turn conversations and topic changes
    workflow.add_conditional_edges(
        "general_response",
        should_continue,
        {
            "continue": END,  # Pause for next user message
            "replan": "orchestrator",  # Topic change - re-evaluate
            "end": END,  # Explicitly end
        },
    )

    workflow.add_conditional_edges(
        "error_handler",
        should_continue,
        {
            "continue": END,
            "end": END,
        },
    )

    # Compile with checkpointing and optional interrupts
    # Note: For production, use SqliteSaver for persistent storage
    if checkpointer is None:
        checkpointer = MemorySaver()

    # Set default interrupts if none specified
    if interrupt_before is None:
        interrupt_before = []  # Can be enabled by passing ["agent_executor"] etc.

    app = workflow.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)

    logger.info("LangGraph orchestration workflow created successfully")

    return app


def get_orchestration_graph(checkpointer=None, interrupt_before=None):
    """
    Create an orchestration graph with the provided checkpointer.

    Each call creates a fresh graph instance to ensure proper checkpointer
    lifecycle management. This avoids issues with:
    - Stale checkpointer references
    - Race conditions in async environments
    - Testing isolation

    Args:
        checkpointer: Checkpointer instance for state persistence.
                     If None, a MemorySaver will be used (ephemeral storage).
        interrupt_before: Optional list of nodes to interrupt before.

    Returns:
        Compiled StateGraph ready for execution

    Usage:
        # With async persistence (recommended for production)
        async with persistence.get_checkpointer() as checkpointer:
            graph = get_orchestration_graph(checkpointer=checkpointer)
            result = await graph.astream(inputs, config)

        # With MemorySaver for testing
        graph = get_orchestration_graph()
    """
    if checkpointer is None:
        checkpointer = MemorySaver()
        logger.debug("Creating graph with default MemorySaver (ephemeral)")
    else:
        logger.debug("Creating graph with provided checkpointer")

    return create_orchestration_graph(
        checkpointer=checkpointer, interrupt_before=interrupt_before
    )
