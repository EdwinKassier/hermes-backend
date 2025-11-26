"""
Unified Legion Orchestration System.

This module implements the "Legion" swarm architecture using LangGraph's `Send` API
for dynamic, parallel agent execution. It unifies previous "Council" and "Parallel"
modes into a single, flexible system.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langgraph.constants import Send

from app.shared.utils.service_loader import get_gemini_service

from ..agents.factory import AgentFactory
from ..models import SubAgentState, SubAgentStatus
from ..state import GraphDecision, Message, OrchestratorState, TaskInfo, TaskStatus
from ..strategies.council import CouncilStrategy
from ..strategies.intelligent import IntelligentStrategy
from ..strategies.parallel import ParallelStrategy
from ..strategies.registry import get_strategy_registry
from ..utils import ToolAllocator

logger = logging.getLogger(__name__)

# Register default strategies
registry = get_strategy_registry()
registry.register("council", CouncilStrategy())
registry.register("parallel", ParallelStrategy())
registry.register("intelligent", IntelligentStrategy())


# --- Legion Worker State ---


class LegionWorkerState(TypedDict):
    """
    State passed to a single Legion worker.

    This is a subset/derivative of the main OrchestratorState,
    containing only what the worker needs.
    """

    worker_id: str
    role: str  # e.g., "researcher", "optimist", "coder"
    task_description: str
    tools: List[str]
    user_id: str
    persona: str  # The main system persona (e.g. "hermes")
    context: Dict[str, Any]  # Shared context/collected info


# --- Legion Orchestrator Node ---


def legion_orchestrator_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    The central brain of the Legion swarm.

    Decides on the strategy (Council vs Parallel) and dispatches workers
    using the `Send` API.

    This node includes a human-in-the-loop interrupt point where execution
    pauses to allow approval or modification of the worker plan.
    """
    logger.info("Legion Orchestrator executing")

    user_message = state["messages"][-1]["content"]
    user_id = state["user_id"]

    # 1. Determine Strategy (if not already set)
    strategy_name = state.get("legion_strategy")

    if not strategy_name:
        # Simple heuristic for now, can be enhanced with LLM classifier
        # We can move this logic to a "StrategySelector" service later
        from ..parallel.task_decomposer import ParallelTaskDecomposer

        decomposer = ParallelTaskDecomposer()
        if decomposer.is_multi_agent_task(user_message):
            strategy_name = "parallel"
        else:
            strategy_name = "council"

        logger.info(f"Legion strategy determined: {strategy_name}")

    # 2. Get Strategy Implementation
    registry = get_strategy_registry()
    strategy = registry.get(strategy_name)

    if not strategy:
        logger.warning(f"Unknown strategy {strategy_name}, defaulting to council")
        strategy_name = "council"
        strategy = registry.get("council")

    # 3. Generate Worker Configurations
    try:
        workers = asyncio.run(
            strategy.generate_workers(
                query=user_message, context=state.get("collected_info", {})
            )
        )
    except Exception as e:
        logger.error(f"Strategy {strategy_name} failed to generate workers: {e}")
        # Fallback to a safe default
        workers = [
            {
                "worker_id": "fallback_worker",
                "role": "general",
                "task_description": user_message,
                "tools": [],
            }
        ]

    # 4. CONFIGURABLE INTERRUPT: Determine if we should pause for approval
    # Considers: auto-approve setting, complexity, risk level, estimated cost
    from langgraph.types import interrupt

    # Get routing decision from metadata for risk assessment
    routing_decision = state.get("metadata", {}).get("routing_decision")

    # Determine if interrupt is needed
    should_interrupt = _should_interrupt_for_approval(
        state=state, workers=workers, routing_decision=routing_decision
    )

    user_response = None
    if should_interrupt:
        logger.info(
            f"Interrupting for approval: {len(workers)} workers, "
            f"risk_level={routing_decision.get('risk_level') if routing_decision else 'unknown'}"
        )

        user_response = interrupt(
            {
                "type": "worker_plan_review",
                "strategy": strategy_name,
                "workers": workers,
                "worker_count": len(workers),
                "routing_info": (
                    {
                        "conversation_type": routing_decision.get("conversation_type"),
                        "risk_level": routing_decision.get("risk_level"),
                        "complexity_estimate": routing_decision.get(
                            "complexity_estimate"
                        ),
                        "estimated_cost": routing_decision.get("estimated_cost"),
                        "estimated_duration": routing_decision.get(
                            "estimated_duration"
                        ),
                    }
                    if routing_decision
                    else None
                ),
                "message": f"I've planned {len(workers)} workers using {strategy_name} strategy. Review the plan below.",
                "actions": ["approve", "modify", "cancel"],
            }
        )
    else:
        logger.info(
            f"Auto-approving: {len(workers)} workers (interrupts disabled or below threshold)"
        )

    # 5. Process user response (if resuming from interrupt)
    if user_response:
        action = user_response.get("action", "approve")

        if action == "cancel":
            logger.info("User cancelled worker plan")
            return {
                "legion_strategy": strategy_name,
                "legion_results": {},
                "next_action": GraphDecision.COMPLETE.value,
                "messages": state.get("messages", [])
                + [
                    Message(
                        role="assistant",
                        content="Worker plan cancelled. How else can I help?",
                        timestamp=datetime.utcnow().isoformat(),
                        metadata={"cancelled": True},
                    )
                ],
                "metadata": {**state.get("metadata", {}), "legion_cancelled": True},
            }

        elif action == "modify":
            logger.info("User modified worker plan")
            workers = user_response.get("workers", workers)
            logger.info(f"Updated to {len(workers)} workers after modification")

    # 6. Prepare for Dispatch
    logger.info(
        f"Proceeding with {len(workers)} Legion workers using strategy {strategy_name}"
    )

    return {
        "legion_strategy": strategy_name,
        "legion_results": {},  # Reset results
        "metadata": {**state.get("metadata", {}), "legion_worker_plans": workers},
    }


def legion_routing_edge(state: OrchestratorState):
    """
    DEPRECATED: Use RouterService.route_legion_workers() instead.

    Conditional edge function that generates `Send` objects.
    Kept for backwards compatibility during migration.
    """
    from .router_service import RouterService

    router = RouterService()
    return router.route_legion_workers(state)


# --- Legion Worker Node ---


def legion_worker_node(state: LegionWorkerState) -> Dict[str, Any]:
    """
    Executes a single unit of work.

    This node runs in parallel for each worker.
    Returns a state update that will be merged into the main state.
    """
    worker_id = state["worker_id"]
    logger.info(f"Legion worker {worker_id} starting")

    try:
        # Create Agent
        # We need to re-instantiate tools here because they can't be pickled in Send
        tool_allocator = ToolAllocator()

        # Allocate tools
        tools = tool_allocator.allocate_tools_for_task(
            task_type="general",  # We can refine this
            task_description=state["task_description"],
        )
        # Filter to keep only assigned tools if we passed them
        if state["tools"]:
            tools = [t for t in tools if t.name in state["tools"]]

        # Create Agent
        from ..state import AgentConfig

        config = AgentConfig(agent_type="legion_worker", required_tools=[])
        agent = AgentFactory.create_agent(config, tools)

        # Create SubAgentState
        sub_state = SubAgentState(
            agent_id=worker_id,
            status=SubAgentStatus.PROCESSING,
            task=state["task_description"],
            task_type="legion_worker",
            triggering_message=state["task_description"],
            collected_info=state["context"],
            metadata={"user_id": state["user_id"], "persona": state["persona"]},
        )

        # Execute
        if hasattr(agent, "execute_task_async"):
            result = asyncio.run(agent.execute_task_async(sub_state))
        else:
            # Use sync execute_task directly
            result = agent.execute_task(sub_state)

        logger.info(f"Legion worker {worker_id} finished")

        return {
            "legion_results": {
                worker_id: {
                    "result": result,
                    "role": state["role"],
                    "status": "success",
                }
            }
        }

    except Exception as e:
        logger.error(f"Legion worker {worker_id} failed: {e}")
        return {
            "legion_results": {
                worker_id: {"result": str(e), "role": state["role"], "status": "failed"}
            }
        }


# --- Legion Synthesis Node ---


def legion_synthesis_node(state: OrchestratorState) -> OrchestratorState:
    """
    Synthesizes results from all Legion workers.
    """
    logger.info("Legion Synthesis executing")

    results = state.get("legion_results", {})
    strategy_name = state.get("legion_strategy", "council")
    original_query = state["messages"][-1]["content"]

    if not results:
        return {
            **state,
            "next_action": GraphDecision.COMPLETE.value,
            "messages": state["messages"]
            + [Message(role="assistant", content="Legion swarm produced no results.")],
        }

    # Get Strategy
    registry = get_strategy_registry()
    strategy = registry.get(strategy_name)

    if not strategy:
        logger.warning(
            f"Unknown strategy {strategy_name} for synthesis, defaulting to council"
        )
        strategy = registry.get("council")

    try:
        final_response = asyncio.run(
            strategy.synthesize_results(
                original_query=original_query, results=results, persona=state["persona"]
            )
        )
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        final_response = (
            "I apologize, but I encountered an error synthesizing the results."
        )

    response_message = Message(
        role="assistant",
        content=final_response,
        timestamp=datetime.utcnow().isoformat(),
        metadata={"legion_strategy": strategy_name, "worker_count": len(results)},
    )

    return {
        **state,
        "messages": state["messages"] + [response_message],
        "next_action": GraphDecision.COMPLETE.value,
    }


def _should_interrupt_for_approval(
    state: OrchestratorState,
    workers: List[Dict],
    routing_decision: Optional[Dict] = None,
) -> bool:
    """
    Determine if we should interrupt for user approval.

    Configurable interrupt logic that considers:
    - Auto-approve setting in metadata
    - Number of workers (complexity threshold)
    - Risk level from routing decision
    - Estimated cost from routing decision

    Args:
        state: Current orchestrator state
        workers: List of planned workers
        routing_decision: Routing decision metadata (from routing intelligence)

    Returns:
        True if should interrupt for approval, False to auto-approve
    """
    # Check if auto-approve is explicitly enabled
    if state.get("metadata", {}).get("auto_approve", False):
        logger.debug("Auto-approve enabled in metadata, skipping interrupt")
        return False

    # Complexity threshold: More than 3 workers = interrupt
    if len(workers) > 3:
        logger.debug(f"Complexity threshold met: {len(workers)} > 3 workers")
        return True

    # Risk-based threshold from routing intelligence
    if routing_decision:
        risk_level = routing_decision.get("risk_level", "LOW")

        # HIGH or CRITICAL risk always requires approval
        if risk_level in ["HIGH", "CRITICAL"]:
            logger.debug(f"High risk level detected: {risk_level}")
            return True

        # Cost threshold: Estimated cost > $1.00 requires approval
        estimated_cost = routing_decision.get("estimated_cost")
        if estimated_cost and estimated_cost > 1.0:
            logger.debug(f"Cost threshold met: ${estimated_cost:.2f}")
            return True

        # Approval was recommended by routing intelligence
        if routing_decision.get("should_seek_approval", False):
            logger.debug("Routing intelligence recommended seeking approval")
            return True

    # Default: No interrupt needed (auto-approve)
    logger.debug("All thresholds passed, auto-approving")
    return False
