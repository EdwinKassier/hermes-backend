"""
Unified Legion Orchestration System.

This module implements the "Legion" swarm architecture using LangGraph's `Send` API
for dynamic, parallel agent execution. It unifies previous "Council" and "Parallel"
modes into a single, flexible system.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from ..agents.factory import AgentFactory
from ..models import SubAgentState, SubAgentStatus
from ..state import GraphDecision, Message, OrchestratorState
from ..strategies.council import CouncilStrategy
from ..strategies.intelligent import IntelligentStrategy
from ..strategies.parallel import ParallelStrategy
from ..strategies.registry import get_strategy_registry
from ..utils import ToolAllocator

logger = logging.getLogger(__name__)

# Register default strategies using lazy factories to avoid import-time dependency checks
registry = get_strategy_registry()
registry.register("council", lambda: CouncilStrategy())
registry.register("parallel", lambda: ParallelStrategy())
registry.register("intelligent", lambda: IntelligentStrategy())


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
    execution_level: int  # For dependency-aware execution ordering
    # Retry configuration
    max_retries: int  # Maximum retry attempts (default: 2)
    retry_delay_seconds: float  # Base delay between retries (default: 1.0)


# --- Legion Orchestrator Node ---


async def legion_orchestrator_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    The central brain of the Legion swarm.

    Decides on the strategy (Council vs Parallel) and dispatches workers
    using the `Send` API.

    This node includes a human-in-the-loop interrupt point where execution
    pauses to allow approval or modification of the worker plan.

    Note: This is an async node to properly await strategy.generate_workers()
    without blocking the event loop.
    """
    logger.info("Legion Orchestrator executing")

    user_message = state["messages"][-1]["content"]
    # user_id used by workers, not directly in orchestrator

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

        logger.info("Legion strategy determined: %s", strategy_name)

    # 2. Get Strategy Implementation
    strategy_registry = get_strategy_registry()
    strategy = strategy_registry.get(strategy_name)

    if not strategy:
        logger.warning("Unknown strategy %s, defaulting to council", strategy_name)
        strategy_name = "council"
        strategy = strategy_registry.get("council")

    # 3. Generate Worker Configurations (properly awaited)
    try:
        workers = await strategy.generate_workers(
            query=user_message, context=state.get("collected_info", {})
        )
    except (ValueError, RuntimeError) as e:
        logger.error("Strategy %s failed to generate workers: %s", strategy_name, e)
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
            "Interrupting for approval: %d workers, risk_level=%s",
            len(workers),
            routing_decision.get("risk_level") if routing_decision else "unknown",
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
            "Auto-approving: %d workers (interrupts disabled or below threshold)",
            len(workers),
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
                "execution_path": [
                    {
                        "node": "legion_orchestrator",
                        "timestamp": datetime.now().isoformat(),
                    }
                ],
            }

        elif action == "modify":
            logger.info("User modified worker plan")
            workers = user_response.get("workers", workers)
            logger.info("Updated to %d workers after modification", len(workers))

    # 6. Calculate execution levels for dependency-aware execution
    # Group workers by their execution_level and determine total levels
    execution_levels_set = set()
    for worker in workers:
        level = worker.get("execution_level", 0)
        execution_levels_set.add(level)

    total_levels = len(execution_levels_set) if execution_levels_set else 1

    # 7. Prepare for Dispatch
    logger.info(
        "Proceeding with %d Legion workers using strategy %s across %d execution level(s)",
        len(workers),
        strategy_name,
        total_levels,
    )

    # Get fail_on_level_error from metadata or default to False (continue with partial)
    fail_on_level_error = state.get("metadata", {}).get("fail_on_level_error", False)

    return {
        "legion_strategy": strategy_name,
        "legion_results": {},  # Reset results
        "current_execution_level": 0,  # Start at level 0
        "total_execution_levels": total_levels,
        "level_results": {},  # Initialize per-level results storage
        "fail_on_level_error": fail_on_level_error,
        "fail_on_level_error": fail_on_level_error,
        "metadata": {**state.get("metadata", {}), "legion_worker_plans": workers},
        "execution_path": [
            {"node": "legion_orchestrator", "timestamp": datetime.now().isoformat()}
        ],
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


async def legion_dispatch_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Pass-through node that serves as a dispatch point for level-based execution.

    This node is used when looping back from legion_level_complete to dispatch
    the next level's workers. It doesn't modify state; it just allows the
    conditional edge (legion_routing_edge) to run and dispatch workers.
    """
    current_level = state.get("current_execution_level", 0)
    total_levels = state.get("total_execution_levels", 1)

    logger.info(
        "Legion dispatch node: Preparing to dispatch level %d of %d",
        current_level,
        total_levels,
    )

    # Return minimal update - the routing edge will handle actual dispatch
    return {
        "execution_path": [
            {"node": "legion_dispatch", "timestamp": datetime.now().isoformat()}
        ],
    }


# --- Graceful Degradation Helper (Issue 2: Error Recovery) ---


def _generate_graceful_degradation_message(
    worker_id: str,
    role: str,
    task: str,
    error: Optional[Exception],
) -> str:
    """
    Generate a helpful message when worker cannot complete fully.

    Instead of returning an error message, provides context about what
    was attempted and any partial insights that might still be useful.

    Args:
        worker_id: ID of the worker
        role: Worker's role (e.g., "researcher", "coder")
        task: Task description
        error: The exception that caused failure

    Returns:
        Helpful degradation message with context
    """
    error_type = type(error).__name__ if error else "Unknown"
    error_msg = str(error) if error else "Unknown error occurred"

    # Provide context-aware guidance based on role
    role_guidance = {
        "researcher": "Consider trying a more specific query or different search terms.",
        "coder": "The code generation encountered issues. You may need to provide more specific requirements.",
        "analyst": "Data analysis was incomplete. Try with a smaller dataset or simpler query.",
        "optimist": "Could not fully analyze from this perspective.",
        "critic": "Critical analysis was incomplete.",
        "general": "The task could not be fully completed.",
    }

    guidance = role_guidance.get(role.lower(), role_guidance["general"])

    return f"""**Partial Result - {role.title()} Worker**

I attempted to complete: "{task[:100]}{'...' if len(task) > 100 else ''}"

**What happened:** The {role} component encountered a {error_type} during execution.

**Recommendation:** {guidance}

While I couldn't fully complete this task, here's what I can tell you:
- The request was understood and processing was attempted
- The specific issue was: {error_msg[:200]}{'...' if len(error_msg) > 200 else ''}

This partial result is still included so the synthesis can acknowledge this limitation and work with available information from other workers."""


# --- Legion Worker Node ---


async def legion_worker_node(state: LegionWorkerState) -> Dict[str, Any]:
    """
    Executes a single unit of work with retry support.

    This node runs in parallel for each worker via LangGraph's Send API.
    Returns a state update that will be merged into the main state.

    Note: This is an async node to properly support async agent execution
    without blocking the event loop.

    Features:
    - Retry with exponential backoff for transient failures
    - Timeout protection to prevent runaway execution
    - Graceful error handling
    - Execution metrics tracking
    """
    from ..utils.task_timeout import (
        DEFAULT_WORKER_TIMEOUT,
        TaskTimeoutError,
        with_timeout,
    )

    worker_id = state["worker_id"]
    start_time = datetime.utcnow()
    max_retries = state.get("max_retries", 2)
    retry_delay = state.get("retry_delay_seconds", 1.0)

    logger.info("Legion worker %s starting (max_retries=%d)", worker_id, max_retries)

    async def execute_worker_task() -> str:
        """Inner function for timeout wrapping."""
        # Use ToolRegistry for efficient tool lookup (avoids re-instantiation)
        from ..utils.tool_registry import get_tool_registry

        registry = get_tool_registry()

        # Get tools by name from registry - much more efficient than reallocating
        if state["tools"]:
            tools = registry.get_tools(state["tools"])
        else:
            # Fallback to allocating if no specific tools specified
            tool_allocator = ToolAllocator()
            tools = tool_allocator.allocate_tools_for_task(
                task_type="general",
                task_description=state["task_description"],
            )

        # Create Agent using the worker's role as the agent type
        from ..state import AgentConfig

        # Map worker role to agent type
        # Roles like "researcher", "coder" should map to "research", "code"
        role = state["role"].lower()
        agent_type_map = {
            "researcher": "research",
            "coder": "code",
            "programmer": "code",
            "analyst": "analysis",
            "data_analyst": "data",
        }
        agent_type = agent_type_map.get(role, role)  # Default to role if not in map

        config = AgentConfig(agent_type=agent_type, required_tools=[])
        agent = AgentFactory.create_agent(config, tools)

        # Create SubAgentState
        sub_state = SubAgentState(
            agent_id=worker_id,
            status=SubAgentStatus.PROCESSING,
            task=state["task_description"],
            task_type=agent_type,  # Use mapped agent_type
            triggering_message=state["task_description"],
            collected_info=state["context"],
            metadata={"user_id": state["user_id"], "persona": state["persona"]},
        )

        # Execute - prefer async if available
        if hasattr(agent, "execute_task_async"):
            return await agent.execute_task_async(sub_state)
        else:
            # Fall back to sync execute_task (run in thread pool to avoid blocking)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, agent.execute_task, sub_state)

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            # Execute with timeout protection
            result = await with_timeout(
                execute_worker_task(),
                timeout=DEFAULT_WORKER_TIMEOUT,
                task_id=worker_id,
                raise_on_timeout=True,
            )

            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.info(
                "Legion worker %s finished in %.2fs (attempt %d/%d)",
                worker_id,
                duration,
                attempt + 1,
                max_retries + 1,
            )

            return {
                "legion_results": {
                    worker_id: {
                        "result": result,
                        "role": state["role"],
                        "status": "success",
                        "duration_seconds": duration,
                        "execution_level": state.get("execution_level", 0),
                        "attempts": attempt + 1,
                    }
                }
            }

        except TaskTimeoutError as e:
            last_error = e
            duration = (datetime.utcnow() - start_time).total_seconds()
            # Don't retry timeouts - they likely indicate the task is too complex
            logger.error("Legion worker %s timed out after %.2fs", worker_id, duration)
            return {
                "legion_results": {
                    worker_id: {
                        "result": f"Task timed out after {duration:.1f}s. The operation was too complex or slow.",
                        "role": state["role"],
                        "status": "timeout",
                        "duration_seconds": duration,
                        "error": str(e),
                        "execution_level": state.get("execution_level", 0),
                        "attempts": attempt + 1,
                    }
                }
            }

        except (ValueError, RuntimeError, ConnectionError) as e:
            last_error = e
            if attempt < max_retries:
                # Calculate exponential backoff delay
                delay = retry_delay * (2**attempt)
                logger.warning(
                    "Legion worker %s attempt %d/%d failed: %s. Retrying in %.1fs...",
                    worker_id,
                    attempt + 1,
                    max_retries + 1,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)
                continue

            # Final failure after all retries
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.error(
                "Legion worker %s failed after %d attempts in %.2fs: %s",
                worker_id,
                max_retries + 1,
                duration,
                e,
            )

    # Should not reach here, but handle as final failure with graceful degradation
    # ENHANCEMENT (Issue 2): Instead of hard failure, produce a helpful partial result
    duration = (datetime.utcnow() - start_time).total_seconds()

    degraded_result = _generate_graceful_degradation_message(
        worker_id=worker_id,
        role=state["role"],
        task=state["task_description"],
        error=last_error,
    )

    logger.warning(
        "Legion worker %s returning degraded result after %d attempts in %.2fs",
        worker_id,
        max_retries + 1,
        duration,
    )

    return {
        "legion_results": {
            worker_id: {
                "result": degraded_result,
                "role": state["role"],
                "status": "degraded",  # Changed from "failed" for graceful handling
                "duration_seconds": duration,
                "error": str(last_error) if last_error else "Unknown error",
                "execution_level": state.get("execution_level", 0),
                "attempts": max_retries + 1,
                "is_partial": True,  # Flag for synthesis to handle appropriately
            }
        }
    }


# --- Legion Level Complete Node ---


async def legion_level_complete_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Handles completion of an execution level and determines next steps.

    This node:
    1. Stores current level's results in level_results
    2. Tracks dispatched workers for backpressure batching
    3. Checks for failures if fail_on_level_error=True
    4. Checks if all workers in level completed (for batching support)
    5. Increments current_execution_level when level is complete
    6. Returns state update for routing decision

    The routing edge (legion_level_routing_edge) will decide whether to:
    - Loop back for more workers in same level (batching)
    - Loop back for the next level
    - Proceed to synthesis (all levels complete)
    - Handle errors
    """
    current_level = state.get("current_execution_level", 0)
    total_levels = state.get("total_execution_levels", 1)
    fail_on_error = state.get("fail_on_level_error", False)
    legion_results = state.get("legion_results", {})
    worker_plans = state.get("metadata", {}).get("legion_worker_plans", [])

    # Get workers planned for this level
    level_planned_workers = [
        w for w in worker_plans if w.get("execution_level", 0) == current_level
    ]

    # Collect results for this level
    level_workers = {}
    failed_workers = []
    for worker_id, result_data in legion_results.items():
        worker_level = result_data.get("execution_level", 0)
        if worker_level == current_level:
            level_workers[worker_id] = result_data
            if result_data.get("status") in ["failed", "timeout"]:
                failed_workers.append(worker_id)

    # Track dispatched workers for backpressure batching
    completed_worker_ids = set(level_workers.keys())
    dispatched_ids = list(completed_worker_ids)

    # Check if all planned workers for this level have completed
    all_level_workers_complete = len(level_workers) >= len(level_planned_workers)

    logger.info(
        "Level %d: %d/%d workers completed, %d failed. All complete: %s",
        current_level,
        len(level_workers),
        len(level_planned_workers),
        len(failed_workers),
        all_level_workers_complete,
    )

    # Store level results
    level_results = state.get("level_results", {})
    level_results[current_level] = {
        "workers": level_workers,
        "failed_count": len(failed_workers),
        "success_count": len(level_workers) - len(failed_workers),
        "total_planned": len(level_planned_workers),
    }

    # Check for failure handling
    if fail_on_error and failed_workers:
        logger.warning(
            "Level %d had failures and fail_on_level_error=True. Stopping execution.",
            current_level,
        )
        return {
            "level_results": level_results,
            "current_execution_level": current_level,  # Don't increment
            "metadata": {
                **state.get("metadata", {}),
                "level_execution_stopped": True,
                "stopped_at_level": current_level,
                "failed_workers": failed_workers,
                "dispatched_worker_ids": dispatched_ids,
                "dispatched_worker_ids": dispatched_ids,
            },
            "execution_path": [
                {
                    "node": "legion_level_complete",
                    "timestamp": datetime.now().isoformat(),
                }
            ],
        }

    # If not all workers in this level are complete, stay at this level
    # (for backpressure batching - more workers need to be dispatched)
    if not all_level_workers_complete:
        logger.info(
            "Level %d has %d more workers to dispatch (backpressure)",
            current_level,
            len(level_planned_workers) - len(level_workers),
        )
        return {
            "level_results": level_results,
            "current_execution_level": current_level,  # Stay at same level
            "metadata": {
                **state.get("metadata", {}),
                "dispatched_worker_ids": dispatched_ids,
                "pending_workers_in_level": True,
                "pending_workers_in_level": True,
            },
            "execution_path": [
                {
                    "node": "legion_level_complete",
                    "timestamp": datetime.now().isoformat(),
                }
            ],
        }

    # All workers in level complete - increment to next level
    next_level = current_level + 1

    return {
        "level_results": level_results,
        "current_execution_level": next_level,
        "metadata": {
            **state.get("metadata", {}),
            "dispatched_worker_ids": [],  # Reset for next level
            "pending_workers_in_level": False,
            "pending_workers_in_level": False,
        },
        "execution_path": [
            {"node": "legion_level_complete", "timestamp": datetime.now().isoformat()}
        ],
    }


def legion_level_routing_edge(state: OrchestratorState) -> str:
    """
    Routing edge that determines whether to continue to next level or synthesize.

    Supports backpressure batching by routing back to dispatch if there are
    pending workers in the current level.

    Returns:
        "legion_dispatch" - More workers/levels to process
        "legion_synthesis" - All levels complete, proceed to synthesis
    """
    current_level = state.get("current_execution_level", 0)
    total_levels = state.get("total_execution_levels", 1)
    metadata = state.get("metadata", {})

    # Check if execution was stopped due to errors
    if metadata.get("level_execution_stopped"):
        logger.info(
            "Execution was stopped due to level errors, proceeding to synthesis"
        )
        return "legion_synthesis"

    # Check if there are pending workers in current level (backpressure batching)
    if metadata.get("pending_workers_in_level"):
        logger.info(
            "Pending workers in level %d, dispatching next batch",
            current_level,
        )
        return "legion_dispatch"

    # Check if all levels are complete
    if current_level >= total_levels:
        logger.info("All %d levels complete, proceeding to synthesis", total_levels)
        return "legion_synthesis"

    # More levels to process
    logger.info("Proceeding to level %d of %d", current_level, total_levels)
    return "legion_dispatch"


# --- Legion Synthesis Node ---


async def legion_synthesis_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Synthesizes results from all Legion workers.

    Includes result validation to filter out invalid/empty results before synthesis.

    Note: This is an async node to properly await strategy.synthesize_results()
    without blocking the event loop.

    Returns only changed fields to leverage state reducers properly,
    avoiding the anti-pattern of spreading the entire state.
    """
    from ..utils.result_validator import ResultValidator

    logger.info("Legion Synthesis executing")

    results = state.get("legion_results", {})
    strategy_name = state.get("legion_strategy", "council")
    original_query = state["messages"][-1]["content"]

    if not results:
        # Return only changed fields - messages will be appended via reducer
        return {
            "next_action": GraphDecision.COMPLETE.value,
            "messages": [
                Message(
                    role="assistant",
                    content="Legion swarm produced no results.",
                    timestamp=datetime.utcnow().isoformat(),
                    metadata={"legion_strategy": strategy_name, "no_results": True},
                )
            ],
            "execution_path": [
                {"node": "legion_synthesis", "timestamp": datetime.now().isoformat()}
            ],
        }

    # Validate results before synthesis
    validator = ResultValidator()
    valid_results, validation_issues = validator.validate_batch(results)

    if not valid_results:
        # All results were invalid
        logger.warning("All worker results failed validation")
        issue_summary = "; ".join(
            [
                f"{issue['worker_id']}: {issue['reason']}"
                for issue in validation_issues[:3]  # Limit to first 3
            ]
        )
        return {
            "next_action": GraphDecision.COMPLETE.value,
            "messages": [
                Message(
                    role="assistant",
                    content=f"I apologize, but I couldn't generate valid results. Issues: {issue_summary}",
                    timestamp=datetime.utcnow().isoformat(),
                    metadata={
                        "legion_strategy": strategy_name,
                        "validation_failed": True,
                        "issues": validation_issues,
                    },
                )
            ],
            "execution_path": [
                {"node": "legion_synthesis", "timestamp": datetime.now().isoformat()}
            ],
        }

    logger.info(
        "Result validation: %d/%d valid results for synthesis",
        len(valid_results),
        len(results),
    )

    # Get Strategy
    strategy_registry = get_strategy_registry()
    strategy = strategy_registry.get(strategy_name)

    if not strategy:
        logger.warning(
            "Unknown strategy %s for synthesis, defaulting to council", strategy_name
        )
        strategy = strategy_registry.get("council")

    try:
        # Use only validated results for synthesis
        final_response = await strategy.synthesize_results(
            original_query=original_query,
            results=valid_results,
            persona=state["persona"],
        )
    except (ValueError, RuntimeError) as e:
        logger.error("Synthesis failed: %s", e)
        final_response = (
            "I apologize, but I encountered an error synthesizing the results."
        )

    response_message = Message(
        role="assistant",
        content=final_response,
        timestamp=datetime.utcnow().isoformat(),
        metadata={
            "legion_strategy": strategy_name,
            "worker_count": len(results),
            "valid_worker_count": len(valid_results),
            "validation_issues": len(validation_issues),
        },
    )

    # Return only changed fields - messages list will be appended via operator.add reducer
    return {
        "messages": [response_message],
        "next_action": GraphDecision.COMPLETE.value,
        "execution_path": [
            {"node": "legion_synthesis", "timestamp": datetime.now().isoformat()}
        ],
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
        logger.debug("Complexity threshold met: %d > 3 workers", len(workers))
        return True

    # Risk-based threshold from routing intelligence
    if routing_decision:
        risk_level = routing_decision.get("risk_level", "LOW")

        # HIGH or CRITICAL risk always requires approval
        if risk_level in ["HIGH", "CRITICAL"]:
            logger.debug("High risk level detected: %s", risk_level)
            return True

        # Cost threshold: Estimated cost > $1.00 requires approval
        estimated_cost = routing_decision.get("estimated_cost")
        if estimated_cost and estimated_cost > 1.0:
            logger.debug("Cost threshold met: $%.2f", estimated_cost)
            return True

        # Approval was recommended by routing intelligence
        if routing_decision.get("should_seek_approval", False):
            logger.debug("Routing intelligence recommended seeking approval")
            return True

    # Default: No interrupt needed (auto-approve)
    logger.debug("All thresholds passed, auto-approving")
    return False
