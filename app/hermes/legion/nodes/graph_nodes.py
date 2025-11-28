"""LangGraph nodes for orchestration workflow."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List

from app.shared.utils.service_loader import get_async_llm_service, get_gemini_service

from ..agents.factory import AgentFactory
from ..models import SubAgentState, SubAgentStatus
from ..orchestrator import InformationExtractor, TaskIdentifier
from ..state import (
    AgentConfig,
    AgentInfo,
    GraphDecision,
    Message,
    OrchestratorState,
    TaskInfo,
    TaskStatus,
)
from ..utils import ToolAllocator

logger = logging.getLogger(__name__)


async def orchestrator_node(state: OrchestratorState) -> OrchestratorState:
    """
    Main orchestrator node - analyzes request and routes to appropriate action.

    This is the entry point and decision-making hub of the graph.
    """
    logger.info("Orchestrator node processing request")

    # Initialize decision rationale tracking
    decision_rationale = state.get("decision_rationale", [])
    current_decision = {
        "timestamp": datetime.now().isoformat(),
        "node": "orchestrator",
        "analysis": {},
        "decisions": {},
        "reasoning": {},
    }

    # Get latest user message
    messages = state.get("messages", [])
    if not messages:
        return _handle_empty_message(state, current_decision, decision_rationale)

    latest_message = messages[-1]
    user_message = latest_message.get("content", "")

    # NEW: Analyze routing with dynamic intelligence
    from ..intelligence.routing_intelligence import RoutingAction
    from ..intelligence.routing_service import RoutingIntelligence

    routing_intel = RoutingIntelligence()

    # Build agent context for routing analysis
    agent_context = {
        "active_task": state.get("current_task_id"),
        "waiting_for_info": bool(state.get("required_info")),
        "collected_info": state.get("collected_info", {}),
        "conversation_phase": state.get("metadata", {}).get(
            "conversation_phase", "initiating"
        ),
    }

    routing_decision = await routing_intel.analyze(
        message=user_message,
        conversation_history=messages[:-1],  # Exclude current message
        current_agent_context=agent_context,
    )

    # Store routing decision in metadata and decision rationale
    current_decision["analysis"]["routing_decision"] = {
        "action": routing_decision.action,
        "conversation_type": routing_decision.conversation_type,
        "confidence": routing_decision.confidence,
        "user_goal": routing_decision.user_goal,
        "phase": routing_decision.conversation_phase,
    }

    state_metadata = state.get("metadata", {})
    state_metadata["routing_decision"] = routing_decision.dict()
    state_metadata["conversation_phase"] = routing_decision.conversation_phase

    logger.info(
        f"Routing intelligence: {routing_decision.action} "
        f"(type: '{routing_decision.conversation_type}', confidence: {routing_decision.confidence:.2f})"
    )

    # Handle topic changes detected by routing intelligence
    if routing_decision.topic_change_detected:
        logger.info(
            f"Topic change detected by routing intel ({routing_decision.topic_change_confidence:.2f}): "
            f"{routing_decision.previous_topic_description} → {routing_decision.new_topic_description}"
        )

        if routing_decision.should_abandon_current_task and state.get(
            "current_task_id"
        ):
            return _handle_topic_change(
                state, routing_decision, current_decision, decision_rationale
            )

    # Fast-path for simple responses (no orchestration needed)
    if routing_decision.action == RoutingAction.SIMPLE_RESPONSE:
        logger.info(
            f"Fast-path: Simple response for '{routing_decision.conversation_type}'"
        )
        current_decision["decisions"]["action"] = "simple_response"
        current_decision["reasoning"]["action"] = routing_decision.reasoning
        decision_rationale.append(current_decision)

        return {
            "next_action": GraphDecision.COMPLETE.value,
            "decision_rationale": decision_rationale,
            "metadata": state_metadata,
        }

    # Handle GATHER_INFO action from routing intelligence
    if routing_decision.action == RoutingAction.GATHER_INFO:
        logger.info(
            f"Routing intelligence: Need to gather info - {routing_decision.reasoning}"
        )
        current_decision["decisions"]["action"] = "gather_info"
        current_decision["reasoning"]["action"] = routing_decision.reasoning
        decision_rationale.append(current_decision)

        return {
            "next_action": GraphDecision.GATHER_INFO.value,
            "decision_rationale": decision_rationale,
            "metadata": state_metadata,
        }

    # Handle ORCHESTRATE action from routing intelligence
    if routing_decision.action == RoutingAction.ORCHESTRATE:
        logger.info(
            f"Routing intelligence: Orchestration needed - {routing_decision.reasoning}"
        )
        current_decision["decisions"]["action"] = "orchestrate"
        current_decision["reasoning"]["action"] = routing_decision.reasoning
        current_decision["analysis"][
            "requires_agents"
        ] = routing_decision.requires_agents
        current_decision["analysis"][
            "complexity"
        ] = routing_decision.complexity_estimate

        # Check if multi-agent task (parallel/legion mode)
        from ..parallel.task_decomposer import ParallelTaskDecomposer

        decomposer = ParallelTaskDecomposer()

        if decomposer.is_multi_agent_task(user_message):
            logger.info("Multi-agent orchestration detected - routing to legion")
            decision_rationale.append(current_decision)
            return {
                "next_action": "legion_orchestrate",
                "decision_rationale": decision_rationale,
                "metadata": state_metadata,
            }

        # Single agent orchestration - infer agent type from routing decision
        # This avoids an extra LLM call to TaskIdentifier
        task_type = _infer_agent_type(routing_decision)
        logger.info(f"Inferred task type '{task_type}' from routing decision")

        if not task_type:
            logger.warning(
                "Orchestration needed but no task type identified, falling back to general response"
            )
            decision_rationale.append(current_decision)
            return {
                "next_action": GraphDecision.COMPLETE.value,
                "decision_rationale": decision_rationale,
                "metadata": state_metadata,
            }

        # Create agent for single-agent orchestration
        return _handle_new_task(
            state, user_message, task_type, current_decision, decision_rationale
        )

    # Handle ERROR action from routing intelligence
    if routing_decision.action == RoutingAction.ERROR:
        logger.error(f"Routing intelligence error: {routing_decision.reasoning}")
        current_decision["decisions"]["action"] = "error"
        current_decision["reasoning"]["action"] = routing_decision.reasoning
        decision_rationale.append(current_decision)

        return {
            "next_action": GraphDecision.ERROR.value,
            "decision_rationale": decision_rationale,
            "metadata": state_metadata,
        }

    # Fallback: Should never reach here if routing intelligence works correctly
    logger.error(
        f"Unexpected routing state: action={routing_decision.action}, "
        f"falling back to general response"
    )
    current_decision["decisions"]["action"] = "fallback"
    current_decision["reasoning"][
        "action"
    ] = "Unexpected routing state - using fallback"
    decision_rationale.append(current_decision)

    return {
        "next_action": GraphDecision.COMPLETE.value,
        "decision_rationale": decision_rationale,
        "metadata": state_metadata,
    }


def _infer_agent_type(routing_decision) -> str:
    """
    Infer agent type from routing decision metadata.

    Avoids redundant LLM calls by using the rich context from RoutingIntelligence.
    """
    # Combine relevant fields for keyword analysis
    text = (
        f"{routing_decision.conversation_type} "
        f"{routing_decision.user_goal} "
        f"{routing_decision.reasoning}"
    ).lower()

    # Check for specific agent capabilities
    if any(
        k in text
        for k in ["code", "program", "implement", "function", "class", "script"]
    ):
        return "code"

    if any(k in text for k in ["data", "analy", "metric", "statistic", "trend"]):
        return "analysis"

    # Default to research for most complex tasks as it's the most versatile
    return "research"


def _handle_empty_message(state, decision, rationale):
    """Handle case with no messages."""
    decision["decisions"]["action"] = "complete"
    decision["reasoning"]["action"] = "No messages to process"
    rationale.append(decision)

    return {
        "next_action": GraphDecision.COMPLETE.value,
        "decision_rationale": rationale,
    }


def _handle_cancellation(state, decision, rationale):
    """Handle user cancellation request."""
    decision["decisions"]["action"] = "cancel"
    decision["reasoning"]["action"] = "User requested cancellation"
    rationale.append(decision)

    logger.info("Cancellation requested - stopping all tasks")

    # Cancel all active tasks - create new ledger to avoid mutation
    task_ledger = {**state.get("task_ledger", {})}
    for task_id, task_info in task_ledger.items():
        if task_info.status in [TaskStatus.IN_PROGRESS, TaskStatus.AWAITING_INPUT]:
            # Create copy with updated status to avoid mutating original
            task_info.status = TaskStatus.CANCELLED

    return {
        "next_action": GraphDecision.COMPLETE.value,
        "decision_rationale": rationale,
        "task_ledger": task_ledger,
        "current_task_id": None,
        "current_agent_id": None,
    }


def _handle_topic_change(state, routing_decision, decision, rationale):
    """Handle topic change detected by routing intelligence."""
    decision["analysis"]["topic_change"] = {
        "detected": True,
        "confidence": routing_decision.topic_change_confidence,
        "previous_topic": routing_decision.previous_topic_description,
        "new_topic": routing_decision.new_topic_description,
    }
    decision["decisions"]["action"] = "topic_change_replan"
    decision["reasoning"][
        "action"
    ] = f"Topic change: {routing_decision.previous_topic_description} → {routing_decision.new_topic_description}"
    rationale.append(decision)

    logger.info(
        f"Handling topic change: {routing_decision.previous_topic_description} → {routing_decision.new_topic_description}"
    )

    # Cancel current task - create new ledger to avoid mutation
    task_ledger = {**state.get("task_ledger", {})}
    current_task_id = state.get("current_task_id")
    if current_task_id and current_task_id in task_ledger:
        task_ledger[current_task_id].status = TaskStatus.CANCELLED
        task_ledger[current_task_id].error = (
            f"User changed topic to: {routing_decision.new_topic_description}"
        )

    # Add system message explaining the change
    cancel_msg = {
        "role": "assistant",
        "content": f"I've stopped working on {routing_decision.previous_topic_description}. Let me help with {routing_decision.new_topic_description} instead.",
        "timestamp": datetime.now().isoformat(),
        "metadata": {"topic_change": True, "system_message": True},
    }

    return {
        "messages": [cancel_msg],  # Will be appended via operator.add reducer
        "next_action": GraphDecision.REPLAN.value,
        "decision_rationale": rationale,
        "task_ledger": task_ledger,
        "current_task_id": None,
        "current_agent_id": None,
        "metadata": {
            **state.get("metadata", {}),
            "topic_change": {
                "confidence": routing_decision.topic_change_confidence,
                "previous": routing_decision.previous_topic_description,
                "new": routing_decision.new_topic_description,
            },
        },
    }


def _handle_active_agent(state, agent_id, decision, rationale):
    """Handle routing when an agent is already active."""
    decision["analysis"]["has_active_agent"] = True
    decision["analysis"]["active_agent_id"] = agent_id
    decision["decisions"]["action"] = "gather_info"
    decision["reasoning"][
        "action"
    ] = f"Continuing with active agent '{agent_id}' to gather required information"
    rationale.append(decision)

    logger.info(f"Active agent {agent_id} - routing to info gathering")
    return {
        "next_action": GraphDecision.GATHER_INFO.value,
        "decision_rationale": rationale,
    }


def _handle_multi_agent(state, decision, rationale):
    """Handle multi-agent task routing."""
    decision["analysis"]["multi_agent_task_detected"] = True
    decision["decisions"]["action"] = "legion_orchestration"
    decision["reasoning"][
        "action"
    ] = "Detected multi-agent task, routing to legion orchestrator"
    rationale.append(decision)

    logger.info("Multi-agent task detected - routing to legion orchestrator")
    return {
        "next_action": "legion_orchestrate",
        "decision_rationale": rationale,
    }


def _handle_general_conversation(state, decision, rationale):
    """Handle general conversation without agents."""
    decision["decisions"]["action"] = "complete"
    decision["decisions"]["agent_needed"] = False
    decision["reasoning"][
        "action"
    ] = "User message appears to be general conversation, not requiring specialized agent"
    decision["reasoning"][
        "task_analysis"
    ] = "No specific task keywords detected (research, code, analysis, data)"
    rationale.append(decision)

    logger.info("No specific task identified - general conversation")
    return {
        "next_action": GraphDecision.COMPLETE.value,
        "decision_rationale": rationale,
    }


def _is_factual_question(user_message: str, task_type: str) -> bool:
    """Check if message is a simple factual question."""
    user_message_lower = user_message.lower()

    factual_indicators = [
        "where",
        "when",
        "who",
        "what",
        "which",
        "how old",
        "how many",
        "does he",
        "does she",
        "is he",
        "is she",
        "are they",
        "currently",
        "work",
        "skill",
        "experience",
        "background",
    ]

    explicit_task_indicators = [
        "write code",
        "generate",
        "create",
        "build",
        "implement",
        "analyze data",
        "calculate",
        "process",
        "transform",
        "research and",
        "investigate and",
        "find out about",
    ]

    is_explicit_task = any(
        indicator in user_message_lower for indicator in explicit_task_indicators
    )
    is_factual_question = any(
        indicator in user_message_lower for indicator in factual_indicators
    )

    return is_factual_question and not is_explicit_task


def _handle_factual_question(state, task_type, decision, rationale):
    """Handle factual questions using persona knowledge."""
    decision["decisions"]["action"] = "complete"
    decision["decisions"]["agent_needed"] = False
    decision["decisions"]["task_type_detected"] = task_type
    decision["reasoning"][
        "action"
    ] = f"Factual question detected (task_type: {task_type}), using persona knowledge instead of agent"
    decision["reasoning"][
        "analysis"
    ] = "Simple factual queries should be answered directly rather than through agent orchestration"
    rationale.append(decision)

    logger.info(
        f"Factual question detected (task_type: {task_type}) - routing to general response"
    )
    return {
        "next_action": GraphDecision.COMPLETE.value,
        "decision_rationale": rationale,
    }


def _handle_new_task(state, user_message, task_type, decision, rationale):
    """Handle creation of a new task and agent."""
    decision["decisions"]["agent_needed"] = True
    decision["decisions"]["selected_task_type"] = task_type

    # Allocate tools
    tool_allocator = ToolAllocator()
    allocated_tools = tool_allocator.allocate_tools_for_task(
        task_type=task_type, task_description=user_message
    )

    decision["analysis"]["tool_allocation"] = {
        "task_type": task_type,
        "tools_allocated": [t.name for t in allocated_tools] if allocated_tools else [],
        "allocation_strategy": "task-based (not persona-based)",
    }
    decision["reasoning"][
        "tool_selection"
    ] = f"Selected tools relevant to '{task_type}' tasks based on tool capability mappings"

    # Create agent
    try:
        agent, agent_info = AgentFactory.create_agent_from_task(
            task_description=user_message,
            task_type=task_type,
            tools=allocated_tools,
        )

        decision["decisions"]["agent_created"] = True
        decision["decisions"]["agent_id"] = agent_info.agent_id
        decision["decisions"]["agent_type"] = agent_info.agent_type
        decision["reasoning"]["agent_selection"] = (
            f"Created '{agent_info.agent_type}' agent (ID: {agent_info.agent_id}) "
            f"because task type '{task_type}' matches this agent's capabilities: {agent.task_types}"
        )

        logger.info(f"Created agent: {agent_info.agent_id} for task type: {task_type}")

        # Create task in ledger
        task_id = f"task_{datetime.now().timestamp()}"
        task_info = TaskInfo(
            task_id=task_id,
            agent_id=agent_info.agent_id,
            description=user_message,
            status=TaskStatus.PENDING,
            metadata={"task_type": task_type},
        )

        decision["decisions"]["task_created"] = True
        decision["decisions"]["task_id"] = task_id

        # Check if agent needs more information
        required_info = agent.identify_required_info(user_message, user_message)

        if required_info:
            # Need to gather information first
            decision["decisions"]["action"] = "gather_info"
            decision["analysis"]["required_information"] = list(required_info.keys())
            decision["reasoning"][
                "action"
            ] = f"Agent requires additional information before execution: {list(required_info.keys())}"
            rationale.append(decision)

            logger.info(f"Agent needs info: {list(required_info.keys())}")
            return {
                "current_agent_id": agent_info.agent_id,
                "current_task_id": task_id,
                "required_info": required_info,
                "collected_info": {},
                "pending_questions": [],
                "next_action": GraphDecision.GATHER_INFO.value,
                "agents": {**state.get("agents", {}), agent_info.agent_id: agent_info},
                "tool_allocations": {
                    **state.get("tool_allocations", {}),
                    agent_info.agent_id: (
                        [t.name for t in allocated_tools] if allocated_tools else []
                    ),  # Store names only
                },
                "task_ledger": {**state.get("task_ledger", {}), task_id: task_info},
                "decision_rationale": rationale,
            }
        else:
            # Can execute directly
            decision["decisions"]["action"] = "execute_agent"
            decision["analysis"]["required_information"] = []
            decision["reasoning"][
                "action"
            ] = "Agent has all information needed, proceeding to execution"
            rationale.append(decision)

            logger.info("Agent ready - executing task")
            return {
                "current_agent_id": agent_info.agent_id,
                "current_task_id": task_id,
                "next_action": GraphDecision.EXECUTE_AGENT.value,
                "agents": {**state.get("agents", {}), agent_info.agent_id: agent_info},
                "tool_allocations": {
                    **state.get("tool_allocations", {}),
                    agent_info.agent_id: (
                        [t.name for t in allocated_tools] if allocated_tools else []
                    ),  # Store names only
                },
                "task_ledger": {**state.get("task_ledger", {}), task_id: task_info},
                "decision_rationale": rationale,
            }

    except Exception as e:
        decision["decisions"]["action"] = "error"
        decision["decisions"]["agent_created"] = False
        decision["reasoning"]["error"] = f"Failed to create agent: {str(e)}"
        rationale.append(decision)

        logger.error(f"Failed to create agent: {e}")
        return {
            "next_action": GraphDecision.ERROR.value,
            "metadata": {**state.get("metadata", {}), "error": str(e)},
            "decision_rationale": rationale,
        }


async def information_gathering_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Multi-turn information gathering node.

    This node manages the conversational flow of collecting required information
    from the user across multiple turns, with context awareness.

    Returns only changed fields to properly leverage LangGraph reducers.
    """
    logger.info("Information gathering node processing")

    # Get messages - analyze last 3 for better context
    messages = state.get("messages", [])
    if not messages:
        logger.error("No messages in information gathering")
        return {"next_action": GraphDecision.ERROR.value}

    # Multi-message context: analyze last 3 messages for better understanding
    context_window = messages[-3:] if len(messages) >= 3 else messages
    latest_message = messages[-1]
    user_message = latest_message.get("content", "")

    logger.info(f"Analyzing {len(context_window)} messages for context")

    required_info_dict = state.get("required_info", {})

    # Convert back to RequiredInfoField if needed
    if required_info_dict and not isinstance(
        list(required_info_dict.values())[0], dict
    ):
        # Already in correct format
        pass
    else:
        # Need to handle as dict
        required_info_dict = {k: v for k, v in required_info_dict.items()}

    # Extract information from user's response using multi-message context
    extractor = InformationExtractor()

    extracted = extractor.extract_information(
        user_message=user_message,
        required_info=required_info_dict,
        conversation_history=[
            {"role": msg.get("role"), "content": msg.get("content")}
            for msg in context_window
        ],
    )

    # Merge with collected info - immutably
    new_collected_info = {**state.get("collected_info", {}), **extracted}

    # Check if all required info is collected
    missing_fields = set(required_info_dict.keys()) - set(new_collected_info.keys())

    if missing_fields:
        # Still need more information, ask questions
        # Access question from RequiredInfoField objects properly
        pending_questions = []
        for field in missing_fields:
            field_info = required_info_dict.get(field)
            if field_info and hasattr(field_info, "question"):
                pending_questions.append(field_info.question)
            elif isinstance(field_info, dict):
                pending_questions.append(
                    field_info.get("question", f"Please provide {field}")
                )
            else:
                pending_questions.append(f"Please provide {field}")

        # Generate response asking for missing info
        questions_text = "\n".join([f"- {q}" for q in pending_questions])
        response_message: Message = {
            "role": "assistant",
            "content": f"I need a bit more information to help you:\n{questions_text}",
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {"awaiting_info": True},
        }

        # Return only changed fields - messages will be appended via operator.add reducer
        return {
            "collected_info": new_collected_info,
            "pending_questions": pending_questions,
            "messages": [response_message],
            "next_action": GraphDecision.GATHER_INFO.value,
        }
    else:
        # All info collected, proceed to execution
        return {
            "collected_info": new_collected_info,
            "pending_questions": [],
            "next_action": GraphDecision.EXECUTE_AGENT.value,
        }


async def agent_executor_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Async node that executes the agent's task with collected information.

    This node is async to properly support agents with async execution
    capabilities without blocking the event loop.

    Returns only changed fields to properly leverage LangGraph reducers.
    """
    logger.info("Agent executor node executing")

    agent_id = state.get("current_agent_id")
    task_id = state.get("current_task_id")

    if not agent_id or not task_id:
        logger.error("No current agent or task set")
        return {"next_action": GraphDecision.ERROR.value}

    # Get agent info
    agent_info_dict = state["agents"].get(agent_id)
    if not agent_info_dict:
        logger.error(f"Agent {agent_id} not found in state")
        return {"next_action": GraphDecision.ERROR.value}

    # Get tool names (they're stored as names to avoid serialization issues)
    tool_names = state["tool_allocations"].get(agent_id, [])

    # Recreate tools from names using ToolAllocator
    # This avoids the serialization issue with LangGraph checkpointing
    tool_allocator = ToolAllocator()
    task_info = state["task_ledger"][task_id]
    task_description = task_info.description
    task_type = task_info.metadata.get("task_type", "general")

    # Re-allocate tools based on task type
    tools = tool_allocator.allocate_tools_for_task(
        task_type=task_type, task_description=task_description
    )

    # Recreate agent
    try:
        # agent_info_dict is already an AgentInfo object, not a dict
        agent_info = (
            agent_info_dict
            if isinstance(agent_info_dict, AgentInfo)
            else AgentInfo(**agent_info_dict)
        )
        config = AgentConfig(
            agent_type=agent_info.agent_type,
            required_tools=[],
        )
        agent = AgentFactory.create_agent(config, tools)

        # Get TaskInfo object from ledger
        task_info = state["task_ledger"][task_id]

        # Create SubAgentState for execution
        agent_state = SubAgentState(
            agent_id=agent_id,
            status=SubAgentStatus.PROCESSING,
            task=task_info.description,
            task_type=agent_info.agent_type,
            triggering_message=task_info.description,
            collected_info=state.get("collected_info", {}),
            required_info={},  # Already collected
            metadata={
                "user_id": state["user_id"],
                "persona": state["persona"],
            },
        )

        # Execute task - prefer async if available, else run sync in executor
        logger.info(f"Executing task for agent {agent_id}")
        if hasattr(agent, "execute_task_async"):
            result = await agent.execute_task_async(agent_state)
        else:
            # Run sync execute_task in thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, agent.execute_task, agent_state)
        logger.info(f"Task execution completed for agent {agent_id}")

        # Build updated task ledger with completed task - immutably
        updated_task_info = TaskInfo(
            task_id=task_info.task_id,
            agent_id=task_info.agent_id,
            description=task_info.description,
            status=TaskStatus.COMPLETED,
            dependencies=task_info.dependencies,
            created_at=task_info.created_at,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            result=result,
            error=None,
            metadata=task_info.metadata,
        )
        new_task_ledger = {**state.get("task_ledger", {}), task_id: updated_task_info}

        # Add result to messages
        response_message: Message = {
            "role": "assistant",
            "content": result,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "agent_id": agent_id,
                "task_id": task_id,
                "task_type": agent_info.agent_type,
            },
        }

        # Track agents and tools used in metadata - immutably
        state_metadata = state.get("metadata", {})
        agents_used = list(state_metadata.get("agents_used", []))
        tools_used = list(state_metadata.get("tools_used", []))

        if agent_id not in agents_used:
            agents_used.append(agent_id)

        # Add tools from this agent's allocation
        agent_tools = state["tool_allocations"].get(agent_id, [])
        for tool in agent_tools:
            if tool not in tools_used:
                tools_used.append(tool)

        new_metadata = {
            **state_metadata,
            "agents_used": agents_used,
            "tools_used": tools_used,
        }

        # Return only changed fields
        return {
            "task_ledger": new_task_ledger,
            "messages": [response_message],  # Will be appended via operator.add
            "metadata": new_metadata,
            "current_agent_id": None,
            "current_task_id": None,
            "required_info": {},
            "collected_info": {},
            "pending_questions": [],
            "next_action": GraphDecision.COMPLETE.value,
        }

    except Exception as e:
        logger.error(f"Agent execution failed: {e}")

        # Build updated task ledger with failed task - immutably
        updated_task_info = TaskInfo(
            task_id=task_info.task_id,
            agent_id=task_info.agent_id,
            description=task_info.description,
            status=TaskStatus.FAILED,
            dependencies=task_info.dependencies,
            created_at=task_info.created_at,
            started_at=task_info.started_at,
            completed_at=None,
            result=None,
            error=str(e),
            metadata=task_info.metadata,
        )
        new_task_ledger = {**state.get("task_ledger", {}), task_id: updated_task_info}

        return {
            "task_ledger": new_task_ledger,
            "next_action": GraphDecision.ERROR.value,
        }


async def general_response_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Async node that handles general conversation without agents.

    Uses async LLM service for better performance in async graph context.
    Returns only changed fields to properly leverage LangGraph reducers.
    """
    logger.info("General response node executing")

    user_message = state["messages"][-1]["content"]
    persona = state["persona"]

    # Use async LLM service for better performance
    llm_service = get_async_llm_service()

    try:
        # Use async generation for general responses
        result = await llm_service.generate_async(
            prompt=user_message,
            persona=persona,
        )

        response_message: Message = {
            "role": "assistant",
            "content": result,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {"general_conversation": True},
        }

        # Return only changed fields - messages will be appended via operator.add
        return {"messages": [response_message]}

    except Exception as e:
        logger.error(f"General response generation failed: {e}")
        response_message: Message = {
            "role": "assistant",
            "content": "I apologize, but I encountered an error. Please try again.",
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {"error": str(e)},
        }

        return {"messages": [response_message]}


async def error_handler_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Async node that handles errors gracefully.

    Returns only changed fields to properly leverage LangGraph reducers.
    """
    logger.error("Error handler node executing")

    error_message: Message = {
        "role": "assistant",
        "content": "I encountered an error processing your request. Please try rephrasing or try again later.",
        "timestamp": datetime.utcnow().isoformat(),
        "metadata": {"error": True},
    }

    # Return only changed fields - messages will be appended via operator.add
    return {
        "messages": [error_message],
        "current_agent_id": None,
        "current_task_id": None,
        "required_info": {},
        "collected_info": {},
        "pending_questions": [],
    }
