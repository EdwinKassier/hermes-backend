"""LangGraph nodes for orchestration workflow."""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List

from app.shared.utils.service_loader import get_gemini_service

from ..agents.factory import AgentFactory
from ..models import SubAgentState, SubAgentStatus
from ..orchestrator import InformationExtractor, IntentDetector, TaskIdentifier
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


def orchestrator_node(state: OrchestratorState) -> OrchestratorState:
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

    import asyncio

    routing_decision = asyncio.run(
        routing_intel.analyze(
            message=user_message,
            conversation_history=messages[:-1],  # Exclude current message
            current_agent_context=agent_context,
        )
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
            **state,
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
            **state,
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
                **state,
                "next_action": "legion_orchestrate",
                "decision_rationale": decision_rationale,
                "metadata": state_metadata,
            }

        # Single agent orchestration - create agent and route to execution
        task_identifier = TaskIdentifier()
        task_type = task_identifier.identify_task_type(user_message)

        if not task_type:
            logger.warning(
                "Orchestration needed but no task type identified, falling back to general response"
            )
            decision_rationale.append(current_decision)
            return {
                **state,
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
            **state,
            "next_action": GraphDecision.ERROR.value,
            "decision_rationale": decision_rationale,
            "metadata": state_metadata,
        }

    # 1. Check for cancellation first
    intent_detector = IntentDetector()
    if intent_detector.is_cancellation_intent(user_message):
        return _handle_cancellation(state, current_decision, decision_rationale)

    # 2. Check for topic change (if we have an active task)
    current_task_id = state.get("current_task_id")
    if current_task_id and current_task_id in state.get("task_ledger", {}):
        task_info = state["task_ledger"][current_task_id]

        # Only check topic change if task is active (not completed/failed)
        if task_info.status in [TaskStatus.IN_PROGRESS, TaskStatus.AWAITING_INPUT]:
            from ..utils.topic_change_detector import TopicChangeDetector

            detector = TopicChangeDetector()
            detection_result = asyncio.run(
                detector.detect_topic_change(
                    current_task_description=task_info.description,
                    new_user_message=user_message,
                    conversation_history=messages,
                )
            )

            # If topic changed with high confidence, trigger replan
            if detector.should_trigger_replan(detection_result):
                logger.info(
                    f"Topic change detected (confidence: {detection_result['confidence']:.2f}): "
                    f"{detection_result['reason']}"
                )

                current_decision["analysis"]["topic_change_detected"] = True
                current_decision["analysis"]["new_topic"] = detection_result.get(
                    "new_topic"
                )
                current_decision["analysis"]["confidence"] = detection_result[
                    "confidence"
                ]
                current_decision["decisions"]["action"] = "replan"
                current_decision["reasoning"]["replan_reason"] = detection_result[
                    "reason"
                ]

                rationale = decision_rationale + [current_decision]

                return {
                    "decision_rationale": rationale,
                    "next_action": GraphDecision.REPLAN.value,
                    "metadata": {
                        **state.get("metadata", {}),
                        "topic_change": detection_result,
                    },
                }

    # 3. Check if we have an agent awaiting input
    current_agent_id = state.get("current_agent_id")
    if current_agent_id:
        return _handle_active_agent(
            state, current_agent_id, current_decision, decision_rationale
        )

    # 4. Check for multi-agent tasks
    # Fix local import
    from ..parallel.task_decomposer import ParallelTaskDecomposer

    decomposer = ParallelTaskDecomposer()

    if decomposer.is_multi_agent_task(user_message):
        return _handle_multi_agent(state, current_decision, decision_rationale)

    # 5. Identify task type
    task_identifier = TaskIdentifier()
    task_type = task_identifier.identify_task_type(user_message)

    current_decision["analysis"]["user_message"] = (
        user_message[:100] + "..." if len(user_message) > 100 else user_message
    )
    current_decision["analysis"]["identified_task_type"] = task_type

    if not task_type:
        return _handle_general_conversation(state, current_decision, decision_rationale)

    # 6. Check if it's a simple factual question vs complex task
    if _is_factual_question(user_message, task_type):
        return _handle_factual_question(
            state, task_type, current_decision, decision_rationale
        )

    # 7. Task identified - create agent dynamically
    return _handle_new_task(
        state, user_message, task_type, current_decision, decision_rationale
    )


def _handle_empty_message(state, decision, rationale):
    """Handle case with no messages."""
    decision["decisions"]["action"] = "complete"
    decision["reasoning"]["action"] = "No messages to process"
    rationale.append(decision)

    return {
        **state,
        "next_action": GraphDecision.COMPLETE.value,
        "decision_rationale": rationale,
    }


def _handle_cancellation(state, decision, rationale):
    """Handle user cancellation request."""
    decision["decisions"]["action"] = "cancel"
    decision["reasoning"]["action"] = "User requested cancellation"
    rationale.append(decision)

    logger.info("Cancellation requested - stopping all tasks")

    # Cancel all active tasks
    task_ledger = state.get("task_ledger", {})
    for task_id, task_info in task_ledger.items():
        if task_info.status in [TaskStatus.IN_PROGRESS, TaskStatus.AWAITING_INPUT]:
            task_info.status = TaskStatus.CANCELLED

    return {
        **state,
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

    # Cancel current task
    task_ledger = state.get("task_ledger", {})
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
        **state,
        "messages": state.get("messages", []) + [cancel_msg],
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
        **state,
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
        **state,
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
        **state,
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
        **state,
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
                **state,
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
                **state,
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
                "orchestration_reasoning": decision,
            }

    except Exception as e:
        decision["decisions"]["action"] = "error"
        decision["decisions"]["agent_created"] = False
        decision["reasoning"]["error"] = f"Failed to create agent: {str(e)}"
        rationale.append(decision)

        logger.error(f"Failed to create agent: {e}")
        return {
            **state,
            "next_action": GraphDecision.ERROR.value,
            "metadata": {**state.get("metadata", {}), "error": str(e)},
            "decision_rationale": rationale,
        }


def information_gathering_node(state: OrchestratorState) -> OrchestratorState:
    """
    Multi-turn information gathering node.

    This node manages the conversational flow of collecting required information
    from the user across multiple turns, with context awareness.
    """
    logger.info("Information gathering node processing")

    # Get messages - analyze last 3 for better context
    messages = state.get("messages", [])
    if not messages:
        logger.error("No messages in information gathering")
        return {**state, "next_action": GraphDecision.ERROR.value}

    # Multi-message context: analyze last 3 messages for better understanding
    context_window = messages[-3:] if len(messages) >= 3 else messages
    latest_message = messages[-1]
    user_message = latest_message.get("content", "")

    logger.info(f"Analyzing {len(context_window)} messages for context")

    required_info_dict = state.get("required_info", {})
    collected_info = state.get("collected_info", {})

    # Handle RequiredInfoField objects (they get serialized/deserialized)
    from ..orchestrator import InformationExtractor

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

    # Build context-aware prompt from last N messages
    context_messages = "\n".join(
        [
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')[:200]}"
            for msg in context_window
        ]
    )

    extracted = extractor.extract_information(
        user_message=user_message,
        required_info=required_info_dict,
        conversation_history=[
            {"role": msg.get("role"), "content": msg.get("content")}
            for msg in context_window
        ],
    )

    # Merge with collected info
    collected_info = state.get("collected_info", {})
    collected_info.update(extracted)
    state["collected_info"] = collected_info

    # Check if all required info is collected
    missing_fields = set(required_info_dict.keys()) - set(collected_info.keys())

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

        state["pending_questions"] = pending_questions

        # Generate response asking for missing info
        questions_text = "\n".join([f"- {q}" for q in pending_questions])
        response_message: Message = {
            "role": "assistant",
            "content": f"I need a bit more information to help you:\n{questions_text}",
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {"awaiting_info": True},
        }
        state["messages"] = state.get("messages", []) + [response_message]

        # Stay in gathering mode
        state["next_action"] = GraphDecision.GATHER_INFO.value
    else:
        # All info collected, proceed to execution
        state["pending_questions"] = []
        state["next_action"] = GraphDecision.EXECUTE_AGENT.value

    return state


def agent_executor_node(state: OrchestratorState) -> OrchestratorState:
    """
    Node that executes the agent's task with collected information.
    """
    logger.info("Agent executor node executing")

    agent_id = state.get("current_agent_id")
    task_id = state.get("current_task_id")

    if not agent_id or not task_id:
        logger.error("No current agent or task set")
        state["next_action"] = GraphDecision.ERROR.value
        return state

    # Get agent info
    agent_info_dict = state["agents"].get(agent_id)
    if not agent_info_dict:
        logger.error(f"Agent {agent_id} not found in state")
        state["next_action"] = GraphDecision.ERROR.value
        return state

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

        # Update task status
        task_info.status = TaskStatus.IN_PROGRESS
        task_info.started_at = datetime.utcnow()
        state["task_ledger"][task_id] = task_info

        # Execute task (all agents use sync execute_task)
        logger.info(f"Executing task for agent {agent_id}")
        result = agent.execute_task(agent_state)
        logger.info(f"Task execution completed for agent {agent_id}")

        # Update task as completed
        task_info.status = TaskStatus.COMPLETED
        task_info.completed_at = datetime.utcnow()
        task_info.result = result
        state["task_ledger"][task_id] = task_info

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
        state["messages"] = state.get("messages", []) + [response_message]

        # Track agents and tools used in metadata
        state_metadata = state.get("metadata", {})
        agents_used = state_metadata.get("agents_used", [])
        tools_used = state_metadata.get("tools_used", [])

        if agent_id not in agents_used:
            agents_used.append(agent_id)

        # Add tools from this agent's allocation
        agent_tools = state["tool_allocations"].get(agent_id, [])
        for tool in agent_tools:
            if tool not in tools_used:
                tools_used.append(tool)

        state["metadata"] = {
            **state_metadata,
            "agents_used": agents_used,
            "tools_used": tools_used,
        }

        # Clear current context
        state["current_agent_id"] = None
        state["current_task_id"] = None
        state["required_info"] = {}
        state["collected_info"] = {}
        state["pending_questions"] = []

        state["next_action"] = GraphDecision.COMPLETE.value

    except Exception as e:
        logger.error(f"Agent execution failed: {e}")

        # Update task as failed
        task_info = state["task_ledger"][task_id]
        task_info.status = TaskStatus.FAILED
        task_info.error = str(e)
        state["task_ledger"][task_id] = task_info

        state["next_action"] = GraphDecision.ERROR.value

    return state


def general_response_node(state: OrchestratorState) -> OrchestratorState:
    """
    Node that handles general conversation without agents.
    Uses RAG to access persona knowledge base.
    """
    logger.info("General response node executing")

    user_message = state["messages"][-1]["content"]
    persona = state["persona"]
    user_id = state["user_id"]

    # Generate response using Gemini (without RAG for general responses)
    # RAG is only for persona knowledge base queries, not general conversation
    gemini_service = get_gemini_service()

    try:
        # Use standard generation for general responses
        result = gemini_service.generate_gemini_response(
            prompt=user_message,
            persona=persona,
            user_id=user_id,
        )

        response_message: Message = {
            "role": "assistant",
            "content": result,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {"general_conversation": True, "used_rag": True},
        }
        state["messages"] = state.get("messages", []) + [response_message]

    except Exception as e:
        logger.error(f"General response generation failed: {e}")
        response_message: Message = {
            "role": "assistant",
            "content": "I apologize, but I encountered an error. Please try again.",
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {"error": str(e)},
        }
        state["messages"] = state.get("messages", []) + [response_message]

    return state


def error_handler_node(state: OrchestratorState) -> OrchestratorState:
    """
    Node that handles errors gracefully.
    """
    logger.error("Error handler node executing")

    error_message: Message = {
        "role": "assistant",
        "content": "I encountered an error processing your request. Please try rephrasing or try again later.",
        "timestamp": datetime.utcnow().isoformat(),
        "metadata": {"error": True},
    }
    state["messages"] = state.get("messages", []) + [error_message]

    # Clean up state
    state["current_agent_id"] = None
    state["current_task_id"] = None
    state["required_info"] = {}
    state["collected_info"] = {}
    state["pending_questions"] = []

    return state
