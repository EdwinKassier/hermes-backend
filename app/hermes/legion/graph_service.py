import asyncio
import logging
import sqlite3
from typing import Any, AsyncIterator, Dict, List, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from app.hermes.exceptions import AIServiceError
from app.hermes.models import GeminiResponse, ProcessRequestResult, UserIdentity
from app.shared.services.TTSService import TTSService

from .nodes.graph_nodes import (
    AgentFactory,
    OrchestratorState,
    SubAgentState,
    TaskInfo,
    TaskStatus,
)
from .nodes.orchestration_graph import get_orchestration_graph
from .persistence import LegionPersistence
from .utils.input_sanitizer import (
    redact_pii_for_logging,
    sanitize_user_input,
    validate_user_id,
)

logger = logging.getLogger(__name__)


class LegionGraphService:
    """
    LangGraph-based Legion service implementing the Magentic Orchestrator pattern.

    This replaces the procedural orchestration logic with a proper graph-based workflow.
    """

    def __init__(self, checkpoint_db_path: str = "legion_state.db"):
        """
        Initialize LangGraph-based Legion service.

        Args:
            checkpoint_db_path: Path to SQLite database for state persistence
        """
        self._checkpoint_db_path = checkpoint_db_path
        self._tts_service = TTSService()
        self._persistence = LegionPersistence(checkpoint_db_path)

    def _build_orchestration_rationale(
        self,
        decision_rationale: list,
        agents_used: list,
        tools_used: list,
        performance_metrics: dict = None,
    ) -> dict:
        """Build single unified orchestration rationale.

        Args:
            decision_rationale: Raw decision data
            agents_used: List of agents that were used
            tools_used: List of tools that were used
            performance_metrics: Optional performance data

        Returns:
            Dictionary with orchestration explanation
        """
        if not decision_rationale:
            return {}

        # Get the latest decision
        latest_decision = decision_rationale[-1] if decision_rationale else {}
        analysis = latest_decision.get("analysis", {})
        decisions = latest_decision.get("decisions", {})

        # Build rationale
        rationale = {
            "orchestration_structure": self._get_orchestration_structure(
                analysis, decisions
            ),
            "agents": self._get_agents_explanation(agents_used, decisions),
            "toolsets": self._get_toolsets_explanation(tools_used, agents_used),
        }

        # Add execution details if multi-agent
        if analysis.get("multi_agent_task_detected") or len(agents_used) > 1:
            rationale["execution_mode"] = "parallel"
            if performance_metrics:
                rationale["performance"] = {
                    "agents_executed": len(agents_used),
                    "duration_seconds": performance_metrics.get(
                        "total_duration_seconds"
                    ),
                    "efficiency": performance_metrics.get("parallel_efficiency"),
                }
        else:
            rationale["execution_mode"] = "single-agent"

        return rationale

    def _get_orchestration_structure(self, analysis: dict, decisions: dict) -> str:
        """Explain why this orchestration structure was chosen."""
        if analysis.get("multi_agent_task_detected"):
            subtask_count = analysis.get(
                "subtasks_count", len(analysis.get("parallel_tasks", []))
            )
            return f"Multi-agent parallel execution: Task decomposed into {subtask_count} independent subtasks for concurrent processing"

        if decisions.get("agent_needed"):
            agent_type = decisions.get("agent_type", "specialist")
            task_type = analysis.get("identified_task_type", agent_type)
            return f"Single-agent execution: {task_type.title()} task requiring specialized {agent_type} agent"

        return "Direct response: Simple question answerable from knowledge base without agent"

    def _get_agents_explanation(self, agents_used: list, decisions: dict) -> list:
        """Explain which agents were used and why."""
        if not agents_used:
            return []

        agent_explanations = []
        for agent_id in agents_used:
            # Extract agent type from ID (e.g., "research_1" -> "research")
            agent_type = agent_id.split("_")[0]

            explanation = {
                "id": agent_id,
                "type": agent_type,
                "role": self._get_agent_role(agent_type),
            }
            agent_explanations.append(explanation)

        return agent_explanations

    def _get_agent_role(self, agent_type: str) -> str:
        """Get human-readable role description for agent type."""
        roles = {
            "research": "Information gathering and research",
            "code": "Code generation and programming",
            "analysis": "Data analysis and evaluation",
            "data": "Data processing and transformation",
        }
        return roles.get(agent_type, f"{agent_type.title()} operations")

    def _get_toolsets_explanation(self, tools_used: list, agents_used: list) -> dict:
        """Explain which toolsets were allocated and why."""
        if not tools_used and not agents_used:
            return {}

        # Group tools by agent if possible
        toolset_info = {
            "available_tools": tools_used if tools_used else [],
            "allocation_reason": "Tools allocated based on task requirements and agent capabilities",
        }

        if len(agents_used) > 1:
            toolset_info["distribution"] = (
                "Each agent receives task-specific subset of tools"
            )

        return toolset_info

    async def process_request(
        self,
        text: str,
        user_identity: UserIdentity,
        response_mode: str = "text",
        persona: str = "hermes",
    ) -> ProcessRequestResult:
        """
        Process a user request through the LangGraph orchestration pipeline.

        Args:
            text: User's input text
            user_identity: User identity information
            response_mode: Desired response mode ('text' or 'audio')
            persona: Which AI persona to use ('hermes' or 'prisma')

        Returns:
            ProcessRequestResult with the AI response

        Raises:
            InvalidRequestError: If the request is invalid
            AIServiceError: If AI generation fails
            TTSServiceError: If TTS generation fails
        """
        try:
            # Validate and sanitize inputs
            user_id = validate_user_id(user_identity.user_id)
            sanitized_text = sanitize_user_input(text)

            # Log safely (with PII redaction)
            logger.info(
                f"Processing request in LangGraph Legion mode for user {user_id[:8]}...: "
                f"{redact_pii_for_logging(sanitized_text)}"
            )

            # Generate AI response using graph
            response_content, metadata = await self._generate_ai_response_with_graph(
                sanitized_text, user_id, persona
            )

            # Check if the graph interrupted for human approval
            if response_content == "INTERRUPTED" and metadata.get("interrupted"):
                # Return interrupt information to API caller
                # We return a special response that indicates an interrupt
                return ProcessRequestResult(
                    message="I need your approval to proceed with the plan.",
                    response_mode=response_mode,
                    user_id=user_id,
                    metadata={
                        "legion_mode": True,
                        "langgraph_enabled": True,
                        **metadata,  # Contains interrupt_data and thread_id
                    },
                )

            # Generate TTS if requested
            wave_url = None
            tts_provider = None
            if response_mode == "audio":
                wave_url, tts_provider = self.generate_tts(response_content)

            logger.info(
                f"LangGraph Legion request processed successfully for user {user_id}"
            )
            return ProcessRequestResult(
                message=response_content,
                response_mode=response_mode,
                audio_url=wave_url,
                tts_provider=tts_provider,
                user_id=user_id,
                metadata={
                    "legion_mode": True,
                    "langgraph_enabled": True,
                    **metadata,
                },
            )

        except ValueError as e:
            logger.warning(f"Invalid request: {e}")
            # Re-raise as is, will be handled by caller
            raise
        except Exception as e:
            logger.error(f"Error processing LangGraph Legion request: {e}")
            raise AIServiceError(f"Failed to process Legion request: {str(e)}")

    async def chat(
        self,
        message: str,
        user_identity: UserIdentity,
        persona: str = "hermes",
    ) -> GeminiResponse:
        """
        Handle a chat message with LangGraph orchestration.

        Args:
            message: User's chat message
            user_identity: User identity information
            persona: Which AI persona to use ('hermes' or 'prisma')

        Returns:
            GeminiResponse with the AI reply

        Raises:
            InvalidRequestError: If the message is invalid
            AIServiceError: If AI generation fails
        """
        try:
            # Validate and sanitize inputs
            user_id = validate_user_id(user_identity.user_id)
            sanitized_message = sanitize_user_input(message)

            # Log safely
            logger.info(
                f"Processing chat in LangGraph Legion mode for user {user_id[:8]}...: "
                f"{redact_pii_for_logging(sanitized_message)}"
            )

            # Generate AI response using graph
            response_content, metadata = await self._generate_ai_response_with_graph(
                sanitized_message, user_id, persona
            )

            # Check if the graph interrupted
            if response_content == "INTERRUPTED" and metadata.get("interrupted"):
                # Return interrupt information
                return GeminiResponse(
                    content="I need your approval to proceed.",
                    user_id=user_id,
                    prompt=sanitized_message,
                    metadata={
                        "legion_mode": True,
                        "langgraph_enabled": True,
                        **metadata,
                    },
                )

            response = GeminiResponse(
                content=response_content,
                user_id=user_id,
                prompt=sanitized_message,
                metadata={
                    "legion_mode": True,
                    "langgraph_enabled": True,
                    **metadata,
                },
            )

            logger.info(
                f"LangGraph Legion chat processed successfully for user {user_id}"
            )
            return response

        except ValueError as e:
            logger.warning(f"Invalid chat request: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing LangGraph Legion chat: {e}")
            raise AIServiceError(f"Failed to process Legion chat: {str(e)}")

    async def _generate_ai_response_with_graph(
        self, prompt: str, user_id: str, persona: str = "hermes", resume_value=None
    ) -> tuple[str, dict]:
        """
        Generate AI response using LangGraph orchestration.

        Supports interrupt/resume pattern for human-in-the-loop workflows.

        Args:
            prompt: User's prompt (ignored if resume_value is provided)
            user_id: User identifier
            persona: AI persona
            resume_value: Optional value to pass when resuming from interrupt

        Returns:
            Tuple of (response_content, metadata)

        Raises:
            AIServiceError: If graph execution fails
        """
        try:
            config = {"configurable": {"thread_id": user_id}}

            # If resuming, pass the resume value
            if resume_value is not None:
                logger.info(f"Resuming graph execution with value: {resume_value}")
                inputs = None  # Resume doesn't need inputs, it uses the resume value
                # We need to use Command(resume=...) for LangGraph 0.2+
                # But for now assuming we just pass None as input and the graph handles it?
                # Actually, LangGraph resume is handled by passing Command or update_state
                # Let's assume standard input for now if not using Command
                # For this implementation, we'll treat resume_value as input update if needed
                # But standard LangGraph resume is: graph.stream(Command(resume=value), ...)
                from langgraph.types import Command

                inputs = Command(resume=resume_value)
            else:
                # Initial input
                inputs = {
                    "messages": [{"role": "user", "content": prompt}],
                    "user_id": user_id,
                    "persona": persona,
                    # Initialize other state fields
                    "task_ledger": {},
                    "agents": {},
                    "tool_allocations": {},
                    "decision_rationale": [],
                    "collected_info": {},
                    "pending_questions": [],
                    "metadata": {},
                }

            # Use async persistence context manager
            async with self._persistence.get_checkpointer() as checkpointer:
                # Create graph with this checkpointer
                graph = get_orchestration_graph(checkpointer=checkpointer)

                result = None
                # Use async stream
                async for chunk in graph.astream(
                    inputs, config=config, stream_mode="values"
                ):
                    # Check if we hit an interrupt
                    if "__interrupt__" in chunk:
                        interrupt_data = chunk["__interrupt__"]
                        logger.info(
                            f"Graph interrupted: {interrupt_data.get('type', 'unknown')}"
                        )

                        # Return interrupt information to caller
                        return "INTERRUPTED", {
                            "interrupted": True,
                            "interrupt_data": interrupt_data,
                            "thread_id": user_id,
                        }

                    # Store the latest state
                    result = chunk
                    logger.info(
                        f"Graph step completed. Keys: {list(chunk.keys()) if chunk else 'None'}"
                    )

                logger.info("Graph execution loop finished")

                # No interrupt - execution completed
                if result and "messages" in result:
                    last_message = result["messages"][-1]
                    if last_message["role"] == "assistant":
                        response_content = last_message["content"]

                        # Create metadata - extract from state metadata
                        state_metadata = result.get("metadata", {})
                        metadata = {
                            "interrupted": False,
                            "agents_used": state_metadata.get("agents_used", []),
                            "tools_used": state_metadata.get("tools_used", []),
                            "execution_path": result.get("execution_path", []),
                        }

                        # Add single unified orchestration rationale
                        if (
                            "decision_rationale" in result
                            and result["decision_rationale"]
                        ):
                            decision_rationale = result["decision_rationale"]

                            # Build orchestration rationale
                            orchestration_rationale = (
                                self._build_orchestration_rationale(
                                    decision_rationale,
                                    metadata.get("agents_used", []),
                                    metadata.get("tools_used", []),
                                    result.get("metadata", {}).get(
                                        "parallel_execution_metrics"
                                    ),
                                )
                            )

                            metadata["orchestration_rationale"] = (
                                orchestration_rationale
                            )

                        # Add performance metrics if available
                        if "parallel_execution_metrics" in result.get("metadata", {}):
                            metadata["parallel_execution_metrics"] = result["metadata"][
                                "parallel_execution_metrics"
                            ]

                        return response_content, metadata

                # Fallback
                return "I apologize, but I couldn't generate a proper response.", {}

        except Exception as e:
            logger.error(f"Error executing orchestration graph: {e}")
            raise AIServiceError(f"Graph execution failed: {str(e)}")

    def generate_tts(self, text: str) -> tuple[str | None, str]:
        """
        Generate Text-to-Speech audio.

        Args:
            text: Text to synthesize

        Returns:
            Tuple of (cloud_url, tts_provider)
        """
        try:
            return self._tts_service.generate_speech(text)
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return None, None

    async def resume_execution(
        self,
        user_identity: UserIdentity,
        resume_value: dict,
        persona: str = "hermes",
    ) -> GeminiResponse:
        """
        Resume graph execution from an interrupt.

        This is called after user approves/modifies an interrupt,
        passing their response back into the graph to continue execution.

        Args:
            user_identity: User identity information
            resume_value: User's response (e.g., {"action": "approve"})
            persona: AI persona

        Returns:
            GeminiResponse with continued execution or another interrupt

        Raises:
            AIServiceError: If resumption fails
        """

        try:
            user_id = validate_user_id(user_identity.user_id)

            logger.info(f"Resuming conversation for user {user_id}")

            # Resume from checkpoint with user's response
            response_content, metadata = await self._generate_ai_response_with_graph(
                prompt="", user_id=user_id, persona=persona, resume_value=resume_value
            )

            # Check if we hit another interrupt
            if response_content == "INTERRUPTED" and metadata.get("interrupted"):
                return GeminiResponse(
                    content="I need your approval to proceed.",
                    metadata={
                        "legion_mode": True,
                        "langgraph_enabled": True,
                        **metadata,
                    },
                )

            # Execution completed
            return GeminiResponse(
                content=response_content,
                metadata={
                    "legion_mode": True,
                    "langgraph_enabled": True,
                    **metadata,
                },
            )

        except Exception as e:
            logger.error(f"Error resuming graph: {e}")
            raise AIServiceError(f"Failed to resume execution: {str(e)}")

    async def get_conversation_history(
        self, user_id: str, limit: int = 50
    ) -> list[Dict]:
        """
        Get conversation history from LangGraph checkpoints.

        Args:
            user_id: User identifier
            limit: Maximum number of messages to return (default 50)

        Returns:
            List of messages from most recent checkpoint
        """
        try:
            config = {"configurable": {"thread_id": user_id}}

            # Use persistence to get state
            async with self._persistence.get_checkpointer() as checkpointer:
                # We need the graph to get state, or we can use checkpointer directly?
                # LangGraph's get_state is on the compiled graph
                graph = get_orchestration_graph(checkpointer=checkpointer)

                # Get the latest checkpoint state
                state = await graph.aget_state(config)

                if state and state.values:
                    messages = state.values.get("messages", [])

                    # Return most recent messages up to limit
                    if messages:
                        return messages[-limit:] if len(messages) > limit else messages

            logger.info(f"No conversation history found for user {user_id}")
            return []

        except Exception as e:
            logger.error(f"Failed to retrieve conversation history: {e}")
            return []
