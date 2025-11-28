import asyncio
import logging
import time
from typing import Dict, Optional

from app.hermes.exceptions import AIServiceError
from app.hermes.models import GeminiResponse, ProcessRequestResult, UserIdentity
from app.shared.services.TTSService import TTSService

from .nodes.orchestration_graph import get_orchestration_graph
from .persistence import LegionPersistence
from .utils.conversation_memory import ConversationContextBuilder
from .utils.input_sanitizer import (
    redact_pii_for_logging,
    sanitize_user_input,
    validate_user_id,
)
from .utils.observability import get_observability
from .utils.task_timeout import DEFAULT_ORCHESTRATION_TIMEOUT, OrchestrationTimeoutError

# Threshold for when to apply conversation memory management
MAX_MESSAGES_BEFORE_MEMORY_MANAGEMENT = 50

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
        self._context_builder = ConversationContextBuilder()

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
            "agents": self._get_agents_explanation(agents_used),
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

    def _get_agents_explanation(self, agents_used: list) -> list:
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
        orchestration_timeout: Optional[float] = None,
    ) -> ProcessRequestResult:
        """
        Process a user request through the LangGraph orchestration pipeline.

        Args:
            text: User's input text
            user_identity: User identity information
            response_mode: Desired response mode ('text' or 'audio')
            persona: Which AI persona to use ('hermes' or 'prisma')
            orchestration_timeout: Optional timeout in seconds for the entire
                orchestration (default: 300s / 5 minutes). Set higher for
                complex multi-level workflows.

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
                "Processing request in LangGraph Legion mode for user %s...: %s",
                user_id[:8],
                redact_pii_for_logging(sanitized_text),
            )

            # Generate AI response using graph
            response_content, metadata = await self._generate_ai_response_with_graph(
                sanitized_text,
                user_id,
                persona,
                orchestration_timeout=orchestration_timeout,
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
                "LangGraph Legion request processed successfully for user %s", user_id
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
            logger.warning("Invalid request: %s", e)
            # Re-raise as is, will be handled by caller
            raise
        except Exception as e:
            logger.error("Error processing LangGraph Legion request: %s", e)
            raise AIServiceError(f"Failed to process Legion request: {str(e)}") from e

    async def chat(
        self,
        message: str,
        user_identity: UserIdentity,
        persona: str = "hermes",
        orchestration_timeout: Optional[float] = None,
    ) -> GeminiResponse:
        """
        Handle a chat message with LangGraph orchestration.

        Args:
            message: User's chat message
            user_identity: User identity information
            persona: Which AI persona to use ('hermes' or 'prisma')
            orchestration_timeout: Optional timeout in seconds for the entire
                orchestration (default: 300s / 5 minutes). Set higher for
                complex multi-level workflows.

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
                "Processing chat in LangGraph Legion mode for user %s...: %s",
                user_id[:8],
                redact_pii_for_logging(sanitized_message),
            )

            # Generate AI response using graph
            response_content, metadata = await self._generate_ai_response_with_graph(
                sanitized_message,
                user_id,
                persona,
                orchestration_timeout=orchestration_timeout,
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
                "LangGraph Legion chat processed successfully for user %s", user_id
            )
            return response

        except ValueError as e:
            logger.warning("Invalid chat request: %s", e)
            raise
        except Exception as e:
            logger.error("Error processing LangGraph Legion chat: %s", e)
            raise AIServiceError(f"Failed to process Legion chat: {str(e)}") from e

    async def _generate_ai_response_with_graph(
        self,
        prompt: str,
        user_id: str,
        persona: str = "hermes",
        resume_value=None,
        orchestration_timeout: Optional[float] = None,
    ) -> tuple[str, dict]:
        """
        Generate AI response using LangGraph orchestration.

        Supports interrupt/resume pattern for human-in-the-loop workflows.
        Includes configurable global timeout to prevent runaway orchestrations.

        Args:
            prompt: User's prompt (ignored if resume_value is provided)
            user_id: User identifier
            persona: AI persona
            resume_value: Optional value to pass when resuming from interrupt
            orchestration_timeout: Optional timeout in seconds (default: 300s / 5 min)

        Returns:
            Tuple of (response_content, metadata)

        Raises:
            AIServiceError: If graph execution fails
            OrchestrationTimeoutError: If execution exceeds timeout
        """
        # Initialize observability
        obs = get_observability()
        start_time = time.time()
        obs.log_orchestration_start(user_id, prompt, strategy=None)

        # Use provided timeout or default
        timeout = orchestration_timeout or DEFAULT_ORCHESTRATION_TIMEOUT
        logger.info(
            "Starting orchestration with timeout: %.0fs for user %s",
            timeout,
            user_id[:8],
        )

        # Track partial results for timeout recovery
        partial_result = None
        workers_completed = 0
        total_workers = 0

        try:
            config = {"configurable": {"thread_id": user_id}}

            # Use async persistence context manager - must wrap all checkpointer usage
            async with self._persistence.get_checkpointer() as checkpointer:

                # If resuming, pass the resume value
                if resume_value is not None:
                    logger.info("Resuming graph execution with value: %s", resume_value)
                    from langgraph.types import Command

                    inputs = Command(resume=resume_value)
                else:
                    # Initial input - check if we need conversation memory management
                    new_user_message = {"role": "user", "content": prompt}

                    # Get existing conversation from checkpointer if available
                    existing_messages = []
                    conversation_summaries = []

                    try:
                        # Try to get existing state to check conversation length
                        existing_state = await checkpointer.aget(config)
                        if existing_state:
                            existing_messages = existing_state.get("messages", [])
                            conversation_summaries = existing_state.get(
                                "conversation_summaries", []
                            )
                    except Exception as e:
                        logger.debug("No existing state found: %s", e)

                    # Apply conversation memory management if conversation is long
                    if len(existing_messages) >= MAX_MESSAGES_BEFORE_MEMORY_MANAGEMENT:
                        logger.info(
                            "Long conversation detected (%d messages), applying memory management",
                            len(existing_messages),
                        )
                        try:
                            managed_context = await self._context_builder.build_context(
                                messages=existing_messages,
                                current_query=prompt,
                            )

                            # Store new summary if created
                            if managed_context.get("conversation_summary"):
                                conversation_summaries.append(
                                    {
                                        "summary": managed_context[
                                            "conversation_summary"
                                        ],
                                        "messages_summarized": len(existing_messages)
                                        - len(
                                            managed_context.get("recent_messages", [])
                                        ),
                                        "created_at": managed_context.get("created_at"),
                                    }
                                )
                                logger.info(
                                    "Created conversation summary, keeping %d recent messages",
                                    len(managed_context.get("recent_messages", [])),
                                )
                        except Exception as e:
                            logger.warning("Failed to apply conversation memory: %s", e)
                            # Fall back to using all messages
                            managed_context = None
                    else:
                        managed_context = None

                    # Build inputs
                    inputs = {
                        "messages": [new_user_message],
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
                        # Initialize execution level tracking
                        "current_execution_level": 0,
                        "total_execution_levels": 1,
                        "level_results": {},
                        "fail_on_level_error": False,
                        # Conversation memory
                        "conversation_summaries": conversation_summaries,
                    }

                # Create graph with this checkpointer
                graph = get_orchestration_graph(checkpointer=checkpointer)

                async def execute_graph_with_tracking():
                    """Inner function to track execution for timeout handling."""
                    nonlocal partial_result, workers_completed, total_workers

                    result = None
                    async for chunk in graph.astream(
                        inputs, config=config, stream_mode="values"
                    ):
                        # Check if we hit an interrupt
                        if "__interrupt__" in chunk:
                            interrupt_data = chunk["__interrupt__"]
                            logger.info(
                                "Graph interrupted: %s",
                                interrupt_data.get("type", "unknown"),
                            )
                            return "INTERRUPTED", {
                                "interrupted": True,
                                "interrupt_data": interrupt_data,
                                "thread_id": user_id,
                            }

                        # Store the latest state for timeout recovery
                        result = chunk
                        partial_result = chunk

                        # Track worker progress
                        if "legion_results" in chunk:
                            workers_completed = len(chunk["legion_results"])
                        if "metadata" in chunk:
                            plans = chunk["metadata"].get("legion_worker_plans", [])
                            total_workers = len(plans)

                        logger.info(
                            "Graph step completed. Keys: %s",
                            list(chunk.keys()) if chunk else "None",
                        )

                    return result, None  # result, no interrupt

                # Execute with global timeout
                try:
                    execution_result = await asyncio.wait_for(
                        execute_graph_with_tracking(),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    elapsed = time.time() - start_time
                    logger.error(
                        "Orchestration timed out after %.1fs (limit: %.0fs). "
                        "Workers completed: %d/%d",
                        elapsed,
                        timeout,
                        workers_completed,
                        total_workers,
                    )

                    # Log timeout
                    obs.log_orchestration_complete(
                        user_id=user_id,
                        duration_ms=elapsed * 1000,
                        worker_count=workers_completed,
                        success=False,
                    )

                    # Return graceful timeout response with partial results
                    timeout_message = self._build_timeout_response(
                        partial_result,
                        workers_completed,
                        total_workers,
                        elapsed,
                        timeout,
                    )

                    return timeout_message, {
                        "interrupted": False,
                        "timed_out": True,
                        "timeout_seconds": timeout,
                        "elapsed_seconds": elapsed,
                        "workers_completed": workers_completed,
                        "total_workers": total_workers,
                        "partial_results_available": partial_result is not None,
                    }

                # Check for interrupt
                if isinstance(execution_result, tuple) and len(execution_result) == 2:
                    result, interrupt_info = execution_result
                    if interrupt_info is not None:
                        return "INTERRUPTED", interrupt_info
                else:
                    result = execution_result

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
                            "timed_out": False,
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

                        # Add level execution info if available
                        if "level_results" in result and result["level_results"]:
                            metadata["level_execution"] = {
                                "levels_completed": len(result["level_results"]),
                                "total_levels": result.get("total_execution_levels", 1),
                            }

                        # Log successful completion
                        duration_ms = (time.time() - start_time) * 1000
                        worker_count = len(metadata.get("agents_used", []))
                        obs.log_orchestration_complete(
                            user_id=user_id,
                            duration_ms=duration_ms,
                            worker_count=worker_count,
                            success=True,
                        )

                        return response_content, metadata

                # Fallback
                duration_ms = (time.time() - start_time) * 1000
                obs.log_orchestration_complete(
                    user_id=user_id,
                    duration_ms=duration_ms,
                    worker_count=0,
                    success=False,
                )
                return "I apologize, but I couldn't generate a proper response.", {}

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            obs.log_orchestration_complete(
                user_id=user_id,
                duration_ms=duration_ms,
                worker_count=0,
                success=False,
            )
            logger.error("Error executing orchestration graph: %s", e)
            raise AIServiceError(f"Graph execution failed: {str(e)}") from e

    def _build_timeout_response(
        self,
        partial_result: Optional[dict],
        workers_completed: int,
        total_workers: int,
        elapsed: float,
        timeout: float,
    ) -> str:
        """
        Build a graceful response when orchestration times out.

        Attempts to extract any useful partial results.

        Args:
            partial_result: Last captured state before timeout
            workers_completed: Number of workers that completed
            total_workers: Total number of planned workers
            elapsed: Actual time elapsed
            timeout: Configured timeout

        Returns:
            User-friendly timeout message with any available results
        """
        base_message = (
            f"I apologize, but the request took longer than expected "
            f"({elapsed:.0f}s, limit: {timeout:.0f}s)."
        )

        if partial_result is None:
            return f"{base_message} Please try a simpler request or try again later."

        # Try to extract partial results
        partial_content = []

        # Check for any completed worker results
        legion_results = partial_result.get("legion_results", {})
        if legion_results:
            successful_results = [
                r
                for r in legion_results.values()
                if r.get("status") == "success" and r.get("result")
            ]
            if successful_results:
                partial_content.append(
                    f"\n\nI was able to gather some information before timing out "
                    f"({len(successful_results)}/{total_workers} tasks completed):\n"
                )
                for result in successful_results[:3]:  # Limit to 3 results
                    role = result.get("role", "worker")
                    content = result.get("result", "")[:500]  # Truncate long results
                    partial_content.append(f"\n**{role.title()}**: {content}")

        # Check level results
        level_results = partial_result.get("level_results", {})
        if level_results and not partial_content:
            total_success = sum(
                level.get("success_count", 0) for level in level_results.values()
            )
            if total_success > 0:
                partial_content.append(
                    f"\n\n{total_success} subtasks completed across "
                    f"{len(level_results)} execution level(s) before timeout."
                )

        if partial_content:
            return base_message + "".join(partial_content)

        return (
            f"{base_message} "
            f"{workers_completed}/{total_workers} workers had started processing. "
            "Please try a simpler request or try again later."
        )

    def generate_tts(self, text: str) -> tuple[Optional[str], Optional[str]]:
        """
        Generate Text-to-Speech audio.

        Args:
            text: Text to synthesize

        Returns:
            Tuple of (cloud_url, tts_provider)
        """
        try:
            result = self._tts_service.generate_audio(text)
            cloud_url = result.get("cloud_url")
            return cloud_url, self._tts_service.tts_provider
        except (ValueError, RuntimeError) as e:
            logger.error("TTS generation failed: %s", e)
            return None, None

    async def resume_execution(
        self,
        user_identity: UserIdentity,
        resume_value: dict,
        persona: str = "hermes",
        orchestration_timeout: Optional[float] = None,
    ) -> GeminiResponse:
        """
        Resume graph execution from an interrupt.

        This is called after user approves/modifies an interrupt,
        passing their response back into the graph to continue execution.

        Args:
            user_identity: User identity information
            resume_value: User's response (e.g., {"action": "approve"})
            persona: AI persona
            orchestration_timeout: Optional timeout in seconds for the remaining
                orchestration (default: 300s / 5 minutes)

        Returns:
            GeminiResponse with continued execution or another interrupt

        Raises:
            AIServiceError: If resumption fails
        """

        try:
            user_id = validate_user_id(user_identity.user_id)

            logger.info("Resuming conversation for user %s", user_id)

            # Resume from checkpoint with user's response
            response_content, metadata = await self._generate_ai_response_with_graph(
                prompt="",
                user_id=user_id,
                persona=persona,
                resume_value=resume_value,
                orchestration_timeout=orchestration_timeout,
            )

            # Check if we hit another interrupt
            if response_content == "INTERRUPTED" and metadata.get("interrupted"):
                return GeminiResponse(
                    content="I need your approval to proceed.",
                    user_id=user_id,
                    prompt="[RESUMED]",
                    metadata={
                        "legion_mode": True,
                        "langgraph_enabled": True,
                        **metadata,
                    },
                )

            # Execution completed
            return GeminiResponse(
                content=response_content,
                user_id=user_id,
                prompt="[RESUMED]",
                metadata={
                    "legion_mode": True,
                    "langgraph_enabled": True,
                    **metadata,
                },
            )

        except Exception as e:
            logger.error("Error resuming graph: %s", e)
            raise AIServiceError(f"Failed to resume execution: {str(e)}") from e

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

            logger.info("No conversation history found for user %s", user_id)
            return []

        except (ValueError, RuntimeError) as e:
            logger.error("Failed to retrieve conversation history: %s", e)
            return []
