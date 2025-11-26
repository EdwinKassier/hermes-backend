"""LangGraph-based Legion service implementation."""

import logging
from datetime import datetime
from typing import Optional

from app.shared.utils.service_loader import get_gemini_service, get_tts_service

from ..exceptions import AIServiceError, InvalidRequestError, TTSServiceError
from ..models import GeminiResponse, ProcessRequestResult, ResponseMode, UserIdentity
from .nodes import get_orchestration_graph
from .state import Message, OrchestratorState

logger = logging.getLogger(__name__)


class LegionGraphService:
    """
    LangGraph-based Legion service implementing the Magentic Orchestrator pattern.

    This replaces the procedural orchestration logic with a proper graph-based workflow.
    """

    def __init__(self, checkpoint_db_path: str = ":memory:"):
        """
        Initialize LangGraph-based Legion service.

        Args:
            checkpoint_db_path: Path to SQLite database for state persistence
        """
        self._gemini_service = None
        self._tts_service = None
        self._graph = None
        self._checkpoint_db_path = checkpoint_db_path
        self._checkpointer = None
        self._checkpointer_cm = None

    @property
    def gemini_service(self):
        """Lazy load Gemini service."""
        if self._gemini_service is None:
            self._gemini_service = get_gemini_service()
        return self._gemini_service

    @property
    def tts_service(self):
        """Lazy load TTS service."""
        if self._tts_service is None:
            self._tts_service = get_tts_service()
        return self._tts_service

    def get_graph(self):
        """Get or create orchestration graph with checkpointer."""
        if self._graph is None:
            if self._checkpoint_db_path == ":memory:":
                from langgraph.checkpoint.memory import MemorySaver

                checkpointer = MemorySaver()
            else:
                import sqlite3

                from langgraph.checkpoint.sqlite import SqliteSaver

                # Create persistent SQLite connection
                conn = sqlite3.connect(
                    self._checkpoint_db_path, check_same_thread=False, timeout=30.0
                )

                # Enable WAL mode for better concurrency
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.commit()

                checkpointer = SqliteSaver(conn)

            self._graph = get_orchestration_graph(checkpointer=checkpointer)
        return self._graph

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
            agent_type = agent_id.split("_")[0] if "_" in agent_id else agent_id

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

    def process_request(
        self,
        text: str,
        user_identity: UserIdentity,
        response_mode: ResponseMode = ResponseMode.TEXT,
        persona: str = "hermes",
    ) -> ProcessRequestResult:
        """
        Process a user request through the LangGraph orchestration pipeline.

        Args:
            text: User's input text
            user_identity: User identity information
            response_mode: How to return the response (text or TTS)
            persona: Which AI persona to use ('hermes' or 'prisma')

        Returns:
            ProcessRequestResult with the AI response

        Raises:
            InvalidRequestError: If the request is invalid
            AIServiceError: If AI generation fails
            TTSServiceError: If TTS generation fails
        """
        from .utils.input_sanitizer import (
            redact_pii_for_logging,
            sanitize_user_input,
            validate_user_id,
        )

        try:
            # Validate and sanitize inputs
            user_id = validate_user_id(user_identity.user_id)
            sanitized_text = sanitize_user_input(text)

            # Log safely (with PII redaction)
            logger.info(
                f"Processing request in LangGraph Legion mode for user {user_id[:8]}...: "
                f"{redact_pii_for_logging(sanitized_text, max_len=100)}"
            )

            # Generate AI response using LangGraph with sanitized input
            response_content, metadata = self._generate_ai_response_with_graph(
                sanitized_text, user_id, persona
            )

            # Check if the graph interrupted for human approval
            if response_content == "INTERRUPTED" and metadata.get("interrupted"):
                # Return interrupt information to API caller
                return ProcessRequestResult(
                    message="",  # No message content during interrupt
                    user_id=user_id,
                    response_mode=response_mode,
                    metadata={
                        "legion_mode": True,
                        "langgraph_enabled": True,
                        **metadata,  # Contains interrupt_data and thread_id
                    },
                )

            # Generate TTS if requested
            audio_url = None
            tts_provider = None
            if response_mode == ResponseMode.TTS:
                audio_url, tts_provider = self.generate_tts(response_content)

            # Build result
            result = ProcessRequestResult(
                message=response_content,
                user_id=user_id,
                response_mode=response_mode,
                metadata={
                    "legion_mode": True,
                    "langgraph_enabled": True,
                    "model": "gemini-pro",
                    "prompt_length": len(sanitized_text),
                    "response_length": len(response_content),
                    **metadata,  # Include all metadata from graph execution
                },
                audio_url=audio_url,
                tts_provider=tts_provider,
            )

            logger.info(
                f"LangGraph Legion request processed successfully for user {user_id}"
            )

            return result

        except ValueError as e:
            # Input validation errors
            logger.error(f"Input validation failed: {e}")
            raise InvalidRequestError(f"Invalid input: {e}")
        except InvalidRequestError:
            raise
        except (AIServiceError, TTSServiceError) as e:
            logger.error(f"LangGraph Legion service error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing LangGraph Legion request: {e}")
            raise AIServiceError(f"Failed to process Legion request: {str(e)}")

    def chat(
        self,
        message: str,
        user_identity: UserIdentity,
        include_context: bool = True,
        persona: str = "hermes",
    ) -> GeminiResponse:
        """
        Handle a chat message with LangGraph orchestration.

        Args:
            message: User's chat message
            user_identity: User identity information
            include_context: Whether to include conversation history
            persona: Which AI persona to use ('hermes' or 'prisma')

        Returns:
            GeminiResponse with the AI reply

        Raises:
            InvalidRequestError: If the message is invalid
            AIServiceError: If AI generation fails
        """
        from .utils.input_sanitizer import (
            redact_pii_for_logging,
            sanitize_user_input,
            validate_user_id,
        )

        try:
            # Validate and sanitize inputs
            user_id = validate_user_id(user_identity.user_id)
            sanitized_message = sanitize_user_input(message)

            # Log safely
            logger.info(
                f"Processing chat in LangGraph Legion mode for user {user_id[:8]}...: "
                f"{redact_pii_for_logging(sanitized_message, max_len=100)}"
            )

            # Generate response using LangGraph with sanitized input
            response_content, metadata = self._generate_ai_response_with_graph(
                sanitized_message, user_id, persona
            )

            # Check if the graph interrupted
            if response_content == "INTERRUPTED" and metadata.get("interrupted"):
                # Return interrupt information
                return GeminiResponse(
                    content="",  # Empty content for interrupt
                    user_id=user_id,
                    prompt=sanitized_message,
                    model_used="gemini-pro",
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
                model_used="gemini-pro",
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
            # Input validation errors
            logger.error(f"Input validation failed: {e}")
            raise InvalidRequestError(f"Invalid input: {e}")
        except InvalidRequestError:
            raise
        except AIServiceError as e:
            logger.error(f"LangGraph Legion chat error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing LangGraph Legion chat: {e}")
            raise AIServiceError(f"Failed to process Legion chat: {str(e)}")

    def _generate_ai_response_with_graph(
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
                inputs = resume_value
            else:
                # Create initial state for new conversation
                user_message: Message = {
                    "role": "user",
                    "content": prompt,
                    "timestamp": datetime.utcnow().isoformat(),
                    "metadata": {},
                }

                initial_state: OrchestratorState = {
                    "messages": [user_message],
                    "user_id": user_id,
                    "persona": persona,
                    "task_ledger": {},
                    "agents": {},
                    "tool_allocations": {},
                    "current_agent_id": None,
                    "current_task_id": None,
                    "next_action": "",
                    "required_info": {},
                    "collected_info": {},
                    "pending_questions": [],
                    "decision_rationale": [],
                    "parallel_mode": False,
                    "parallel_tasks": {},
                    "parallel_results": {},
                    "agents_awaiting_info": {},
                    "synthesis_needed": False,
                    "legion_strategy": "council",
                    "legion_results": {},
                    "awaiting_user_response": False,  # NEW: Enable conversation continuation
                    "conversation_complete": False,  # NEW: Track conversation completion
                    "metadata": {},
                }
                inputs = initial_state

            # Get graph and stream execution
            graph = self.get_graph()

            result = None
            for chunk in graph.stream(inputs, config=config, stream_mode="values"):
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
                    if "decision_rationale" in result and result["decision_rationale"]:
                        decision_rationale = result["decision_rationale"]

                        # Build orchestration rationale
                        orchestration_rationale = self._build_orchestration_rationale(
                            decision_rationale,
                            metadata.get("agents_used", []),
                            metadata.get("tools_used", []),
                            result.get("metadata", {}).get(
                                "parallel_execution_metrics"
                            ),
                        )

                        metadata["orchestration_rationale"] = orchestration_rationale

                    # Add performance metrics if available
                    if "parallel_execution_metrics" in result.get("metadata", {}):
                        metadata["parallel_execution_metrics"] = result["metadata"][
                            "parallel_execution_metrics"
                        ]

                    return response_content, metadata

            # Fallback
            return "I apologize, but I couldn't generate a proper response.", {}

        except Exception as e:
            logger.error(f"LangGraph execution failed: {e}")
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
            import uuid

            tts_result = self.tts_service.generate_audio(
                text_input=text,
                upload_to_cloud=True,
                cloud_destination_path=f"legion_tts/{uuid.uuid4().hex}.mp3",
            )

            if not isinstance(tts_result, dict):
                raise TTSServiceError(f"Invalid TTS result type: {type(tts_result)}")

            tts_provider = self.tts_service.tts_provider
            cloud_url = tts_result.get("cloud_url")

            return cloud_url, tts_provider

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error(f"TTS generation failed: {e}")
            raise TTSServiceError(f"Failed to generate audio: {str(e)}")

    def resume(
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
        from .utils.input_sanitizer import validate_user_id

        try:
            user_id = validate_user_id(user_identity.user_id)

            logger.info(f"Resuming conversation for user {user_id}")

            # Resume from checkpoint with user's response
            response_content, metadata = self._generate_ai_response_with_graph(
                prompt="", user_id=user_id, persona=persona, resume_value=resume_value
            )

            # Check if we hit another interrupt
            if response_content == "INTERRUPTED" and metadata.get("interrupted"):
                return GeminiResponse(
                    content="",
                    user_id=user_id,
                    prompt="",
                    model_used="gemini-pro",
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
                prompt="",
                model_used="gemini-pro",
                metadata={
                    "legion_mode": True,
                    "langgraph_enabled": True,
                    "resumed": True,
                    **metadata,
                },
            )

        except Exception as e:
            logger.error(f"Error resuming graph: {e}")
            raise AIServiceError(f"Failed to resume execution: {str(e)}")

    def get_conversation_history(self, user_id: str, limit: int = 50) -> list[Message]:
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

            # Get the latest checkpoint state
            state = self.graph.get_state(config)

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


# Singleton instance
_legion_graph_service: Optional[LegionGraphService] = None


def get_legion_graph_service() -> LegionGraphService:
    """Get or create the LangGraph-based Legion service singleton."""
    global _legion_graph_service
    if _legion_graph_service is None:
        _legion_graph_service = LegionGraphService()
    return _legion_graph_service
