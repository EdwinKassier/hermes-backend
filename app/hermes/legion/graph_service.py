import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from app.hermes.exceptions import AIServiceError
from app.hermes.models import GeminiResponse, ProcessRequestResult, UserIdentity
from app.shared.config.langfuse_config import langfuse_config
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
            "agents": self._get_agents_explanation(agents_used, result),  # noqa: F821
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
            return f"Multi-agent parallel execution: Orchestrator agent decomposed task into {subtask_count} independent subtasks for concurrent processing with specialized sub-agents"

        if decisions.get("agent_needed"):
            agent_type = decisions.get("agent_type", "specialist")
            task_type = analysis.get("identified_task_type", agent_type)
            return f"Single-agent execution: Orchestrator agent routed to specialized {agent_type} sub-agent for {task_type.title()} task"

        return "Direct response: Orchestrator agent determined simple question answerable from knowledge base and is providing the answer directly without specialized sub-agents"

    def _get_agents_explanation(self, agents_used: list, result: dict = None) -> list:
        """Explain which agents were used and why, using rich metadata from dynamic agents."""
        if not agents_used:
            return []

        agent_explanations = []

        # Try to get worker metadata from result if available
        worker_metadata = {}
        if result and "task_ledger" in result:
            for entry in result["task_ledger"]:
                if "worker_id" in entry and "metadata" in entry:
                    worker_metadata[entry["worker_id"]] = entry["metadata"]

        for agent_id in agents_used:
            # For dynamic agents, extract rich information from metadata
            metadata = worker_metadata.get(agent_id, {})

            if metadata and "agent_type" in metadata:
                # Rich metadata available from dynamic agent
                agent_type = metadata.get("agent_type", "dynamic_agent")
                capabilities = metadata.get("capabilities", {})
                persona = metadata.get("persona", "specialist")
                task_types = metadata.get("task_types", [])
                specialization = metadata.get("specialization", "")

                explanation = {
                    "id": agent_id,
                    "type": agent_type,
                    "role": self._get_dynamic_agent_role(
                        agent_type, capabilities, task_types
                    ),
                    "persona": persona,
                    "capabilities": (
                        list(capabilities.keys())[:3] if capabilities else []
                    ),  # Top 3 capabilities
                    "specialization": specialization,
                    "task_types": task_types,
                }
            else:
                # Fallback for agents without rich metadata
                agent_type = agent_id.split("_")[0] if "_" in agent_id else agent_id
                explanation = {
                    "id": agent_id,
                    "type": agent_type,
                    "role": self._get_agent_role(agent_type),
                }

            agent_explanations.append(explanation)

        return agent_explanations

    def _get_dynamic_agent_role(
        self, agent_type: str, capabilities: dict, task_types: list
    ) -> str:
        """Generate detailed role description for dynamic agents."""
        try:
            # Build comprehensive role description
            role_parts = []

            # Add primary focus from capabilities
            if "primary_focus" in capabilities:
                role_parts.append(capabilities["primary_focus"])
            else:
                role_parts.append(f"{agent_type.replace('_', ' ').title()} specialist")

            # Add expertise level
            expertise = capabilities.get("expertise_level", "specialized")
            role_parts.append(f"with {expertise} expertise")

            # Add key task types
            if task_types:
                primary_tasks = task_types[:2]  # Show top 2 task types
                task_desc = ", ".join(primary_tasks)
                role_parts.append(f"handling {task_desc} tasks")

            # Add specialization if available
            specializations = capabilities.get("specializations", [])
            if specializations:
                spec_desc = ", ".join(specializations[:2])
                role_parts.append(f"specializing in {spec_desc}")

            return " ".join(role_parts)

        except Exception:
            # Fallback to simple description
            return f"{agent_type.replace('_', ' ').title()} operations"

    def _get_agent_role(self, agent_type: str) -> str:
        """Get human-readable role description for agent type."""
        # Dynamic agent system uses custom types - provide generic descriptions
        legacy_roles = {
            "orchestrator": "Request routing, decision-making, and direct response generation",
        }
        return legacy_roles.get(
            agent_type, f"{agent_type.replace('_', ' ').title()} operations"
        )

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

    def _extract_judge_metadata(self, task_ledger: dict) -> dict:
        """Extract judge evaluation metadata from task ledger.

        Args:
            task_ledger: Dictionary of task_id -> TaskInfo

        Returns:
            Dictionary with judge evaluation summary
        """
        if not task_ledger:
            return {}

        judge_summary = {
            "enabled": False,
            "evaluations_performed": 0,
            "tasks_evaluated": 0,
            "retries_triggered": 0,
            "average_score": 0.0,
            "tasks": [],
        }

        total_score = 0.0
        score_count = 0

        for task_id, task_info in task_ledger.items():
            # Check if this task has judgment history
            judgment_history = getattr(task_info, "judgment_history", [])

            if judgment_history:
                judge_summary["enabled"] = True
                judge_summary["tasks_evaluated"] += 1
                judge_summary["evaluations_performed"] += len(judgment_history)

                # Count retries (evaluations - 1, since first eval is not a retry)
                if len(judgment_history) > 1:
                    judge_summary["retries_triggered"] += len(judgment_history) - 1

                # Calculate scores
                for judgment in judgment_history:
                    score = judgment.get("score", 0.0)
                    total_score += score
                    score_count += 1

                # Add task-level summary
                final_judgment = judgment_history[-1]
                task_summary = {
                    "task_id": task_id,
                    "evaluations": len(judgment_history),
                    "final_score": final_judgment.get("score", 0.0),
                    "final_status": (
                        "accepted"
                        if final_judgment.get("is_valid")
                        else "rejected_max_retries"
                    ),
                    "retry_count": getattr(task_info, "retry_count", 0),
                    "criteria": getattr(task_info, "judge_criteria", None),
                }
                judge_summary["tasks"].append(task_summary)

        # Calculate average score
        if score_count > 0:
            judge_summary["average_score"] = round(total_score / score_count, 2)

        # Return empty dict if judge was not used
        if not judge_summary["enabled"]:
            return {}

        return judge_summary

    async def _build_structured_metadata(
        self,
        result: dict,
        start_time: float,
        duration_ms: float,
        original_query: Optional[str] = None,
    ) -> dict:
        """Build structured metadata in hierarchical format for UI rendering.

        Uses AI-powered dynamic generation for context-aware descriptions
        while preserving the existing metadata structure for frontend compatibility.

        Args:
            result: Final graph state
            start_time: Orchestration start time
            duration_ms: Total duration in milliseconds

        Returns:
            Structured metadata dictionary
        """
        from datetime import datetime

        state_metadata = result.get("metadata", {})
        execution_path = result.get("execution_path", [])
        level_results = result.get("level_results", {})
        legion_strategy = result.get("legion_strategy")
        decision_rationale = result.get("decision_rationale", [])

        # Extract actual agents and tools used from execution results
        agents_used = []
        tools_used = []

        if level_results:
            for level_data in level_results.values():
                level_workers = level_data.get("workers", {})
                for worker_id, worker_data in level_workers.items():
                    # Add worker ID to agents_used
                    if worker_id not in agents_used:
                        agents_used.append(worker_id)

                    # Extract tools used by this worker (if available in metadata)
                    worker_tools = worker_data.get("tools_used", [])
                    for tool in worker_tools:
                        if tool not in tools_used:
                            tools_used.append(tool)

        # Fallback to state metadata if no execution results found
        if not agents_used:
            agents_used = state_metadata.get("agents_used", [])
        if not tools_used:
            tools_used = state_metadata.get("tools_used", [])

        # Determine execution mode
        if legion_strategy:
            mode = "parallel"
        elif agents_used:
            mode = "single-agent"
        else:
            mode = "direct"

        # Build status block
        status = {
            "code": "completed",
            "duration_ms": duration_ms,
            "timestamp": datetime.now().isoformat(),
            "interrupted": False,
            "timed_out": False,
        }

        # Build execution block
        execution = {
            "mode": mode,
            "trace": execution_path,
        }

        # Add strategy for parallel execution
        if legion_strategy:
            execution["strategy"] = legion_strategy

        # Add levels for parallel execution
        if level_results:
            levels = []
            for level_id in sorted(level_results.keys()):
                level_data = level_results[level_id]
                level_workers = level_data.get("workers", {})

                workers = []
                for worker_id, worker_data in level_workers.items():
                    # Create human-readable agent name
                    role = worker_data.get("role", "unknown")
                    if worker_id.startswith("task_") and role == "research":
                        agent_name = "Research Assistant"
                    elif worker_id.startswith("dynamic_worker_"):
                        agent_name = role  # Dynamic workers already have good names
                    else:
                        agent_name = f"{role.replace('_', ' ').title()} Agent"

                    workers.append(
                        {
                            "id": worker_id,
                            "role": agent_name,  # Use human-readable name
                            "status": worker_data.get("status", "unknown"),
                            "result_summary": (
                                worker_data.get("result", "")[:200] + "..."
                                if len(worker_data.get("result", "")) > 200
                                else worker_data.get("result", "")
                            ),
                            "tools_used": worker_data.get(
                                "tools_used", []
                            ),  # Extract from worker metadata
                            "duration_ms": int(
                                worker_data.get("duration_seconds", 0) * 1000
                            ),
                        }
                    )

                levels.append(
                    {
                        "level_id": level_id,
                        "status": (
                            "completed"
                            if level_data.get("failed_count", 0) == 0
                            else "partial"
                        ),
                        "workers": workers,
                    }
                )

            execution["levels"] = levels

            # Calculate parallel execution metrics
            total_duration = sum(
                worker.get("duration_seconds", 0)
                for level in level_results.values()
                for worker in level.get("workers", {}).values()
            )
            wall_time = duration_ms / 1000.0
            total_workers = sum(
                len(level.get("workers", {})) for level in level_results.values()
            )

            if wall_time > 0 and total_workers > 0:
                efficiency = min(1.0, total_duration / (wall_time * total_workers))
                execution["metrics"] = {
                    "total_duration_seconds": total_duration,
                    "wall_time_seconds": wall_time,
                    "parallel_efficiency": round(efficiency, 2),
                    "worker_count": total_workers,
                }

        # Use provided original query, or extract from messages as fallback
        if original_query is None:
            messages = result.get("messages", [])
            original_query = ""
            for msg in messages:
                if msg.get("role") == "user":
                    original_query = msg.get("content", "")
                    break

        # Build rationale block with dynamic summary
        rationale = {}
        if decision_rationale:
            latest_decision = decision_rationale[-1] if decision_rationale else {}
            analysis = latest_decision.get("analysis", {})
            decisions = latest_decision.get("decisions", {})

            # Generate detailed orchestration summary with dynamic agent information
            rationale["summary"] = await self._generate_dynamic_orchestration_summary(
                analysis=analysis,
                decisions=decisions,
                original_query=original_query,
                agents_used=agents_used,
                level_results=level_results,
                execution_mode=mode,
            )
            rationale["decisions"] = decision_rationale

        # Extract judge information from task ledger
        task_ledger = result.get("task_ledger", {})
        judge_info = self._extract_judge_metadata(task_ledger)

        # Build usage block with detailed toolset explanation
        toolset_explanation = await self._generate_dynamic_toolset_explanation(
            tools_used=tools_used,
            agents_used=agents_used,
            original_query=original_query,
        )

        usage = {
            "agents": agents_used,
            "tools": tools_used,
            "toolset_rationale": toolset_explanation,
        }

        # Add judge info if present
        if judge_info:
            usage["quality_assurance"] = judge_info

        return {
            "status": status,
            "execution": execution,
            "rationale": rationale,
            "usage": usage,
        }

    async def _generate_dynamic_orchestration_summary(
        self,
        analysis: Dict[str, Any],
        decisions: Dict[str, Any],
        original_query: str,
        agents_used: List[str],
        level_results: Dict[str, Any],
        execution_mode: str,
    ) -> str:
        """
        Generate detailed orchestration summary for dynamic agents.

        Provides comprehensive information about the execution including:
        - Number and types of dynamic agents used
        - Execution strategy and mode
        - Task breakdown and agent specialization
        - Performance and coordination details
        """
        try:
            # Build agent breakdown with actual roles from execution
            agent_breakdown = []
            agent_roles = set()

            # Extract roles from level_results
            if level_results:
                for level_data in level_results.values():
                    level_workers = level_data.get("workers", {})
                    for worker_id, worker_data in level_workers.items():
                        role = worker_data.get("role", "unknown")
                        agent_roles.add(role)
                        agent_breakdown.append(f"{worker_id} ({role})")

            # Fallback to agent IDs if no level results
            if not agent_breakdown:
                for agent_id in agents_used:
                    agent_type = agent_id.split("_")[0] if "_" in agent_id else agent_id
                    agent_role = self._get_agent_role(agent_type)
                    agent_breakdown.append(f"{agent_id} ({agent_role})")
                    agent_roles.add(agent_role)

            agents_summary = (
                ", ".join(agent_breakdown) if agent_breakdown else "dynamic agents"
            )

            # Generate comprehensive summary
            worker_count = len(agent_breakdown) if agent_breakdown else len(agents_used)
            summary_parts = [
                f"Executed {worker_count} specialized dynamic agents in {execution_mode} orchestration mode.",
                f"Agent composition: {agents_summary}",
                f"Task: {original_query[:150]}{'...' if len(original_query) > 150 else ''}",
            ]

            # Add execution details if available
            if analysis:
                complexity = analysis.get("complexity_estimate", "unknown")
                strategy = decisions.get("strategy", execution_mode)
                summary_parts.append(
                    f"Complexity assessment: {complexity} | Strategy: {strategy}"
                )

            # Add coordination information
            if len(agents_used) > 1:
                summary_parts.append(
                    f"Multi-agent coordination achieved through dynamic task decomposition "
                    f"and parallel execution with specialized agent roles."
                )
            else:
                summary_parts.append(
                    f"Single-agent execution optimized for task requirements "
                    f"through dynamic capability matching."
                )

            return " ".join(summary_parts)

        except Exception as e:
            # Fallback summary if generation fails
            return (
                f"Executed {len(agents_used)} dynamic agents in {execution_mode} mode "
                f"to process: {original_query[:100]}{'...' if len(original_query) > 100 else ''}"
            )

    async def _generate_dynamic_toolset_explanation(
        self,
        tools_used: List[str],
        agents_used: List[str],
        original_query: str,
    ) -> str:
        """
        Generate detailed toolset explanation for dynamic agents.

        Explains tool allocation, agent specialization, and execution efficiency.
        """
        try:
            # Group tools by agent type for dynamic agents
            tool_agent_mapping = {}

            for agent_id in agents_used:
                agent_type = agent_id.split("_")[0] if "_" in agent_id else agent_id
                if agent_type not in tool_agent_mapping:
                    tool_agent_mapping[agent_type] = []
                # For dynamic agents, we distribute tools across agent types
                tools_per_agent = len(tools_used) // len(agents_used) or 1
                start_idx = agents_used.index(agent_id) * tools_per_agent
                end_idx = start_idx + tools_per_agent
                agent_tools = tools_used[start_idx:end_idx] if tools_used else []
                tool_agent_mapping[agent_type].extend(agent_tools)

            # Calculate actual worker count from agents_used or use provided count
            worker_count = len(agents_used) if agents_used else 0

            # Build comprehensive explanation
            explanation_parts = [
                f"Deployed {len(tools_used)} specialized tools across {worker_count} dynamic agents.",
            ]

            # Add agent-tool mapping details
            if tool_agent_mapping:
                mapping_details = []
                for agent_type, agent_tools in tool_agent_mapping.items():
                    if agent_tools:
                        agent_role = self._get_agent_role(agent_type)
                        tool_list = ", ".join(agent_tools[:3])  # Limit to first 3 tools
                        if len(agent_tools) > 3:
                            tool_list += f" +{len(agent_tools) - 3} more"
                        mapping_details.append(f"{agent_role}: {tool_list}")

                if mapping_details:
                    explanation_parts.append(
                        f"Tool allocation: {'; '.join(mapping_details)}"
                    )

            # Add efficiency and specialization notes
            if len(tools_used) > 0:
                efficiency_note = (
                    f"Tool utilization optimized for task requirements with "
                    f"{len(set(tools_used))} unique tool types deployed."
                )
                explanation_parts.append(efficiency_note)

            if len(agents_used) > 1:
                specialization_note = (
                    f"Multi-agent specialization enabled complementary tool usage "
                    f"across {len(agents_used)} distinct execution contexts."
                )
                explanation_parts.append(specialization_note)

            return " ".join(explanation_parts)

        except Exception as e:
            # Fallback explanation if generation fails
            return (
                f"Utilized {len(tools_used)} tools across {len(agents_used)} dynamic agents "
                f"to efficiently complete the task requirements."
            )

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
        trace_id: Optional[str] = None,
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
            trace_id: Optional trace ID for distributed tracing

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

        # Create explicit Langfuse trace for the entire graph execution
        langfuse_trace = None
        if langfuse_config.is_enabled:
            try:
                from langfuse import get_client

                client = get_client()
                if client:
                    langfuse_trace = client.trace(
                        name="legion_graph_execution",
                        user_id=user_id,
                        metadata={
                            "persona": persona,
                            "prompt": prompt[:200],  # Truncate for metadata
                            "langgraph_enabled": True,
                        },
                    )
                    if trace_id:
                        langfuse_trace.id = trace_id
                    logger.info("Created Langfuse trace for graph execution")
                else:
                    logger.warning("Langfuse client is None - cannot create trace")
            except ImportError:
                logger.warning(
                    "Langfuse not available for trace creation (ImportError)"
                )
            except Exception as e:
                logger.warning("Failed to create Langfuse trace: %s", e, exc_info=True)
        else:
            logger.warning(
                "Langfuse is not enabled - skipping trace creation. "
                "Check LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables."
            )

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
            # Build config with Langfuse callbacks and metadata
            config = self._build_langgraph_config(
                user_id=user_id, persona=persona, trace_id=trace_id
            )

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
                        # Execution path tracking
                        "execution_path": [],
                    }

                # Create graph with this checkpointer
                graph = get_orchestration_graph(checkpointer=checkpointer)

                async def execute_graph_with_tracking():
                    """Inner function to track execution for timeout handling."""
                    nonlocal partial_result, workers_completed, total_workers

                    try:
                        # Use ainvoke instead of astream to avoid async generator issues
                        # For initial Legion requests (not resume), we don't need streaming
                        result = await graph.ainvoke(inputs, config=config)

                        # Store the final result for potential timeout recovery
                        partial_result = result

                        # Track worker progress from final result
                        if "legion_results" in result:
                            workers_completed = len(result["legion_results"])
                        if "metadata" in result:
                            plans = result["metadata"].get("legion_worker_plans", [])
                            total_workers = len(plans)

                        logger.info(
                            "Graph execution completed successfully. Result keys: %s",
                            list(result.keys()) if result else "None",
                        )

                        return result, None  # result, no interrupt

                    except Exception as e:
                        # Log the execution error
                        logger.error("Graph execution failed: %s", e)
                        raise
                    except asyncio.CancelledError:
                        # Task was cancelled (e.g., by timeout) - ensure generator is closed
                        # The finally block above will handle cleanup
                        raise
                    except Exception as e:
                        # Log the exception before re-raising
                        logger.debug("Exception during graph execution: %s", e)
                        raise

                # Execute with global timeout
                try:
                    execution_result = await asyncio.wait_for(
                        execute_graph_with_tracking(),
                        timeout=timeout,
                    )
                except asyncio.CancelledError:
                    # Task was cancelled (likely due to timeout)
                    # The generator cleanup should have happened in the finally block
                    logger.warning("Graph execution was cancelled")
                    raise
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

                    # Flush Langfuse events on timeout
                    try:
                        langfuse_config.flush()
                    except Exception as flush_error:
                        logger.warning(
                            "Failed to flush Langfuse events: %s", flush_error
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

                        # Create structured metadata with dynamic generation
                        duration_ms = (time.time() - start_time) * 1000
                        try:
                            metadata = await self._build_structured_metadata(
                                result=result,
                                start_time=start_time,
                                duration_ms=duration_ms,
                                original_query=prompt,
                            )
                        except Exception as e:
                            logger.error(
                                f"Failed to build structured metadata: {e}",
                                exc_info=True,
                            )
                            metadata = {
                                "execution": {
                                    "mode": "parallel",
                                    "strategy": "parallel",
                                },
                                "rationale": {
                                    "summary": f"Error generating metadata: {str(e)}"
                                },
                                "usage": {"agents": [], "tools": []},
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

                        # Complete and flush Langfuse trace
                        if langfuse_trace:
                            try:
                                langfuse_trace.update(
                                    output=response_content[:500],  # Truncate output
                                    metadata={"status": "completed", **metadata},
                                )
                            except Exception as trace_error:
                                logger.warning(
                                    "Failed to update Langfuse trace: %s", trace_error
                                )

                        # Flush Langfuse events to ensure they're sent
                        try:
                            langfuse_config.flush()
                            logger.debug(
                                "Langfuse events flushed after successful completion"
                            )
                        except Exception as flush_error:
                            logger.warning(
                                "Failed to flush Langfuse events: %s", flush_error
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
                # Flush Langfuse events even on fallback
                try:
                    langfuse_config.flush()
                except Exception as flush_error:
                    logger.warning("Failed to flush Langfuse events: %s", flush_error)
                return "I apologize, but I couldn't generate a proper response.", {}

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            obs.log_orchestration_complete(
                user_id=user_id,
                duration_ms=duration_ms,
                worker_count=0,
                success=False,
            )
            # Complete Langfuse trace with error
            if langfuse_trace:
                try:
                    langfuse_trace.update(
                        output=None,
                        level="ERROR",
                        status_message=str(e)[:500],
                        metadata={"status": "error", "error_type": type(e).__name__},
                    )
                except Exception as trace_error:
                    logger.warning(
                        "Failed to update Langfuse trace with error: %s", trace_error
                    )

            # Flush Langfuse events even on error
            try:
                langfuse_config.flush()
                logger.debug("Langfuse events flushed after error")
            except Exception as flush_error:
                logger.warning("Failed to flush Langfuse events: %s", flush_error)
            logger.error("Error executing orchestration graph: %s", e)
            raise AIServiceError(f"Graph execution failed: {str(e)}") from e

    def _build_langgraph_config(
        self,
        user_id: str,
        persona: str = "hermes",
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build LangGraph config with Langfuse callbacks and metadata.

        Args:
            user_id: User identifier
            persona: AI persona name
            trace_id: Optional trace ID for distributed tracing

        Returns:
            Config dictionary with callbacks and metadata for LangGraph execution
        """
        # Setup Langfuse callback handler for observability
        langfuse_handler = langfuse_config.get_callback_handler()
        callbacks = [langfuse_handler] if langfuse_handler else []

        # Log Langfuse status for debugging (use info level so it's visible)
        if langfuse_handler:
            logger.info("Langfuse callback handler created successfully")
        else:
            # Check if Langfuse is available
            try:
                from app.shared.config.langfuse_config import (
                    LANGFUSE_AVAILABLE as LF_AVAILABLE,
                )

                available_str = str(LF_AVAILABLE)
            except ImportError:
                available_str = "unknown"
            logger.warning(
                "Langfuse callback handler not available (enabled=%s, available=%s)",
                langfuse_config.is_enabled,
                available_str,
            )

        # Build metadata for Langfuse trace attributes (v3 pattern)
        metadata = {
            "langfuse_user_id": user_id,
            "persona": persona,
            "langgraph_enabled": True,
        }
        if trace_id:
            metadata["langfuse_trace_id"] = trace_id

        config = {
            "configurable": {"thread_id": user_id},
            "callbacks": callbacks,
            "metadata": metadata,
        }

        logger.debug(
            "LangGraph config built: callbacks=%d, metadata=%s",
            len(callbacks),
            list(metadata.keys()),
        )

        return config

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
            # Build config with Langfuse callbacks and metadata
            config = self._build_langgraph_config(user_id=user_id)

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
