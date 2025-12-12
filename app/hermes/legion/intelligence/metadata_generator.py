"""
AI-powered metadata generation for Legion orchestration.

Generates dynamic, context-aware descriptions for metadata fields
while preserving the existing metadata structure for frontend compatibility.
"""

import logging
from typing import Any, Dict, List, Optional

from app.shared.utils.service_loader import get_async_llm_service

logger = logging.getLogger(__name__)


class MetadataGenerator:
    """
    AI-powered metadata generation for Legion orchestration.

    Generates dynamic, context-aware descriptions for metadata fields
    while preserving the existing metadata structure for frontend compatibility.
    """

    def __init__(self) -> None:
        """Initialize the metadata generator with async LLM service."""
        self.llm_service = get_async_llm_service()

    async def generate_orchestration_summary(
        self,
        analysis: Dict[str, Any],
        decisions: Dict[str, Any],
        original_query: str,
        agents_used: List[str],
        execution_mode: str,
    ) -> str:
        """
        Generate a dynamic, context-aware orchestration structure explanation.

        Args:
            analysis: Analysis data from orchestration decisions
            decisions: Decision data including agent selection
            original_query: The user's original query
            agents_used: List of agent IDs that were used
            execution_mode: The execution mode (parallel, single-agent, direct)

        Returns:
            Dynamic orchestration summary string
        """
        try:
            # Build context for LLM
            multi_agent = analysis.get("multi_agent_task_detected", False)
            subtask_count = analysis.get(
                "subtasks_count", len(analysis.get("parallel_tasks", []))
            )
            agent_type = decisions.get("agent_type", "specialist")
            task_type = analysis.get("identified_task_type", agent_type)

            prompt = f"""Generate a concise, insightful 1-2 sentence explanation of how this AI orchestration handled the user's request.

USER REQUEST: "{original_query[:200]}"

ORCHESTRATION DETAILS:
- Execution mode: {execution_mode}
- Multi-agent task: {multi_agent}
- Number of subtasks: {subtask_count}
- Agents deployed: {', '.join(agents_used) if agents_used else 'none (direct response)'}
- Primary task type: {task_type}

GUIDELINES:
- Be specific to THIS request, not generic
- Explain WHY this approach was chosen for this particular query
- Use active, engaging language
- Keep under 50 words
- Do not start with "I" or use first person

Return ONLY the explanation text, no preamble."""

            response = await self.llm_service.generate_async(prompt, persona="hermes")
            return response.strip()

        except Exception as e:
            logger.warning(f"Dynamic orchestration summary generation failed: {e}")
            # Fallback to static text
            return self._get_static_orchestration_summary(
                analysis, decisions, execution_mode
            )

    def _get_static_orchestration_summary(
        self,
        analysis: Dict[str, Any],
        decisions: Dict[str, Any],
        execution_mode: str,
    ) -> str:
        """Fallback static orchestration summary."""
        if analysis.get("multi_agent_task_detected"):
            subtask_count = analysis.get(
                "subtasks_count", len(analysis.get("parallel_tasks", []))
            )
            return (
                f"Multi-agent parallel execution: Orchestrator agent decomposed "
                f"task into {subtask_count} independent subtasks for concurrent "
                "processing with specialized sub-agents"
            )

        if decisions.get("agent_needed"):
            agent_type = decisions.get("agent_type", "specialist")
            task_type = analysis.get("identified_task_type", agent_type)
            return (
                f"Single-agent execution: Orchestrator agent routed to specialized "
                f"{agent_type} sub-agent for {task_type.title()} task"
            )

        return (
            "Direct response: Orchestrator agent determined simple question "
            "answerable from knowledge base and is providing the answer directly "
            "without specialized sub-agents"
        )

    async def generate_agent_role_description(
        self,
        agent_type: str,
        agent_id: str,
        task_description: Optional[str] = None,
    ) -> str:
        """
        Generate a dynamic agent role description based on the actual task.

        Args:
            agent_type: Type of agent (research, code, analysis, etc.)
            agent_id: The agent's ID
            task_description: Optional specific task assigned to this agent

        Returns:
            Dynamic role description string
        """
        try:
            context = (
                f" working on: {task_description[:100]}" if task_description else ""
            )

            prompt = f"""Generate a brief, dynamic role description for this AI agent.

AGENT TYPE: {agent_type}
AGENT ID: {agent_id}{context}

Create a 1 sentence description that captures:
- What this agent specializes in
- How it contributes to the overall task

Keep it under 20 words. Be specific and avoid generic descriptions.
Return ONLY the description text."""

            response = await self.llm_service.generate_async(prompt, persona="hermes")
            return response.strip()

        except Exception as e:
            logger.warning(f"Dynamic agent role description failed: {e}")
            # Fallback to static roles
            return self._get_static_agent_role(agent_type)

    def _get_static_agent_role(self, agent_type: str) -> str:
        """Fallback static agent role description."""
        roles = {
            "orchestrator": "Request routing, decision-making, and direct response generation",
            "research": "Information gathering and research",
            "code": "Code generation and programming",
            "analysis": "Data analysis and evaluation",
            "data": "Data processing and transformation",
        }
        return roles.get(agent_type, f"{agent_type.title()} operations")

    async def generate_toolset_explanation(
        self,
        tools_used: List[str],
        agents_used: List[str],
        original_query: str,
    ) -> Dict[str, Any]:
        """
        Generate a dynamic toolset allocation explanation.

        Args:
            tools_used: List of tool names that were used
            agents_used: List of agent IDs that were used
            original_query: The user's original query

        Returns:
            Dictionary with toolset explanation (matching existing structure)
        """
        if not tools_used and not agents_used:
            return {}

        try:
            tools_list = ", ".join(tools_used) if tools_used else "no external tools"
            agent_count = len(agents_used)

            prompt = f"""Generate a brief, insightful explanation of tool allocation for this AI task.

USER REQUEST: "{original_query[:150]}"
TOOLS ALLOCATED: {tools_list}
NUMBER OF AGENTS: {agent_count}

Create a 1-2 sentence explanation that captures:
- Why these specific tools were chosen
- How they help accomplish the user's goal

Keep under 30 words. Be specific to this request.
Return ONLY the explanation text."""

            response = await self.llm_service.generate_async(prompt, persona="hermes")
            allocation_reason = response.strip()

        except Exception as e:
            logger.warning(f"Dynamic toolset explanation failed: {e}")
            allocation_reason = (
                "Tools allocated based on task requirements and agent capabilities"
            )

        # Build response matching existing structure
        toolset_info: Dict[str, Any] = {
            "available_tools": tools_used if tools_used else [],
            "allocation_reason": allocation_reason,
        }

        if len(agents_used) > 1:
            toolset_info["distribution"] = (
                "Each agent receives task-specific subset of tools"
            )

        return toolset_info

    async def generate_agents_explanation(
        self,
        agents_used: List[str],
        level_results: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate dynamic explanations for each agent used.

        Args:
            agents_used: List of agent IDs
            level_results: Optional level results containing worker details

        Returns:
            List of agent explanation dictionaries
        """
        if not agents_used:
            return []

        agent_explanations = []

        for agent_id in agents_used:
            # Extract agent type from ID (e.g., "research_1" -> "research")
            agent_type = agent_id.split("_")[0]

            # Try to get task description from level results
            task_description = None
            if level_results:
                for level_data in level_results.values():
                    workers = level_data.get("workers", {})
                    if agent_id in workers:
                        worker_data = workers[agent_id]
                        task_description = worker_data.get("task_description")
                        break

            # Generate dynamic role
            role = await self.generate_agent_role_description(
                agent_type=agent_type,
                agent_id=agent_id,
                task_description=task_description,
            )

            explanation = {
                "id": agent_id,
                "type": agent_type,
                "role": role,
            }
            agent_explanations.append(explanation)

        return agent_explanations
