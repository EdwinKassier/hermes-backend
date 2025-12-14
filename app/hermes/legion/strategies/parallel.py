"""Parallel strategy implementation with dependency-aware execution."""

import logging
from typing import Any, Dict, List

from ..agents.task_agent_planner import TaskAgentPlanner
from ..parallel.result_synthesizer import ResultSynthesizer

# ToolAllocator removed - dynamic agents handle their own tools
# PersonaGenerator removed - dynamic agents handle their own personas
from .base import LegionStrategy

logger = logging.getLogger(__name__)


class ParallelStrategy(LegionStrategy):
    """
    Parallel strategy: Decomposes a complex task into subtasks
    and executes them with dependency awareness.

    Key Features:
    - AI-powered task decomposition
    - Dependency detection between subtasks
    - Level-based execution (tasks at same level run in parallel)
    - Sequential execution between levels
    """

    def __init__(self):
        """Initialize parallel strategy."""
        # Dynamic agents handle their own personas - no persona generator needed

    async def generate_workers(
        self, query: str, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Analyze task and create dynamic agents with dependency analysis.

        Uses TaskAgentPlanner to invent completely custom agent types.
        Returns workers with dynamic_agent_config for legion orchestrator.
        """
        try:
            # Use TaskAgentPlanner to analyze task and create dynamic agents
            planner = TaskAgentPlanner()
            analysis = planner.analyze_task_and_plan_agents(
                task_description=query,
                user_context=context.get("user_context"),
                complexity_estimate=context.get("complexity_estimate", "moderate"),
            )

            if not analysis.get("agent_plan"):
                logger.warning("Task analysis failed, falling back to single worker")
                # Create a basic fallback dynamic agent
                fallback_config = {
                    "agent_id": "general_assistant",
                    "agent_type": "general_task_handler",
                    "task_types": ["general", "analysis"],
                    "capabilities": {
                        "primary_focus": "general task assistance and analysis",
                        "tools_needed": ["analysis", "reasoning"],
                        "expertise_level": "intermediate",
                        "specializations": ["problem_solving", "task_analysis"],
                        "knowledge_domains": ["general_knowledge", "task_management"],
                    },
                    "prompts": {
                        "identify_required_info": """Analyze this task to determine what information is needed.

Task: "{task}"
User Message: "{user_message}"

Determine what information is needed to complete this task effectively.

Response format (JSON):
{{
  "needs_info": true|false,
  "inferred_values": {{}},
  "required_fields": [],
  "reasoning": "why you need this information"
}}""",
                        "execute_task": """Complete this task using your capabilities.

Task: {task}
{judge_feedback}

Your capabilities: {capabilities}
Available tools: {tool_context}

Provide a comprehensive solution to the task.""",
                    },
                    "persona": "helpful_assistant",
                    "task_portion": query,
                    "dependencies": [],
                }

                fallback_workers = [
                    {
                        "worker_id": "parallel_fallback",
                        "role": "general_assistant",
                        "task_description": query,
                        "tools": [],
                        "execution_level": 0,
                        "dependencies": [],
                        "dynamic_agent_config": fallback_config,
                    }
                ]

                return fallback_workers

            # Convert analysis to worker plan with dynamic agents
            worker_plan = planner.create_worker_plan_from_analysis(analysis, query)

            logger.info(
                f"Created {len(worker_plan)} dynamic agents for parallel execution"
            )
            return worker_plan

        except Exception as e:
            logger.error(f"Dynamic agent planning failed: {e}, using fallback")
            # Fallback to simple single worker
            fallback_config = {
                "agent_id": "emergency_fallback",
                "agent_type": "emergency_task_handler",
                "task_types": ["general"],
                "capabilities": {
                    "primary_focus": "handling tasks when planning fails",
                    "tools_needed": ["basic_reasoning"],
                    "expertise_level": "basic",
                    "specializations": ["fallback_handling"],
                    "knowledge_domains": ["general_assistance"],
                },
                "prompts": {
                    "identify_required_info": """Task: "{task}"

Response: {{"needs_info": false, "inferred_values": {{}}, "required_fields": [], "reasoning": "fallback mode"}}""",
                    "execute_task": """Task: {task}

I apologize, but I encountered an issue with the planning system. However, I can still help you with this task using my general capabilities.

Please provide more details about what you'd like me to help you with.""",
                },
                "persona": "helpful_assistant",
                "task_portion": query,
                "dependencies": [],
            }

            return [
                {
                    "worker_id": "emergency_fallback",
                    "role": "emergency_fallback",
                    "task_description": query,
                    "tools": [],
                    "execution_level": 0,
                    "dependencies": [],
                    "dynamic_agent_config": fallback_config,
                }
            ]

    async def synthesize_results(
        self, original_query: str, results: Dict[str, Any], persona: str
    ) -> str:
        """
        Synthesize subtask results with level-aware ordering.

        Results from earlier execution levels are presented first to
        maintain logical flow in the synthesized response.
        """
        # Format results for synthesizer, preserving level information
        formatted_results = {}
        for worker_id, data in results.items():
            formatted_results[worker_id] = {
                "agent_id": worker_id,
                "result": data["result"],
                "status": data["status"],
                "agent_type": data["role"],
                # Include metadata for synthesis
                "execution_level": data.get("execution_level", 0),
            }

        # Sort by execution level for ordered synthesis
        sorted_results = dict(
            sorted(
                formatted_results.items(),
                key=lambda x: x[1].get("execution_level", 0),
            )
        )

        synthesizer = ResultSynthesizer()
        return synthesizer.synthesize_results(
            original_query=original_query,
            agent_results=sorted_results,
            persona=persona,
        )
