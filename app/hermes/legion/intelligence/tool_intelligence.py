"""
Tool Intelligence Service.

Intelligently recommends tools for agent tasks based on requirements.
An abstraction over the ToolAllocator for AI-powered allocation.
"""

import logging
from typing import Any, Dict, List

from app.shared.utils.service_loader import get_async_llm_service

from ..utils.llm_utils import extract_json_from_llm_response

logger = logging.getLogger(__name__)


class ToolIntelligence:
    """Intelligently recommends tools for tasks."""

    def __init__(self):
        self.llm_service = get_async_llm_service()

    async def recommend_tools(
        self, role: str, task: str, available_tools: List[Any]
    ) -> List[str]:
        """
        Recommend the best tools for a specific worker role and task.
        Returns a list of tool names.
        """
        try:
            # Format available tools for prompt
            tools_desc = "\n".join(
                [f"- {t.name}: {t.description}" for t in available_tools]
            )

            prompt = f"""
            Select the most appropriate tools for an AI agent.

            Agent Role: {role}
            Task: "{task}"

            Available Tools:
            {tools_desc}

            Select ONLY tools that are strictly necessary for this task.
            If no tools are needed, return an empty list.

            Return ONLY valid JSON:
            {{
                "selected_tools": ["tool_name_1", "tool_name_2"]
            }}
            """

            response = await self.llm_service.generate_async(prompt, persona="hermes")
            data = extract_json_from_llm_response(response)

            selected = data.get("selected_tools", [])

            # Validate selection against available tools
            valid_names = {t.name for t in available_tools}
            validated = [t for t in selected if t in valid_names]

            return validated

        except Exception as e:
            logger.error(f"Error recommending tools: {e}")
            return []

    async def rank_tools_by_relevance(
        self, task: str, available_tools: List[Any]
    ) -> Dict[str, float]:
        """
        Rank tools by their relevance to the task (0.0 to 1.0).
        """
        try:
            tools_desc = "\n".join(
                [f"- {t.name}: {t.description}" for t in available_tools]
            )

            prompt = f"""
            Rate the relevance of each tool for the following task (0.0 to 1.0).

            Task: "{task}"

            Tools:
            {tools_desc}

            Return ONLY valid JSON:
            {{
                "tool_relevance": {{
                    "tool_name_1": 0.9,
                    "tool_name_2": 0.1
                }}
            }}
            """

            response = await self.llm_service.generate_async(prompt, persona="hermes")
            data = extract_json_from_llm_response(response)

            return data.get("tool_relevance", {})

        except Exception as e:
            logger.error(f"Error ranking tools: {e}")
            return {}
