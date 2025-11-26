"""Analysis agent for data analysis and evaluation tasks."""

import logging
from typing import Dict, List

from app.shared.utils.service_loader import get_gemini_service

from ..models import RequiredInfoField, SubAgentState
from .base import BaseSubAgent

logger = logging.getLogger(__name__)


class AnalysisAgent(BaseSubAgent):
    """Agent for data analysis and evaluation tasks."""

    @property
    def agent_id(self) -> str:
        """Unique identifier for analysis agent."""
        return "analysis_agent"

    @property
    def task_types(self) -> List[str]:
        """Task types this agent can handle."""
        return ["analysis", "evaluation", "assessment", "comparison"]

    def identify_required_info(
        self, task: str, user_message: str
    ) -> Dict[str, RequiredInfoField]:
        """
        Identify required information for analysis tasks.

        Args:
            task: Task description
            user_message: Original user message

        Returns:
            Dictionary of required information fields
        """
        return {
            "data_source": RequiredInfoField(
                field_name="data_source",
                field_type="string",
                question="What data or information should I analyze?",
                description="Source data or subject for analysis",
            ),
            "analysis_type": RequiredInfoField(
                field_name="analysis_type",
                field_type="enum",
                question="What type of analysis do you need?",
                description="Type of analysis to perform",
                options=[
                    "comparative",
                    "trend",
                    "statistical",
                    "qualitative",
                    "quantitative",
                    "comprehensive",
                ],
            ),
            "focus_areas": RequiredInfoField(
                field_name="focus_areas",
                field_type="list",
                question="Are there specific aspects or metrics you want me to focus on?",
                description="Key areas or metrics to emphasize",
            ),
        }

    def execute_task(self, state: SubAgentState) -> str:
        """
        Execute analysis task using GeminiService.

        Args:
            state: SubAgentState with task and collected information

        Returns:
            Analysis results as a string
        """
        try:
            # Get GeminiService
            gemini_service = get_gemini_service()

            # Build analysis prompt with collected information
            data_source = state.collected_info.get(
                "data_source", "provided information"
            )
            analysis_type = state.collected_info.get("analysis_type", "comprehensive")
            focus_areas = state.collected_info.get("focus_areas", [])

            focus_str = (
                ", ".join(focus_areas)
                if isinstance(focus_areas, list)
                else str(focus_areas)
            )

            analysis_prompt = f"""Conduct a {analysis_type} analysis on the following:

Task: {state.task}

Data/Subject: {data_source}

Focus Areas: {focus_str if focus_str else "all relevant aspects"}

Please provide:
1. Executive summary of key findings
2. Detailed analysis broken down by relevant dimensions
3. Supporting evidence and reasoning
4. Insights and implications
5. Recommendations (if applicable)

Be thorough, objective, and data-driven in your analysis."""

            # Use GeminiService to perform analysis
            user_id = state.metadata.get("user_id", "default")
            # Use circuit breaker for LLM call
            from ..utils.resilience import get_llm_circuit_breaker

            circuit_breaker = get_llm_circuit_breaker()

            result = circuit_breaker.call(
                gemini_service.generate_gemini_response_with_rag,
                prompt=analysis_prompt,
                user_id=user_id,
                persona=self.persona,
            )

            return result

        except Exception as e:
            logger.error("Analysis agent execution failed: %s", e)
            raise
