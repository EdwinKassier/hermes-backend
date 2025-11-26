"""Research agent for conducting research tasks."""

import logging
from typing import Dict, List

from app.shared.utils.service_loader import get_gemini_service

from ..models import RequiredInfoField, SubAgentState
from .base import BaseSubAgent

logger = logging.getLogger(__name__)


class ResearchAgent(BaseSubAgent):
    """Agent for conducting research tasks."""

    @property
    def agent_id(self) -> str:
        """Unique identifier for research agent."""
        return "research_agent"

    @property
    def task_types(self) -> List[str]:
        """Task types this agent can handle."""
        return ["research", "investigation", "analysis"]

    def identify_required_info(
        self, task: str, user_message: str
    ) -> Dict[str, RequiredInfoField]:
        """
        Identify required information for research tasks.

        Uses LLM to check if information is already present in the user message.
        """
        required_fields = {
            "time_period": RequiredInfoField(
                field_name="time_period",
                field_type="string",
                question="What time period should I focus on? (e.g., last 6 months, 2024, all time)",
                description="Time period for the research",
            ),
            "topics": RequiredInfoField(
                field_name="topics",
                field_type="list",
                question="Are there any specific topics or areas you'd like me to focus on?",
                description="Specific topics to research",
            ),
            "depth": RequiredInfoField(
                field_name="depth",
                field_type="enum",
                question="How detailed should the research be?",
                options=["brief", "moderate", "comprehensive"],
                description="Depth of research",
            ),
        }

        # If message is very short, just return all fields to be safe/fast
        if len(user_message.split()) < 5:
            return required_fields

        # Use LLM to check what's already provided
        try:
            gemini_service = get_gemini_service()

            check_prompt = f"""Analyze the following user request and determine if the required information is already provided.

Request: "{user_message}"

Required Information:
1. time_period: Timeframe for research (e.g., "last year", "2023", "recent")
2. topics: Specific sub-topics to focus on
3. depth: Level of detail (brief, moderate, comprehensive)

Return a JSON object with boolean values indicating if the information is present.
Example: {{"time_period": true, "topics": false, "depth": true}}
"""
            # We use a simple generation call here. In a real prod system, we might use structured output or a specialized tool.
            # For now, we'll do a heuristic check or a lightweight LLM call.
            # actually, let's use a simpler heuristic for speed and reliability without extra LLM calls if possible,
            # or just assume if it's long enough it might have it.
            # But the requirement is "Smarter". Let's try to extract it.

            # For this implementation, we will use a simplified heuristic to avoid latency of another LLM call
            # unless we are sure we want that overhead.
            # The prompt above implies an LLM call. Let's do it properly with the service.

            # Use circuit breaker for LLM call
            from ..utils.resilience import get_llm_circuit_breaker

            circuit_breaker = get_llm_circuit_breaker()

            analysis = circuit_breaker.call(
                gemini_service.generate_gemini_response,
                prompt=check_prompt,
                persona=self.persona,
            )

            import json
            import re

            # Extract JSON from response
            json_match = re.search(r"\{.*\}", analysis, re.DOTALL)
            if json_match:
                presence_data = json.loads(json_match.group(0))

                # Filter out fields that are already present
                missing_fields = {}
                for field_name, field in required_fields.items():
                    if not presence_data.get(field_name, False):
                        missing_fields[field_name] = field

                return missing_fields

        except Exception as e:
            logger.warning(
                f"Failed to analyze info presence: {e}. Falling back to asking all questions."
            )

        return required_fields

    def execute_task(self, state: SubAgentState) -> str:
        """
        Execute research task using GeminiService.

        Args:
            state: SubAgentState with task and collected information

        Returns:
            Research result as a string
        """
        try:
            # Get GeminiService
            gemini_service = get_gemini_service()

            # Build research prompt with collected information
            time_period = state.collected_info.get("time_period", "all time")
            topics = state.collected_info.get("topics", [])
            depth = state.collected_info.get("depth", "moderate")

            topics_str = ", ".join(topics) if isinstance(topics, list) else str(topics)

            research_prompt = f"""Conduct a {depth} research on: {state.task}

Time period: {time_period}
Specific topics: {topics_str}

Provide a comprehensive research summary with key findings, relevant information, and sources if available."""

            # Use GeminiService to perform research
            # Note: This uses the user_id from state metadata if available
            user_id = state.metadata.get("user_id", "default")
            # Use circuit breaker for LLM call
            from ..utils.resilience import get_llm_circuit_breaker

            circuit_breaker = get_llm_circuit_breaker()

            result = circuit_breaker.call(
                gemini_service.generate_gemini_response,
                prompt=research_prompt,
                user_id=user_id,
                persona=self.persona,
            )

            return result

        except Exception as e:
            logger.error("Research agent execution failed: %s", e)
            raise
