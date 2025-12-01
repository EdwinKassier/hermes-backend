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
        Intelligently determine if any information is genuinely missing for analysis tasks.

        Uses LLM to analyze the request and infer missing details from context.
        Only returns required fields when information cannot be reasonably inferred.
        """
        try:
            gemini_service = get_gemini_service()

            analysis_prompt = f"""Analyze this analysis request to determine if any critical information is genuinely missing and cannot be reasonably inferred.

User Request: "{user_message}"
Task: "{task}"

**Your Goal**: Determine what information (if any) is truly needed from the user.

**Inference Philosophy**:
- **Strongly prefer inference over asking questions**
- Use domain knowledge and context to make reasonable assumptions
- Only flag information as "required" if it's genuinely ambiguous or critical

**Potential Information to Consider**:
1. **Data Source**: Is it clear what to analyze from the request?
2. **Analysis Type**: Can you infer from the nature of the request?
3. **Focus Areas**: Can you determine relevant metrics from context?

**When to Request Information**:
✅ Request if: "Analyze" with no context about what to analyze
✅ Request if: Contradictory or unclear requirements
❌ Do NOT request if: Subject of analysis is clear from request
❌ Do NOT request if: Analysis type can be inferred from context
❌ Do NOT request if: Can determine relevant focus areas from domain knowledge

**Response Format (JSON)**:
If NO information is needed:
{{
  "needs_info": false,
  "inferred_values": {{
    "data_source": "<what you'll analyze>",
    "analysis_type": "<type you'll perform>",
    "focus_areas": "<metrics/aspects you'll examine>"
  }},
  "reasoning": "<why you can proceed without asking>"
}}

If information IS needed:
{{
  "needs_info": true,
  "required_fields": [
    {{
      "field_name": "...",
      "field_type": "string|enum|list",
      "question": "...",
      "description": "...",
      "options": ["..."] // only for enum type
    }}
  ],
  "reasoning": "<why you cannot infer this information>"
}}

Analyze and respond with JSON only."""

            # Call LLM directly - already wrapped in try/except with fail-open
            response = gemini_service.generate_gemini_response(
                prompt=analysis_prompt,
                persona=self.persona,
            )

            import json
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group(0))

                if not analysis.get("needs_info", False):
                    logger.info(
                        f"Analysis agent can proceed without clarification. "
                        f"Inferred: {analysis.get('inferred_values', {})}"
                    )
                    return {}

                required_info = {}
                for field_spec in analysis.get("required_fields", []):
                    field_name = field_spec["field_name"]
                    required_info[field_name] = RequiredInfoField(
                        field_name=field_name,
                        field_type=field_spec["field_type"],
                        question=field_spec["question"],
                        description=field_spec.get("description", ""),
                        options=field_spec.get("options"),
                    )

                logger.info(
                    f"Analysis agent needs clarification: {list(required_info.keys())}"
                )
                return required_info

        except Exception as e:
            logger.warning(
                f"Failed to analyze info requirements: {e}. Proceeding without clarification."
            )

        return {}

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

            # Incorporate Judge Feedback
            feedback_context = ""
            if state.judge_feedback:
                logger.info(f"Retrying with judge feedback: {state.judge_feedback}")
                feedback_context = f"""
**CRITICAL FEEDBACK FROM PREVIOUS ATTEMPT**:
The previous attempt was rejected. You MUST address the following feedback:
{state.judge_feedback}
"""

            analysis_prompt = f"""Conduct a {analysis_type} analysis on the following:

Task: {state.task}
{feedback_context}

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
                gemini_service.generate_gemini_response,
                prompt=analysis_prompt,
                persona=self.persona,
                user_id=user_id,
            )

            return result

        except Exception as e:
            logger.error("Analysis agent execution failed: %s", e)
            raise
