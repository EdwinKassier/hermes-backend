"""Data agent for data processing and transformation tasks."""

import logging
from typing import Dict, List

from app.shared.utils.service_loader import get_gemini_service

from ..models import RequiredInfoField, SubAgentState
from .base import BaseSubAgent

logger = logging.getLogger(__name__)


class DataAgent(BaseSubAgent):
    """Agent for data processing and transformation tasks."""

    @property
    def agent_id(self) -> str:
        """Unique identifier for data agent."""
        return "data_agent"

    @property
    def task_types(self) -> List[str]:
        """Task types this agent can handle."""
        return ["data", "transformation", "processing", "extraction", "cleaning"]

    def identify_required_info(
        self, task: str, user_message: str
    ) -> Dict[str, RequiredInfoField]:
        """
        Intelligently determine if any information is genuinely missing for data tasks.

        Uses LLM to analyze the request and infer missing details from context.
        Only returns required fields when information cannot be reasonably inferred.
        """
        try:
            gemini_service = get_gemini_service()

            analysis_prompt = f"""Analyze this data processing request to determine if any critical information is genuinely missing and cannot be reasonably inferred.

User Request: "{user_message}"
Task: "{task}"

**Your Goal**: Determine what information (if any) is truly needed from the user.

**Inference Philosophy**:
- **Strongly prefer inference over asking questions**
- Use domain knowledge and context to make reasonable assumptions
- Only flag information as "required" if it's genuinely ambiguous or critical

**Potential Information to Consider**:
1. **Data Format**: Can you infer from context or common formats?
2. **Operation Type**: Is it clear from the request?
3. **Output Format**: Can you assume same as input or standard format?

**When to Request Information**:
✅ Request if: "Process data" with no context about what data or operation
✅ Request if: Contradictory or unclear requirements
❌ Do NOT request if: Format can be inferred from context
❌ Do NOT request if: Operation is clear from task description
❌ Do NOT request if: Standard output format is reasonable

**Response Format (JSON)**:
If NO information is needed:
{{
  "needs_info": false,
  "inferred_values": {{
    "data_format": "<what you'll assume>",
    "operation": "<what you understand>",
    "output_format": "<what you'll produce>"
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
                        f"Data agent can proceed without clarification. "
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
                    f"Data agent needs clarification: {list(required_info.keys())}"
                )
                return required_info

        except Exception as e:
            logger.warning(
                f"Failed to analyze info requirements: {e}. Proceeding without clarification."
            )

        return {}

    def execute_task(self, state: SubAgentState) -> str:
        """
        Execute data processing task using GeminiService.

        Args:
            state: SubAgentState with task and collected information

        Returns:
            Data processing results as a string
        """
        try:
            # Get GeminiService
            gemini_service = get_gemini_service()

            # Build data processing prompt with collected information
            data_format = state.collected_info.get("data_format", "unspecified")
            operation = state.collected_info.get("operation", "processing")
            output_format = state.collected_info.get("output_format", "same as input")

            # Incorporate Judge Feedback
            feedback_context = ""
            if state.judge_feedback:
                logger.info(f"Retrying with judge feedback: {state.judge_feedback}")
                feedback_context = f"""
**CRITICAL FEEDBACK FROM PREVIOUS ATTEMPT**:
The previous attempt was rejected. You MUST address the following feedback:
{state.judge_feedback}
"""

            data_prompt = f"""Perform data {operation} for the following task:

Task: {state.task}
{feedback_context}

Input Format: {data_format}
Output Format: {output_format}

Please provide:
1. Step-by-step approach to the data operation
2. Code or instructions for performing the operation
3. Example output showing the transformation
4. Any important notes about data quality or edge cases
5. Recommendations for validation or testing

Focus on data integrity, efficiency, and clarity."""

            # Use GeminiService to process request
            user_id = state.metadata.get("user_id", "default")
            # Use circuit breaker for LLM call
            from ..utils.resilience import get_llm_circuit_breaker

            circuit_breaker = get_llm_circuit_breaker()

            result = circuit_breaker.call(
                gemini_service.generate_gemini_response,
                prompt=data_prompt,
                persona=self.persona,
                user_id=user_id,
            )

            return result

        except Exception as e:
            logger.error("Data agent execution failed: %s", e)
            raise
