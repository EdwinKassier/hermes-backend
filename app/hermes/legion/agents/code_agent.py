"""Code agent for code generation and programming tasks."""

import logging
from typing import Dict, List

from app.shared.utils.service_loader import get_gemini_service

from ..models import RequiredInfoField, SubAgentState
from .base import BaseSubAgent

logger = logging.getLogger(__name__)


class CodeAgent(BaseSubAgent):
    """Agent for code generation and programming tasks."""

    @property
    def agent_id(self) -> str:
        """Unique identifier for code agent."""
        return "code_agent"

    @property
    def task_types(self) -> List[str]:
        """Task types this agent can handle."""
        return ["code", "programming", "implementation", "debugging"]

    def identify_required_info(
        self, task: str, user_message: str
    ) -> Dict[str, RequiredInfoField]:
        """
        Intelligently determine if any information is genuinely missing for code generation.

        Uses LLM to analyze the request and infer missing details from context.
        Only returns required fields when information cannot be reasonably inferred.
        """
        try:
            gemini_service = get_gemini_service()

            analysis_prompt = f"""Analyze this code generation request to determine if any critical information is genuinely missing and cannot be reasonably inferred.

User Request: "{user_message}"
Task: "{task}"

**Your Goal**: Determine what information (if any) is truly needed from the user.

**Inference Philosophy**:
- **Strongly prefer inference over asking questions**
- Use domain knowledge and context to make reasonable assumptions
- Only flag information as "required" if it's genuinely ambiguous or critical

**Potential Information to Consider**:
1. **Programming Language**: Can you infer from context or use a widely-applicable default?
2. **Requirements/Functionality**: Are they clear from the request?
3. **Constraints**: Can you proceed with standard best practices?

**When to Request Information**:
✅ Request if: "Write code" with absolutely no context about what to build
✅ Request if: Contradictory or unclear requirements
❌ Do NOT request if: Language can be inferred from context or Python is reasonable default
❌ Do NOT request if: Requirements are clear from the task description
❌ Do NOT request if: Standard constraints/best practices apply

**Response Format (JSON)**:
If NO information is needed:
{{
  "needs_info": false,
  "inferred_values": {{
    "language": "<what you'll use>",
    "requirements": "<what you understand>",
    "constraints": "<assumptions you'll make>"
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
                        f"Code agent can proceed without clarification. "
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
                    f"Code agent needs clarification: {list(required_info.keys())}"
                )
                return required_info

        except Exception as e:
            logger.warning(
                f"Failed to analyze info requirements: {e}. Proceeding without clarification."
            )

        return {}

    def execute_task(self, state: SubAgentState) -> str:
        """
        Execute code generation task using GeminiService.

        Args:
            state: SubAgentState with task and collected information

        Returns:
            Generated code with explanations as a string
        """
        try:
            # Get GeminiService
            gemini_service = get_gemini_service()

            # Build code generation prompt with collected information
            language = state.collected_info.get("language", "python")
            requirements = state.collected_info.get("requirements", state.task)
            constraints = state.collected_info.get("constraints", "none")

            # Incorporate Judge Feedback
            feedback_context = ""
            if state.judge_feedback:
                logger.info(f"Retrying with judge feedback: {state.judge_feedback}")
                feedback_context = f"""
**CRITICAL FEEDBACK FROM PREVIOUS ATTEMPT**:
The previous attempt was rejected. You MUST address the following feedback:
{state.judge_feedback}
"""

            code_prompt = f"""Generate clean, well-documented code for the following task:

Task: {state.task}
{feedback_context}

Requirements: {requirements}

Programming Language: {language}

Constraints: {constraints}

Please provide:
1. Complete, working code with proper structure
2. Clear comments explaining key sections
3. Usage examples if applicable
4. Any important notes about the implementation

Focus on code quality, readability, and best practices."""

            # Use GeminiService to generate code
            user_id = state.metadata.get("user_id", "default")
            # Use circuit breaker for LLM call
            from ..utils.resilience import get_llm_circuit_breaker

            circuit_breaker = get_llm_circuit_breaker()

            result = circuit_breaker.call(
                gemini_service.generate_gemini_response,
                prompt=code_prompt,
                persona=self.persona,
                user_id=user_id,
            )

            return result

        except Exception as e:
            logger.error("Code agent execution failed: %s", e)
            raise
