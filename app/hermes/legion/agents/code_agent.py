"""Code agent for code generation and programming tasks."""

import json
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

            # Extract JSON from response - handle nested JSON properly
            # Try to find the outermost JSON object by counting braces
            json_start = response.find("{")
            if json_start != -1:
                brace_count = 0
                json_end = json_start
                for i, char in enumerate(response[json_start:], start=json_start):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break

                json_str = response[json_start:json_end]
                try:
                    analysis = json.loads(json_str)

                    # Validate JSON structure
                    if not isinstance(analysis, dict):
                        logger.warning(
                            "Parsed JSON is not a dictionary, proceeding without clarification"
                        )
                        return {}

                    if not analysis.get("needs_info", False):
                        logger.info(
                            "Code agent can proceed without clarification. "
                            "Inferred: %s",
                            analysis.get("inferred_values", {}),
                        )
                        return {}

                    # Validate required_fields structure
                    required_fields = analysis.get("required_fields", [])
                    if not isinstance(required_fields, list):
                        logger.warning(
                            "required_fields is not a list, proceeding without clarification"
                        )
                        return {}

                    required_info = {}
                    for field_spec in required_fields:
                        if not isinstance(field_spec, dict):
                            logger.warning(
                                "Skipping invalid field_spec: %s", field_spec
                            )
                            continue

                        field_name = field_spec.get("field_name")
                        if not field_name:
                            logger.warning(
                                "Skipping field_spec without field_name: %s", field_spec
                            )
                            continue

                        try:
                            required_info[field_name] = RequiredInfoField(
                                field_name=field_name,
                                field_type=field_spec.get("field_type", "string"),
                                question=field_spec.get("question", ""),
                                description=field_spec.get("description", ""),
                                options=field_spec.get("options"),
                            )
                        except Exception as field_error:
                            logger.warning(
                                "Failed to create RequiredInfoField for %s: %s",
                                field_name,
                                field_error,
                            )
                            continue

                    logger.info(
                        "Code agent needs clarification: %s",
                        list(required_info.keys()),
                    )
                    return required_info
                except json.JSONDecodeError as json_error:
                    logger.warning(
                        "Failed to parse JSON from response: %s. Response: %s",
                        json_error,
                        response[:200],
                    )
            else:
                logger.warning("No JSON object found in response: %s", response[:200])

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

            # Build tool context if tools are available
            tool_context = ""
            if self._tools:
                tool_names = [getattr(t, "name", str(t)) for t in self._tools]
                tool_context = f"""

**Available Tools**: You have access to the following tools that may help with code generation:
{', '.join(tool_names)}

- Use the 'calculator' tool for mathematical calculations or numerical computations
- Use the 'database_query' tool to retrieve relevant data or examples from the database

Use these tools when they would enhance the accuracy or completeness of your code."""
            code_prompt = f"""Generate clean, well-documented code for the following task:

Task: {state.task}
{feedback_context}

Requirements: {requirements}

Programming Language: {language}

Constraints: {constraints}
{tool_context}

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
