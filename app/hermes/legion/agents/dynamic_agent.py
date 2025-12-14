"""Dynamic agent that can be configured with capabilities, prompts, and personas."""

import json
import logging
from typing import Any, Dict, List, Optional

from app.shared.utils.service_loader import get_gemini_service

from ..models import RequiredInfoField, SubAgentState
from .base import BaseSubAgent

logger = logging.getLogger(__name__)


class DynamicAgent(BaseSubAgent):
    """
    A flexible agent that can be configured dynamically with different capabilities,
    prompts, and personas instead of having hardcoded behavior.

    This enables the system to create multiple variants of agents (e.g., different
    coding styles, analysis approaches) without creating new classes.
    """

    def __init__(
        self,
        agent_id: str,
        task_types: List[str],
        persona: str = "hermes",
        capabilities: Optional[Dict[str, Any]] = None,
        prompts: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """
        Initialize dynamic agent with configuration.

        Args:
            agent_id: Unique identifier for this agent type
            task_types: List of task types this agent can handle
            persona: AI persona to use for this agent
            capabilities: Dictionary defining agent capabilities
            prompts: Dictionary of prompt templates for different operations
            **kwargs: Additional configuration options
        """
        super().__init__(persona=persona)

        self._dynamic_agent_id = agent_id
        self._dynamic_task_types = task_types
        self._capabilities = capabilities or {}
        self._prompts = prompts or {}
        self._config = kwargs

        # Validate required prompts
        required_prompts = ["identify_required_info", "execute_task"]
        for prompt_name in required_prompts:
            if prompt_name not in self._prompts:
                raise ValueError(f"Required prompt '{prompt_name}' not provided")

    @property
    def agent_id(self) -> str:
        """Unique identifier for this dynamic agent."""
        return self._dynamic_agent_id

    @property
    def task_types(self) -> List[str]:
        """Task types this agent can handle."""
        return self._dynamic_task_types

    def identify_required_info(
        self, task: str, user_message: str
    ) -> Dict[str, RequiredInfoField]:
        """
        Intelligently determine if any information is needed for the task.

        Uses the configured prompt template to analyze the request and determine
        what information (if any) is required.
        """
        try:
            gemini_service = get_gemini_service()

            # Get the analysis prompt template and fill it
            prompt_template = self._prompts["identify_required_info"]
            analysis_prompt = prompt_template.format(
                task=task,
                user_message=user_message,
                capabilities=json.dumps(self._capabilities, indent=2),
                **self._config,
            )

            # Use Legion persona context if available, otherwise fall back to agent persona
            from ..utils.persona_context import get_current_legion_persona

            persona = get_current_legion_persona() or self.persona

            # Call LLM directly - already wrapped in try/except with fail-open
            response = gemini_service.generate_gemini_response(
                prompt=analysis_prompt,
                persona=persona,
            )

            # Extract JSON from response - handle nested JSON properly
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
                            f"Dynamic agent {self.agent_id} can proceed without clarification. "
                            f"Inferred: {analysis.get('inferred_values', {})}"
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
                        f"Dynamic agent {self.agent_id} needs clarification: {list(required_info.keys())}"
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
                f"Failed to analyze info requirements for {self.agent_id}: {e}. Proceeding without clarification."
            )

        return {}

    def execute_task(self, state: SubAgentState) -> str:
        """
        Execute the task using the configured prompt template and capabilities.

        Args:
            state: SubAgentState with task and collected information

        Returns:
            Task result as a string
        """
        try:
            # Get GeminiService
            gemini_service = get_gemini_service()

            # Build execution prompt from template
            prompt_template = self._prompts["execute_task"]

            # Prepare template variables
            template_vars = {
                "task": state.task,
                "capabilities": json.dumps(self._capabilities, indent=2),
                "collected_info": json.dumps(state.collected_info, indent=2),
                "judge_feedback": state.judge_feedback or "",
                "user_id": state.metadata.get("user_id", "default"),
                **self._config,
            }

            # Add tool context if tools are available
            if self._tools:
                tool_names = [getattr(t, "name", str(t)) for t in self._tools]
                template_vars[
                    "tool_context"
                ] = f"""
**Available Tools**: You have access to the following tools:
{', '.join(tool_names)}

Use these tools when they would enhance your performance."""
            else:
                template_vars["tool_context"] = ""

            # Format the prompt
            execution_prompt = prompt_template.format(**template_vars)

            # Use circuit breaker for LLM call
            from ..utils.resilience import get_llm_circuit_breaker

            circuit_breaker = get_llm_circuit_breaker()

            # Use Legion persona context if available, otherwise fall back to agent persona
            from ..utils.persona_context import get_current_legion_persona

            persona = get_current_legion_persona() or self.persona

            result = circuit_breaker.call(
                gemini_service.generate_gemini_response,
                prompt=execution_prompt,
                persona=persona,
                user_id=state.metadata.get("user_id", "default"),
            )

            return result

        except Exception as e:
            logger.error(f"Dynamic agent {self.agent_id} execution failed: %s", e)
            raise
