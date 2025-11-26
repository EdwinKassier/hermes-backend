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
        Identify required information for code generation tasks.

        Args:
            task: Task description
            user_message: Original user message

        Returns:
            Dictionary of required information fields
        """
        # Check if task mentions specific language/framework
        task_lower = task.lower()
        needs_language = not any(
            lang in task_lower
            for lang in [
                "python",
                "javascript",
                "java",
                "go",
                "rust",
                "c++",
                "typescript",
            ]
        )

        required_info = {}

        if needs_language:
            required_info["language"] = RequiredInfoField(
                field_name="language",
                field_type="enum",
                question="What programming language should I use?",
                description="Programming language for implementation",
                options=[
                    "python",
                    "javascript",
                    "typescript",
                    "java",
                    "go",
                    "rust",
                    "c++",
                ],
            )

        # Always ask for requirements/specs
        required_info["requirements"] = RequiredInfoField(
            field_name="requirements",
            field_type="string",
            question="What are the specific requirements or functionality you need?",
            description="Detailed requirements for the code",
        )

        # Ask for any constraints
        required_info["constraints"] = RequiredInfoField(
            field_name="constraints",
            field_type="string",
            question="Are there any constraints or requirements (e.g., libraries to use/avoid, performance needs)?",
            description="Technical constraints or preferences",
        )

        return required_info

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

            code_prompt = f"""Generate clean, well-documented code for the following task:

Task: {state.task}

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
                gemini_service.generate_gemini_response_with_rag,
                prompt=code_prompt,
                user_id=user_id,
                persona=self.persona,
            )

            return result

        except Exception as e:
            logger.error("Code agent execution failed: %s", e)
            raise
