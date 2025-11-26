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
        Identify required information for data tasks.

        Args:
            task: Task description
            user_message: Original user message

        Returns:
            Dictionary of required information fields
        """
        return {
            "data_format": RequiredInfoField(
                field_name="data_format",
                field_type="enum",
                question="What format is your data in?",
                description="Input data format",
                options=["json", "csv", "xml", "sql", "text", "other"],
            ),
            "operation": RequiredInfoField(
                field_name="operation",
                field_type="enum",
                question="What operation do you need?",
                description="Type of data operation",
                options=[
                    "extraction",
                    "transformation",
                    "cleaning",
                    "validation",
                    "aggregation",
                    "conversion",
                ],
            ),
            "output_format": RequiredInfoField(
                field_name="output_format",
                field_type="string",
                question="What format should the output be in?",
                description="Desired output format",
            ),
        }

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

            data_prompt = f"""Perform data {operation} for the following task:

Task: {state.task}

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
                gemini_service.generate_gemini_response_with_rag,
                prompt=data_prompt,
                user_id=user_id,
                persona=self.persona,
            )

            return result

        except Exception as e:
            logger.error("Data agent execution failed: %s", e)
            raise
