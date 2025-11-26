"""Base sub-agent interface for Legion orchestration."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from ..models import RequiredInfoField, SubAgentState


class BaseSubAgent(ABC):
    """Base class for all sub-agents."""

    def __init__(self, persona: str = "hermes"):
        """
        Initialize base agent.

        Args:
            persona: AI persona to use for this agent (e.g., "hermes", "optimist", "critic")
        """
        self._gemini_service = None
        self._persona = persona
        self._tools: List[Any] = []

    @property
    @abstractmethod
    def agent_id(self) -> str:
        """Unique identifier for this agent."""
        pass

    @property
    @abstractmethod
    def task_types(self) -> List[str]:
        """List of task types this agent can handle."""
        pass

    @property
    def tools(self) -> List[Any]:
        """Tools available to this agent."""
        return self._tools

    @property
    def persona(self) -> str:
        """AI persona used by this agent."""
        return self._persona

    def set_tools(self, tools: List[Any]) -> None:
        """
        Set tools for this agent.

        Args:
            tools: List of tool objects to assign to this agent
        """
        self._tools = tools

    @abstractmethod
    def identify_required_info(
        self, task: str, user_message: str
    ) -> Dict[str, RequiredInfoField]:
        """
        Identify what information is needed to complete the task.

        Args:
            task: The task description
            user_message: The original user message

        Returns:
            Dictionary mapping field names to RequiredInfoField objects
        """
        pass

    @abstractmethod
    def execute_task(self, state: SubAgentState) -> str:
        """
        Execute the agent's task with collected information.

        Args:
            state: SubAgentState containing task and collected information

        Returns:
            Task result as a string
        """
        pass

    def validate_collected_info(
        self,
        collected_info: Dict[str, Any],
        required_info: Dict[str, RequiredInfoField],
    ) -> Tuple[bool, List[str]]:
        """
        Validate collected information.

        Args:
            collected_info: Dictionary of collected information
            required_info: Dictionary of required information fields

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        for field_name, field in required_info.items():
            value = collected_info.get(field_name)

            if value is None:
                continue  # Missing is OK, will ask again

            # Type validation
            if field.field_type == "enum" and field.options:
                if value not in field.options:
                    errors.append(f"{field_name}: '{value}' not in {field.options}")

            # Add more validation as needed

        return len(errors) == 0, errors
