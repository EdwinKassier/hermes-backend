"""Base sub-agent interface for Legion orchestration."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

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
        """
        Agent type identifier.

        Returns a string identifying the agent type (e.g., "research_agent", "code_agent").
        Note: For unique instance identification, the factory generates unique IDs by appending
        a timestamp to this value when creating AgentInfo objects.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def task_types(self) -> List[str]:
        """List of task types this agent can handle."""
        raise NotImplementedError

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
            Dictionary mapping field names to RequiredInfoField objects.
            Returns empty dict if no additional information is needed.
        """
        raise NotImplementedError

    @abstractmethod
    def execute_task(self, state: SubAgentState) -> str:
        """
        Execute the agent's task with collected information.

        Args:
            state: SubAgentState containing task and collected information

        Returns:
            Task result as a string

        Raises:
            Exception: If task execution fails. Subclasses should raise specific exceptions.
        """
        raise NotImplementedError

    def validate_collected_info(
        self,
        collected_info: Dict[str, Any],
        required_info: Dict[str, RequiredInfoField],
    ) -> Tuple[bool, List[str]]:
        """
        Validate collected information against required fields.

        Args:
            collected_info: Dictionary of collected information
            required_info: Dictionary of required information fields

        Returns:
            Tuple of (is_valid, error_messages).
            Missing fields (None values) are allowed and won't cause validation errors,
            as they will be requested again if needed.

        Note:
            Currently validates:
            - Enum values must be in the allowed options list
            - Future: Can be extended to validate other field types (list, number, boolean)
        """
        errors = []

        for field_name, field in required_info.items():
            value = collected_info.get(field_name)

            # Missing values are OK - will be requested again if needed
            if value is None:
                continue

            # Validate enum types
            if field.field_type == "enum":
                if not field.options:
                    errors.append(
                        f"{field_name}: enum field type requires 'options' to be defined"
                    )
                elif value not in field.options:
                    errors.append(
                        f"{field_name}: '{value}' not in allowed options {field.options}"
                    )

            # Validate list types
            elif field.field_type == "list":
                if not isinstance(value, list):
                    errors.append(
                        f"{field_name}: expected list type, got {type(value).__name__}"
                    )

            # Validate number types (int or float)
            elif field.field_type == "number":
                if not isinstance(value, (int, float)):
                    errors.append(
                        f"{field_name}: expected number type, got {type(value).__name__}"
                    )

            # Validate boolean types
            elif field.field_type == "boolean":
                if not isinstance(value, bool):
                    errors.append(
                        f"{field_name}: expected boolean type, got {type(value).__name__}"
                    )

            # String types are always valid (any value can be string)
            # Additional string validation can be added via field.validation if needed

        return len(errors) == 0, errors
