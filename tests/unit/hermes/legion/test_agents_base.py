"""Unit tests for BaseSubAgent."""

from unittest.mock import Mock

import pytest

from app.hermes.legion.agents.base import BaseSubAgent
from app.hermes.legion.models import RequiredInfoField, SubAgentState, SubAgentStatus


class ConcreteSubAgent(BaseSubAgent):
    """Concrete implementation for testing."""

    @property
    def agent_id(self) -> str:
        return "test_agent"

    @property
    def task_types(self) -> list[str]:
        return ["test", "example"]

    def identify_required_info(
        self, task: str, user_message: str
    ) -> dict[str, RequiredInfoField]:
        return {
            "test_field": RequiredInfoField(
                field_name="test_field",
                field_type="string",
                question="Test question?",
            )
        }

    def execute_task(self, state: SubAgentState) -> str:
        return "Task completed"


@pytest.mark.unit
class TestBaseSubAgent:
    """Test BaseSubAgent abstract class."""

    def test_base_sub_agent_is_abstract(self):
        """Test that BaseSubAgent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseSubAgent()

    def test_concrete_agent_creation(self):
        """Test creating a concrete agent."""
        agent = ConcreteSubAgent()
        assert agent.agent_id == "test_agent"
        assert "test" in agent.task_types

    def test_agent_requires_agent_id(self):
        """Test agent must have agent_id property."""
        agent = ConcreteSubAgent()
        assert hasattr(agent, "agent_id")
        assert agent.agent_id == "test_agent"

    def test_agent_requires_task_types(self):
        """Test agent must have task_types property."""
        agent = ConcreteSubAgent()
        assert hasattr(agent, "task_types")
        assert isinstance(agent.task_types, list)

    def test_identify_required_info(self):
        """Test identify_required_info method."""
        agent = ConcreteSubAgent()
        required_info = agent.identify_required_info("test task", "user message")
        assert "test_field" in required_info
        assert isinstance(required_info["test_field"], RequiredInfoField)

    def test_execute_task(self):
        """Test execute_task method."""
        agent = ConcreteSubAgent()
        state = SubAgentState(
            agent_id="test_agent",
            task="test task",
            task_type="test",
            triggering_message="user message",
        )
        result = agent.execute_task(state)
        assert result == "Task completed"

    def test_validate_collected_info_valid(self):
        """Test validation with valid collected info."""
        agent = ConcreteSubAgent()
        collected_info = {"test_field": "value"}
        required_info = {
            "test_field": RequiredInfoField(
                field_name="test_field",
                field_type="string",
                question="Test?",
            )
        }
        is_valid, errors = agent.validate_collected_info(collected_info, required_info)
        assert is_valid
        assert len(errors) == 0

    def test_validate_collected_info_enum_invalid(self):
        """Test validation with invalid enum value."""
        agent = ConcreteSubAgent()
        collected_info = {"depth": "invalid"}
        required_info = {
            "depth": RequiredInfoField(
                field_name="depth",
                field_type="enum",
                question="Depth?",
                options=["brief", "moderate"],
            )
        }
        is_valid, errors = agent.validate_collected_info(collected_info, required_info)
        assert not is_valid
        assert len(errors) > 0

    def test_validate_collected_info_enum_valid(self):
        """Test validation with valid enum value."""
        agent = ConcreteSubAgent()
        collected_info = {"depth": "brief"}
        required_info = {
            "depth": RequiredInfoField(
                field_name="depth",
                field_type="enum",
                question="Depth?",
                options=["brief", "moderate"],
            )
        }
        is_valid, errors = agent.validate_collected_info(collected_info, required_info)
        assert is_valid
        assert len(errors) == 0

    def test_validate_collected_info_missing_field(self):
        """Test validation with missing field (should be OK)."""
        agent = ConcreteSubAgent()
        collected_info = {}
        required_info = {
            "test_field": RequiredInfoField(
                field_name="test_field",
                field_type="string",
                question="Test?",
            )
        }
        is_valid, errors = agent.validate_collected_info(collected_info, required_info)
        # Missing fields are OK (will ask again)
        assert is_valid
        assert len(errors) == 0
