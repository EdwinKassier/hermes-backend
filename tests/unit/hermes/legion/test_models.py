"""Unit tests for Legion models."""

from datetime import datetime

import pytest

from app.hermes.legion.models import RequiredInfoField, SubAgentState, SubAgentStatus


@pytest.mark.unit
class TestSubAgentStatus:
    """Test SubAgentStatus enum."""

    def test_enum_values(self):
        """Test all enum values are present."""
        assert SubAgentStatus.CREATED == "created"
        assert SubAgentStatus.AWAITING_USER_INPUT == "awaiting_user_input"
        assert SubAgentStatus.PROCESSING == "processing"
        assert SubAgentStatus.COMPLETED == "completed"
        assert SubAgentStatus.ERROR == "error"
        assert SubAgentStatus.CANCELLED == "cancelled"

    def test_enum_string_based(self):
        """Test enum values are strings."""
        assert isinstance(SubAgentStatus.CREATED.value, str)
        assert isinstance(SubAgentStatus.AWAITING_USER_INPUT.value, str)


@pytest.mark.unit
class TestRequiredInfoField:
    """Test RequiredInfoField model."""

    def test_required_info_field_creation(self):
        """Test creating a RequiredInfoField."""
        field = RequiredInfoField(
            field_name="time_period",
            field_type="string",
            question="What time period?",
        )
        assert field.field_name == "time_period"
        assert field.field_type == "string"
        assert field.question == "What time period?"
        assert field.description is None
        assert field.options is None

    def test_required_info_field_with_options(self):
        """Test RequiredInfoField with enum options."""
        field = RequiredInfoField(
            field_name="depth",
            field_type="enum",
            question="How detailed?",
            options=["brief", "moderate", "comprehensive"],
        )
        assert field.options == ["brief", "moderate", "comprehensive"]

    def test_required_info_field_serialization(self):
        """Test RequiredInfoField serialization."""
        field = RequiredInfoField(
            field_name="test_field",
            field_type="string",
            question="Test question?",
            description="Test description",
        )
        data = field.model_dump()
        assert data["field_name"] == "test_field"
        assert data["field_type"] == "string"
        assert data["question"] == "Test question?"
        assert data["description"] == "Test description"


@pytest.mark.unit
class TestSubAgentState:
    """Test SubAgentState model."""

    def test_sub_agent_state_creation(self):
        """Test creating a SubAgentState."""
        state = SubAgentState(
            agent_id="test_agent",
            task="Test task",
            task_type="research",
            triggering_message="User message",
        )
        assert state.agent_id == "test_agent"
        assert state.status == SubAgentStatus.CREATED
        assert state.task == "Test task"
        assert state.task_type == "research"
        assert isinstance(state.created_at, datetime)
        assert isinstance(state.last_updated, datetime)

    def test_sub_agent_state_serialization(self):
        """Test SubAgentState serialization."""
        state = SubAgentState(
            agent_id="test_agent",
            task="Test task",
            task_type="research",
            triggering_message="User message",
            status=SubAgentStatus.AWAITING_USER_INPUT,
        )
        data = state.to_dict()
        assert data["agent_id"] == "test_agent"
        assert data["status"] == "awaiting_user_input"  # Enum converted to string
        assert data["task"] == "Test task"
        # Datetime may be serialized by model_dump or remain as datetime
        # Check that it's either a string or datetime
        assert isinstance(data["created_at"], (str, datetime))

    def test_sub_agent_state_deserialization(self):
        """Test SubAgentState deserialization."""
        data = {
            "agent_id": "test_agent",
            "status": "awaiting_user_input",
            "task": "Test task",
            "task_type": "research",
            "triggering_message": "User message",
            "created_at": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat(),
            "required_info": {},
            "collected_info": {},
            "pending_questions": [],
            "metadata": {},
        }
        state = SubAgentState.from_dict(data)
        assert state.agent_id == "test_agent"
        assert state.status == SubAgentStatus.AWAITING_USER_INPUT
        assert isinstance(state.created_at, datetime)

    def test_sub_agent_state_from_dict_with_enum(self):
        """Test deserialization handles enum conversion."""
        data = {
            "agent_id": "test_agent",
            "status": "processing",
            "task": "Test",
            "task_type": "research",
            "triggering_message": "Test",
        }
        state = SubAgentState.from_dict(data)
        assert state.status == SubAgentStatus.PROCESSING

    def test_sub_agent_state_from_dict_invalid_status(self):
        """Test deserialization handles invalid status."""
        data = {
            "agent_id": "test_agent",
            "status": "invalid_status",
            "task": "Test",
            "task_type": "research",
            "triggering_message": "Test",
        }
        state = SubAgentState.from_dict(data)
        # Should default to CREATED
        assert state.status == SubAgentStatus.CREATED

    def test_sub_agent_state_with_required_info(self):
        """Test SubAgentState with required info fields."""
        required_info = {
            "time_period": RequiredInfoField(
                field_name="time_period",
                field_type="string",
                question="What time period?",
            )
        }
        state = SubAgentState(
            agent_id="test_agent",
            task="Test task",
            task_type="research",
            triggering_message="User message",
            required_info=required_info,
        )
        assert "time_period" in state.required_info
        assert isinstance(state.required_info["time_period"], RequiredInfoField)

    def test_sub_agent_state_serialization_with_required_info(self):
        """Test serialization preserves required info."""
        required_info = {
            "time_period": RequiredInfoField(
                field_name="time_period",
                field_type="string",
                question="What time period?",
            )
        }
        state = SubAgentState(
            agent_id="test_agent",
            task="Test task",
            task_type="research",
            triggering_message="User message",
            required_info=required_info,
        )
        data = state.to_dict()
        assert "time_period" in data["required_info"]
        assert isinstance(data["required_info"]["time_period"], dict)

    def test_sub_agent_state_deserialization_with_required_info(self):
        """Test deserialization recreates required info fields."""
        data = {
            "agent_id": "test_agent",
            "status": "created",
            "task": "Test task",
            "task_type": "research",
            "triggering_message": "User message",
            "required_info": {
                "time_period": {
                    "field_name": "time_period",
                    "field_type": "string",
                    "question": "What time period?",
                }
            },
            "collected_info": {},
            "pending_questions": [],
            "metadata": {},
        }
        state = SubAgentState.from_dict(data)
        assert "time_period" in state.required_info
        assert isinstance(state.required_info["time_period"], RequiredInfoField)
        assert state.required_info["time_period"].field_name == "time_period"

    def test_sub_agent_state_validation(self):
        """Test SubAgentState validation."""
        # Should not raise with valid data
        state = SubAgentState(
            agent_id="test_agent",
            task="Test task",
            task_type="research",
            triggering_message="User message",
        )
        assert state.agent_id == "test_agent"
