"""Unit tests for ResearchAgent."""

from unittest.mock import Mock, patch

import pytest

from app.hermes.legion.agents.research_agent import ResearchAgent
from app.hermes.legion.models import SubAgentState, SubAgentStatus


@pytest.mark.unit
class TestResearchAgent:
    """Test ResearchAgent."""

    def test_research_agent_creation(self):
        """Test creating ResearchAgent."""
        agent = ResearchAgent()
        assert agent.agent_id == "research_agent"

    def test_research_agent_task_types(self):
        """Test ResearchAgent task types."""
        agent = ResearchAgent()
        assert "research" in agent.task_types
        assert "investigation" in agent.task_types
        assert "analysis" in agent.task_types

    def test_identify_required_info(self):
        """Test identifying required information."""
        agent = ResearchAgent()
        required_info = agent.identify_required_info(
            "Research quantum computing", "User message"
        )
        assert "time_period" in required_info
        assert "topics" in required_info
        assert "depth" in required_info

    def test_identify_required_info_fields(self):
        """Test required info fields are correct."""
        agent = ResearchAgent()
        required_info = agent.identify_required_info("Test task", "User message")
        assert required_info["time_period"].field_type == "string"
        assert required_info["topics"].field_type == "list"
        assert required_info["depth"].field_type == "enum"
        assert required_info["depth"].options == ["brief", "moderate", "comprehensive"]

    @patch("app.hermes.legion.agents.research_agent.get_gemini_service")
    def test_execute_task(self, mock_get_gemini):
        """Test executing research task."""
        # Mock GeminiService
        mock_gemini = Mock()
        mock_gemini.generate_gemini_response.return_value = "Research result"
        mock_get_gemini.return_value = mock_gemini

        agent = ResearchAgent()
        state = SubAgentState(
            agent_id="research_agent",
            task="Research quantum computing",
            task_type="research",
            triggering_message="Research quantum computing",
            collected_info={
                "time_period": "last 6 months",
                "topics": ["quantum algorithms"],
                "depth": "comprehensive",
            },
            metadata={"user_id": "test_user"},
        )

        result = agent.execute_task(state)
        assert result == "Research result"
        mock_gemini.generate_gemini_response.assert_called_once()

    @patch("app.hermes.legion.agents.research_agent.get_gemini_service")
    def test_execute_task_with_defaults(self, mock_get_gemini):
        """Test executing task with default collected info."""
        mock_gemini = Mock()
        mock_gemini.generate_gemini_response.return_value = "Result"
        mock_get_gemini.return_value = mock_gemini

        agent = ResearchAgent()
        state = SubAgentState(
            agent_id="research_agent",
            task="Research task",
            task_type="research",
            triggering_message="Research task",
            collected_info={},  # Empty collected info
            metadata={"user_id": "test_user"},
        )

        result = agent.execute_task(state)
        assert result == "Result"
        # Should use defaults
        call_args = mock_gemini.generate_gemini_response.call_args
        assert "all time" in call_args[1]["prompt"]  # Default time_period

    @patch("app.hermes.legion.agents.research_agent.get_gemini_service")
    def test_execute_task_error_handling(self, mock_get_gemini):
        """Test error handling in task execution."""
        mock_gemini = Mock()
        mock_gemini.generate_gemini_response.side_effect = Exception("API Error")
        mock_get_gemini.return_value = mock_gemini

        agent = ResearchAgent()
        state = SubAgentState(
            agent_id="research_agent",
            task="Research task",
            task_type="research",
            triggering_message="Research task",
            metadata={"user_id": "test_user"},
        )

        with pytest.raises(Exception):
            agent.execute_task(state)
