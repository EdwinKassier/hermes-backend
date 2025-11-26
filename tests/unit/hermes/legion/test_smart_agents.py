from unittest.mock import Mock, patch

import pytest

from app.hermes.legion.agents.research_agent import ResearchAgent


@pytest.fixture
def research_agent():
    return ResearchAgent()


@patch("app.hermes.legion.agents.research_agent.get_gemini_service")
def test_smart_info_short_message(mock_get_service, research_agent):
    # Setup
    # Short message should bypass LLM check
    message = "research AI"

    # Execute
    required = research_agent.identify_required_info("research AI", message)

    # Verify
    assert "time_period" in required
    assert "topics" in required
    assert "depth" in required
    mock_get_service.assert_not_called()


@patch("app.hermes.legion.agents.research_agent.get_gemini_service")
def test_smart_info_all_present(mock_get_service, research_agent):
    # Setup
    mock_service = Mock()
    mock_get_service.return_value = mock_service

    # Mock LLM response saying everything is present
    mock_service.generate_gemini_response.return_value = (
        '{"time_period": true, "topics": true, "depth": true}'
    )

    message = "Research AI developments in 2023 with a focus on LLMs, provide a comprehensive report."

    # Execute
    required = research_agent.identify_required_info(message, message)

    # Verify
    assert len(required) == 0  # No info needed
    mock_service.generate_gemini_response.assert_called_once()


@patch("app.hermes.legion.agents.research_agent.get_gemini_service")
def test_smart_info_partial_missing(mock_get_service, research_agent):
    # Setup
    mock_service = Mock()
    mock_get_service.return_value = mock_service

    # Mock LLM response saying depth is missing
    mock_service.generate_gemini_response.return_value = (
        '{"time_period": true, "topics": true, "depth": false}'
    )

    message = "Research AI developments in 2023 with a focus on LLMs."

    # Execute
    required = research_agent.identify_required_info(message, message)

    # Verify
    assert len(required) == 1
    assert "depth" in required
    assert "time_period" not in required
    mock_service.generate_gemini_response.assert_called_once()


@patch("app.hermes.legion.agents.research_agent.get_gemini_service")
def test_smart_info_llm_failure(mock_get_service, research_agent):
    # Setup
    mock_service = Mock()
    mock_get_service.return_value = mock_service

    # Mock LLM failure
    mock_service.generate_gemini_response.side_effect = Exception("LLM Error")

    message = "Research AI developments in 2023 with a focus on LLMs."

    # Execute
    required = research_agent.identify_required_info(message, message)

    # Verify fallback to asking all questions
    assert "time_period" in required
    assert "topics" in required
    assert "depth" in required
