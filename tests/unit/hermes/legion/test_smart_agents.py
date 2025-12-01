from unittest.mock import Mock, patch

import pytest

from app.hermes.legion.agents.research_agent import ResearchAgent


@pytest.fixture
def research_agent():
    return ResearchAgent()


@patch("app.hermes.legion.agents.research_agent.get_gemini_service")
def test_smart_info_can_infer(mock_get_service, research_agent):
    """Test that agent can infer and proceed without asking."""
    # Setup
    mock_service = Mock()
    mock_get_service.return_value = mock_service

    # Mock LLM response saying it can infer everything
    mock_service.generate_gemini_response.return_value = '{"needs_info": false, "inferred_values": {"time_period": "recent", "topics": "AI", "depth": "moderate"}, "reasoning": "Can infer from context"}'

    message = "research AI"

    # Execute
    required = research_agent.identify_required_info("research AI", message)

    # Verify - should return empty dict when can infer
    assert len(required) == 0
    mock_service.generate_gemini_response.assert_called_once()


@patch("app.hermes.legion.agents.research_agent.get_gemini_service")
def test_smart_info_all_present(mock_get_service, research_agent):
    # Setup
    mock_service = Mock()
    mock_get_service.return_value = mock_service

    # Mock LLM response saying everything can be inferred
    mock_service.generate_gemini_response.return_value = '{"needs_info": false, "inferred_values": {"time_period": "2023", "topics": "LLMs", "depth": "comprehensive"}, "reasoning": "All details present in request"}'

    message = "Research AI developments in 2023 with a focus on LLMs, provide a comprehensive report."

    # Execute
    required = research_agent.identify_required_info(message, message)

    # Verify
    assert len(required) == 0  # No info needed
    mock_service.generate_gemini_response.assert_called_once()


@patch("app.hermes.legion.agents.research_agent.get_gemini_service")
def test_smart_info_needs_clarification(mock_get_service, research_agent):
    """Test that agent asks for clarification when genuinely needed."""
    # Setup
    mock_service = Mock()
    mock_get_service.return_value = mock_service

    # Mock LLM response saying clarification is needed
    mock_service.generate_gemini_response.return_value = """
    {
        "needs_info": true,
        "required_fields": [
            {
                "field_name": "research_subject",
                "field_type": "string",
                "question": "What would you like me to research?",
                "description": "The subject of research"
            }
        ],
        "reasoning": "Request is too vague - 'research it' with no context"
    }
    """

    message = "research it"

    # Execute
    required = research_agent.identify_required_info(message, message)

    # Verify - should ask for clarification
    assert len(required) == 1
    assert "research_subject" in required
    mock_service.generate_gemini_response.assert_called_once()


@patch("app.hermes.legion.agents.research_agent.get_gemini_service")
def test_smart_info_llm_failure(mock_get_service, research_agent):
    """Test that agent fails open (proceeds without asking) on LLM errors."""
    # Setup
    mock_service = Mock()
    mock_get_service.return_value = mock_service

    # Mock LLM failure
    mock_service.generate_gemini_response.side_effect = Exception("LLM Error")

    message = "Research AI developments in 2023 with a focus on LLMs."

    # Execute
    required = research_agent.identify_required_info(message, message)

    # Verify - should fail open (return empty dict, proceed without asking)
    assert len(required) == 0
