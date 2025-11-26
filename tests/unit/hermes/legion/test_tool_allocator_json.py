from unittest.mock import Mock, patch

import pytest

from app.hermes.legion.utils.tool_allocator import ToolAllocator


@pytest.fixture
def mock_tools():
    tool1 = Mock()
    tool1.name = "web_search"
    tool1.description = "Search the web"

    tool2 = Mock()
    tool2.name = "calculator"
    tool2.description = "Calculate numbers"

    return [tool1, tool2]


@pytest.fixture
def allocator():
    return ToolAllocator()


@patch("app.hermes.legion.utils.tool_allocator.get_gemini_service")
def test_json_allocation_success(mock_get_service, allocator, mock_tools):
    # Setup
    mock_service = Mock()
    mock_get_service.return_value = mock_service
    allocator._gemini_service = mock_service

    # Mock JSON response
    mock_service.generate_gemini_response.return_value = (
        '{"selected_tools": ["web_search"]}'
    )

    # Execute
    result = allocator.allocate_tools_with_ai("research", "Find info", mock_tools)

    # Verify
    assert result == ["web_search"]
    mock_service.generate_gemini_response.assert_called_once()


@patch("app.hermes.legion.utils.tool_allocator.get_gemini_service")
def test_json_allocation_invalid_json(mock_get_service, allocator, mock_tools):
    # Setup
    mock_service = Mock()
    mock_get_service.return_value = mock_service
    allocator._gemini_service = mock_service

    # Mock invalid JSON response
    mock_service.generate_gemini_response.return_value = (
        "I think you should use web_search"
    )

    # Execute (should fallback to empty list from parsing, then fallback allocation)
    # Note: The fallback allocation logic in the code uses static mapping or conservative approach
    # Let's mock fallback to return specific tools to verify it was called
    with patch.object(
        allocator, "_fallback_tool_allocation", return_value=["calculator"]
    ) as mock_fallback:
        result = allocator.allocate_tools_with_ai("research", "Find info", mock_tools)

        # Verify
        assert result == ["calculator"]
        mock_fallback.assert_called_once()


@patch("app.hermes.legion.utils.tool_allocator.get_gemini_service")
def test_json_allocation_non_existent_tool(mock_get_service, allocator, mock_tools):
    # Setup
    mock_service = Mock()
    mock_get_service.return_value = mock_service
    allocator._gemini_service = mock_service

    # Mock JSON with one valid and one invalid tool
    mock_service.generate_gemini_response.return_value = (
        '{"selected_tools": ["web_search", "magic_wand"]}'
    )

    # Execute
    result = allocator.allocate_tools_with_ai("research", "Find info", mock_tools)

    # Verify
    assert result == ["web_search"]  # magic_wand should be filtered out


@patch("app.hermes.legion.utils.tool_allocator.get_gemini_service")
def test_json_allocation_string_format(mock_get_service, allocator, mock_tools):
    # Setup
    mock_service = Mock()
    mock_get_service.return_value = mock_service
    allocator._gemini_service = mock_service

    # Mock JSON where selected_tools is a string (invalid format - should trigger fallback)
    mock_service.generate_gemini_response.return_value = (
        '{"selected_tools": "web_search, calculator"}'
    )

    # Execute - should use fallback allocation since format is invalid
    with patch.object(
        allocator,
        "_fallback_tool_allocation",
        return_value=["web_search", "calculator"],
    ) as mock_fallback:
        result = allocator.allocate_tools_with_ai("research", "Find info", mock_tools)

        # Verify fallback was called due to invalid format
        mock_fallback.assert_called_once()
        assert result == ["web_search", "calculator"]
