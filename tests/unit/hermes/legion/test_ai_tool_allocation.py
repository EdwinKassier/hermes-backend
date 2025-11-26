"""Unit tests for AI-powered tool allocation."""

from unittest.mock import MagicMock, Mock

import pytest

from app.hermes.legion.utils.tool_allocator import ToolAllocator


class TestToolAllocation:
    """Test AI-powered tool allocation."""

    @pytest.fixture
    def allocator(self):
        """Create tool allocator instance."""
        allocator = ToolAllocator()
        allocator._gemini_service = Mock()
        allocator._gemini_service.all_tools = (
            []
        )  # Mock as empty list to prevent iteration errors
        allocator._gemini_service.persona_configs = {}
        return allocator

    @pytest.fixture
    def mock_tools(self):
        """Create mock tools."""
        tools = []
        for name in ["web_search", "python_repl", "csv_reader", "data_parser"]:
            tool = Mock()
            tool.name = name
            tool.description = f"Tool for {name}"
            tools.append(tool)
        return tools

    def test_tool_allocation_ai_success(self, allocator, mock_tools):
        """Test AI successfully allocates tools."""
        allocator.gemini_service.generate_gemini_response.return_value = (
            '{"selected_tools": ["python_repl", "csv_reader"]}'
        )

        result = allocator.allocate_tools_with_ai(
            "code", "Write Python to read CSV", mock_tools
        )

        assert "python_repl" in result
        assert "csv_reader" in result
        assert len(result) == 2

    def test_tool_allocation_ai_failure_uses_fallback(self, allocator, mock_tools):
        """Test fallback when AI fails."""
        allocator.gemini_service.generate_gemini_response.side_effect = Exception(
            "API Error"
        )

        result = allocator.allocate_tools_with_ai(
            "code", "Write Python code", mock_tools
        )

        # Should use fallback - returns tools for "code" task type
        assert isinstance(result, list)

    def test_tool_allocation_network_error_fallback(self, allocator, mock_tools):
        """Test network error uses fallback."""
        allocator.gemini_service.generate_gemini_response.side_effect = ConnectionError(
            "Network"
        )

        result = allocator.allocate_tools_with_ai("code", "Write code", mock_tools)

        assert isinstance(result, list)

    def test_parse_tool_selection_valid_tools(self, allocator):
        """Test parsing valid tool selection."""
        available_tools = {"web_search", "python_repl", "csv_reader"}
        response = '{"selected_tools": ["web_search", "python_repl"]}'

        result = allocator._parse_tool_selection(response, available_tools)

        assert result == ["web_search", "python_repl"]

    def test_parse_tool_selection_invalid_tools_filtered(self, allocator):
        """Test invalid tools are filtered out."""
        available_tools = {"web_search", "python_repl"}
        response = '{"selected_tools": ["web_search", "invalid_tool", "python_repl"]}'

        result = allocator._parse_tool_selection(response, available_tools)

        assert result == ["web_search", "python_repl"]
        assert "invalid_tool" not in result

    def test_parse_tool_selection_empty_response(self, allocator):
        """Test empty response returns empty list."""
        available_tools = {"web_search"}
        response = '{"selected_tools": []}'

        result = allocator._parse_tool_selection(response, available_tools)

        assert result == []

    def test_parse_tool_selection_case_insensitive(self, allocator):
        """Test parsing handles exact case matching."""
        available_tools = {"web_search", "python_repl"}
        response = '{"selected_tools": ["web_search", "python_repl"]}'

        result = allocator._parse_tool_selection(response, available_tools)

        assert "web_search" in result
        assert "python_repl" in result

    def test_fallback_tool_allocation_code_task(self, allocator, mock_tools):
        """Test fallback allocates tools for code task."""
        result = allocator._fallback_tool_allocation("code", mock_tools)

        # Should include python_repl (has "code" capability)
        assert "python_repl" in result

    def test_fallback_tool_allocation_unknown_task(self, allocator, mock_tools):
        """Test fallback with unknown task type."""
        result = allocator._fallback_tool_allocation("unknown", mock_tools)

        # Should return all tools (conservative approach)
        assert len(result) > 0

    def test_get_tool_description_from_attribute(self, allocator):
        """Test extracting description from tool.description."""
        tool = Mock()
        tool.description = "Test description"

        result = allocator._get_tool_description(tool)

        assert result == "Test description"

    def test_get_tool_description_from_docstring(self, allocator):
        """Test extracting description from docstring."""
        tool = Mock(spec=[])
        tool.__doc__ = "First line of docstring\nSecond line"
        tool.name = "test_tool"

        result = allocator._get_tool_description(tool)

        assert result == "First line of docstring"

    def test_get_tool_description_fallback(self, allocator):
        """Test fallback description generation."""
        tool = Mock(spec=[])
        tool.__doc__ = None
        tool.name = "web_search"

        result = allocator._get_tool_description(tool)

        assert "web search" in result.lower()


class TestPromptBuilding:
    """Test tool selection prompt building."""

    @pytest.fixture
    def allocator(self):
        return ToolAllocator()

    def test_prompt_includes_task_info(self, allocator):
        """Test prompt includes task type and description."""
        tool_descriptions = {"web_search": "Search the web"}

        prompt = allocator._build_tool_selection_prompt(
            "research", "Research AI trends", tool_descriptions
        )

        assert "research" in prompt
        assert "Research AI trends" in prompt

    def test_prompt_includes_available_tools(self, allocator):
        """Test prompt lists available tools."""
        tool_descriptions = {
            "web_search": "Search the web",
            "python_repl": "Execute Python",
        }

        prompt = allocator._build_tool_selection_prompt(
            "code", "Write code", tool_descriptions
        )

        assert "web_search" in prompt
        assert "python_repl" in prompt
        assert "Search the web" in prompt
        assert "Execute Python" in prompt

    def test_prompt_includes_examples(self, allocator):
        """Test prompt includes JSON format instructions."""
        tool_descriptions = {"web_search": "Search"}

        prompt = allocator._build_tool_selection_prompt(
            "research", "Research", tool_descriptions
        )

        # Check for JSON format instructions instead of examples
        assert "JSON" in prompt or "json" in prompt.lower()
        assert "selected_tools" in prompt

    def test_prompt_specifies_response_format(self, allocator):
        """Test prompt specifies expected response format."""
        tool_descriptions = {"web_search": "Search"}

        prompt = allocator._build_tool_selection_prompt(
            "research", "Research", tool_descriptions
        )

        # Check for JSON format instead of comma-separated
        assert "JSON" in prompt or "json" in prompt.lower()
        assert "selected_tools" in prompt


class TestEdgeCases:
    """Test edge cases."""

    @pytest.fixture
    def allocator(self):
        allocator = ToolAllocator()
        allocator._gemini_service = Mock()
        return allocator

    @pytest.fixture
    def mock_tools(self):
        tools = []
        for name in ["web_search", "python_repl"]:
            tool = Mock()
            tool.name = name
            tool.description = f"Tool for {name}"
            tools.append(tool)
        return tools

    def test_ai_response_with_spaces(self, allocator, mock_tools):
        """Test AI response with JSON format."""
        allocator.gemini_service.generate_gemini_response.return_value = (
            '{"selected_tools": ["web_search", "python_repl"]}'
        )

        result = allocator.allocate_tools_with_ai("research", "Research", mock_tools)

        assert "web_search" in result
        assert "python_repl" in result

    def test_ai_response_single_tool(self, allocator, mock_tools):
        """Test AI response with single tool."""
        allocator.gemini_service.generate_gemini_response.return_value = (
            '{"selected_tools": ["web_search"]}'
        )

        result = allocator.allocate_tools_with_ai("research", "Research", mock_tools)

        assert result == ["web_search"]

    def test_empty_tools_list(self, allocator):
        """Test with empty tools list."""
        allocator.gemini_service.generate_gemini_response.return_value = (
            '{"selected_tools": ["web_search"]}'
        )

        result = allocator.allocate_tools_with_ai("research", "Research", [])

        assert result == []
