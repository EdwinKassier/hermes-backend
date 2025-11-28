"""Tests for ToolRegistry."""

from unittest.mock import MagicMock, patch

import pytest

from app.hermes.legion.utils.tool_registry import ToolRegistry, get_tool_registry


class MockTool:
    """Mock tool for testing."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def setup_method(self):
        """Reset singleton before each test."""
        ToolRegistry.reset()

    def teardown_method(self):
        """Reset singleton after each test."""
        ToolRegistry.reset()

    def test_singleton_pattern(self):
        """get_instance returns same instance."""
        with patch("app.shared.utils.service_loader.get_gemini_service") as mock_gemini:
            mock_gemini.return_value = MagicMock(all_tools=[])

            instance1 = ToolRegistry.get_instance()
            instance2 = ToolRegistry.get_instance()

            assert instance1 is instance2

    def test_reset_clears_singleton(self):
        """reset() allows new instance creation."""
        with patch("app.shared.utils.service_loader.get_gemini_service") as mock_gemini:
            mock_gemini.return_value = MagicMock(all_tools=[])

            instance1 = ToolRegistry.get_instance()
            ToolRegistry.reset()
            instance2 = ToolRegistry.get_instance()

            assert instance1 is not instance2

    def test_initializes_tools_from_gemini(self):
        """Registry loads tools from Gemini service."""
        mock_tools = [
            MockTool("web_search"),
            MockTool("database_query"),
            MockTool("time_info"),
        ]

        with patch("app.shared.utils.service_loader.get_gemini_service") as mock_gemini:
            mock_gemini.return_value = MagicMock(all_tools=mock_tools)

            registry = ToolRegistry.get_instance()

            assert registry.has_tool("web_search")
            assert registry.has_tool("database_query")
            assert registry.has_tool("time_info")
            assert not registry.has_tool("nonexistent")

    def test_get_tool_by_name(self):
        """get_tool returns correct tool by name."""
        mock_tool = MockTool("test_tool", "A test tool")

        with patch("app.shared.utils.service_loader.get_gemini_service") as mock_gemini:
            mock_gemini.return_value = MagicMock(all_tools=[mock_tool])

            registry = ToolRegistry.get_instance()
            result = registry.get_tool("test_tool")

            assert result is mock_tool

    def test_get_tool_returns_none_for_missing(self):
        """get_tool returns None for non-existent tool."""
        with patch("app.shared.utils.service_loader.get_gemini_service") as mock_gemini:
            mock_gemini.return_value = MagicMock(all_tools=[])

            registry = ToolRegistry.get_instance()
            result = registry.get_tool("nonexistent")

            assert result is None

    def test_get_tools_multiple(self):
        """get_tools returns list of matching tools."""
        mock_tools = [
            MockTool("tool_a"),
            MockTool("tool_b"),
            MockTool("tool_c"),
        ]

        with patch("app.shared.utils.service_loader.get_gemini_service") as mock_gemini:
            mock_gemini.return_value = MagicMock(all_tools=mock_tools)

            registry = ToolRegistry.get_instance()
            result = registry.get_tools(["tool_a", "tool_c"])

            assert len(result) == 2
            assert result[0].name == "tool_a"
            assert result[1].name == "tool_c"

    def test_get_tools_skips_missing(self):
        """get_tools skips non-existent tools."""
        mock_tools = [MockTool("tool_a")]

        with patch("app.shared.utils.service_loader.get_gemini_service") as mock_gemini:
            mock_gemini.return_value = MagicMock(all_tools=mock_tools)

            registry = ToolRegistry.get_instance()
            result = registry.get_tools(["tool_a", "nonexistent"])

            assert len(result) == 1
            assert result[0].name == "tool_a"

    def test_get_all_tools(self):
        """get_all_tools returns all registered tools."""
        mock_tools = [
            MockTool("tool_a"),
            MockTool("tool_b"),
        ]

        with patch("app.shared.utils.service_loader.get_gemini_service") as mock_gemini:
            mock_gemini.return_value = MagicMock(all_tools=mock_tools)

            registry = ToolRegistry.get_instance()
            result = registry.get_all_tools()

            assert len(result) == 2

    def test_get_tool_names(self):
        """get_tool_names returns all tool names."""
        mock_tools = [
            MockTool("tool_a"),
            MockTool("tool_b"),
        ]

        with patch("app.shared.utils.service_loader.get_gemini_service") as mock_gemini:
            mock_gemini.return_value = MagicMock(all_tools=mock_tools)

            registry = ToolRegistry.get_instance()
            names = registry.get_tool_names()

            assert "tool_a" in names
            assert "tool_b" in names

    def test_register_tool_manually(self):
        """register_tool adds new tool to registry."""
        with patch("app.shared.utils.service_loader.get_gemini_service") as mock_gemini:
            mock_gemini.return_value = MagicMock(all_tools=[])

            registry = ToolRegistry.get_instance()
            new_tool = MockTool("custom_tool")
            registry.register_tool(new_tool)

            assert registry.has_tool("custom_tool")
            assert registry.get_tool("custom_tool") is new_tool

    def test_unregister_tool(self):
        """unregister_tool removes tool from registry."""
        mock_tool = MockTool("removable_tool")

        with patch("app.shared.utils.service_loader.get_gemini_service") as mock_gemini:
            mock_gemini.return_value = MagicMock(all_tools=[mock_tool])

            registry = ToolRegistry.get_instance()
            assert registry.has_tool("removable_tool")

            result = registry.unregister_tool("removable_tool")
            assert result is True
            assert not registry.has_tool("removable_tool")

    def test_unregister_nonexistent_returns_false(self):
        """unregister_tool returns False for non-existent tool."""
        with patch("app.shared.utils.service_loader.get_gemini_service") as mock_gemini:
            mock_gemini.return_value = MagicMock(all_tools=[])

            registry = ToolRegistry.get_instance()
            result = registry.unregister_tool("nonexistent")

            assert result is False

    def test_convenience_function(self):
        """get_tool_registry convenience function works."""
        with patch("app.shared.utils.service_loader.get_gemini_service") as mock_gemini:
            mock_gemini.return_value = MagicMock(all_tools=[])

            registry = get_tool_registry()

            assert isinstance(registry, ToolRegistry)

    def test_handles_initialization_error_gracefully(self):
        """Registry handles Gemini service errors gracefully."""
        with patch("app.shared.utils.service_loader.get_gemini_service") as mock_gemini:
            mock_gemini.side_effect = Exception("Service unavailable")

            # Should not raise, just log error
            registry = ToolRegistry.get_instance()

            # Registry exists but is empty
            assert registry.get_tool_names() == []
