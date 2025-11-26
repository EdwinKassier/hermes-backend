"""
Integration tests for all tools.

These tests verify that tools work together and can be loaded by the toolhub.
"""

import pytest

from app.shared.utils.toolhub import get_all_tools


class TestToolIntegration:
    """Integration tests for tool system."""

    def test_get_all_tools_returns_list(self):
        """Test that get_all_tools returns a list."""
        tools = get_all_tools()
        assert isinstance(tools, list)

    def test_get_all_tools_count(self):
        """Test that we have exactly 3 tools."""
        tools = get_all_tools()
        assert len(tools) == 3, f"Expected 3 tools, got {len(tools)}"

    def test_all_tools_have_names(self):
        """Test that all tools have names."""
        tools = get_all_tools()

        for tool in tools:
            assert hasattr(tool, "name")
            assert isinstance(tool.name, str)
            assert len(tool.name) > 0

    def test_tool_names_are_unique(self):
        """Test that all tool names are unique."""
        tools = get_all_tools()
        names = [tool.name for tool in tools]

        assert len(names) == len(set(names)), "Tool names are not unique"

    def test_expected_tools_present(self):
        """Test that expected tools are present."""
        tools = get_all_tools()
        tool_names = {tool.name for tool in tools}

        expected_tools = {"database_query", "time_info", "web_search"}
        assert (
            tool_names == expected_tools
        ), f"Expected {expected_tools}, got {tool_names}"

    def test_all_tools_have_descriptions(self):
        """Test that all tools have descriptions."""
        tools = get_all_tools()

        for tool in tools:
            assert hasattr(tool, "description")
            assert isinstance(tool.description, str)
            assert len(tool.description) > 0

    def test_all_tools_have_run_method(self):
        """Test that all tools have _run method."""
        tools = get_all_tools()

        for tool in tools:
            assert hasattr(tool, "_run")
            assert callable(tool._run)

    def test_all_tools_have_args_schema(self):
        """Test that all tools have args_schema."""
        tools = get_all_tools()

        for tool in tools:
            assert hasattr(tool, "args_schema")

    def test_database_query_tool_loaded(self):
        """Test that database_query tool is loaded correctly."""
        tools = get_all_tools()
        db_tool = next((t for t in tools if t.name == "database_query"), None)

        assert db_tool is not None
        assert "database" in db_tool.description.lower()

    def test_time_info_tool_loaded(self):
        """Test that time_info tool is loaded correctly."""
        tools = get_all_tools()
        time_tool = next((t for t in tools if t.name == "time_info"), None)

        assert time_tool is not None
        assert "time" in time_tool.description.lower()

    def test_web_search_tool_loaded(self):
        """Test that web_search tool is loaded correctly."""
        tools = get_all_tools()
        search_tool = next((t for t in tools if t.name == "web_search"), None)

        assert search_tool is not None
        assert (
            "search" in search_tool.description.lower()
            or "internet" in search_tool.description.lower()
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
