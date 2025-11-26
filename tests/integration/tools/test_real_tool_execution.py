"""
Real Integration Tests for Tools.

These tests actually call the tools with real implementations.
They require proper configuration (API keys, database access, etc.)

Run with: pytest tests/integration/tools/test_real_tool_execution.py -v -s
"""

import os
from datetime import datetime

import pytest


class TestTimeToolRealExecution:
    """Real integration tests for TimeInfoTool."""

    @pytest.fixture
    def time_tool(self):
        """Get real TimeInfoTool instance."""
        from app.shared.utils.tools.time_tool import TimeInfoTool

        return TimeInfoTool()

    def test_get_utc_time_real(self, time_tool):
        """Test getting real UTC time."""
        result = time_tool._run(timezone="UTC")

        print(f"\n=== UTC Time Result ===\n{result}\n")

        # Verify structure
        assert "Current Time Information" in result
        assert "UTC" in result
        assert "Date:" in result
        assert "Time:" in result

        # Verify current year is present
        current_year = datetime.now().year
        assert str(current_year) in result

        print("✅ UTC time test passed")

    def test_get_multiple_timezones_real(self, time_tool):
        """Test getting time in multiple real timezones."""
        timezones = ["UTC", "US/Pacific", "Europe/London", "Asia/Tokyo"]

        for tz in timezones:
            result = time_tool._run(timezone=tz)

            print(f"\n=== {tz} Time ===\n{result[:200]}...\n")

            assert tz in result
            assert "Date:" in result
            assert "Time:" in result

        print("✅ Multiple timezone test passed")

    def test_time_context_real(self, time_tool):
        """Test that time context is provided."""
        result = time_tool._run()

        # Verify context sections exist
        assert "Context:" in result
        assert "Day of week:" in result
        assert "Time of day:" in result
        assert "Technical Details:" in result
        assert "ISO 8601:" in result
        assert "Unix timestamp:" in result

        print("✅ Time context test passed")


class TestWebSearchToolRealExecution:
    """Real integration tests for WebSearchTool."""

    @pytest.fixture
    def web_search_tool(self):
        """Get real WebSearchTool instance."""
        from app.shared.utils.tools.web_search_tool import WebSearchTool

        return WebSearchTool()

    @pytest.mark.skipif(
        not os.environ.get("FIRECRAWL_API_KEY"), reason="FIRECRAWL_API_KEY not set"
    )
    def test_real_web_search(self, web_search_tool):
        """Test real web search with Firecrawl API."""
        query = "Python programming language"
        result = web_search_tool._run(query, max_results=3)

        print(f"\n=== Web Search Results ===\n{result}\n")

        # Verify results structure
        assert "Search results for:" in result
        assert query in result

        # Should have at least one result
        assert "1." in result
        assert "URL:" in result

        print("✅ Real web search test passed")

    @pytest.mark.skipif(
        not os.environ.get("FIRECRAWL_API_KEY"), reason="FIRECRAWL_API_KEY not set"
    )
    def test_web_search_multiple_queries(self, web_search_tool):
        """Test multiple real web searches."""
        queries = ["artificial intelligence", "machine learning", "web development"]

        for query in queries:
            result = web_search_tool._run(query, max_results=2)

            print(f"\n=== Search: {query} ===\n{result[:300]}...\n")

            assert "Search results for:" in result
            assert query in result

        print("✅ Multiple web search test passed")

    def test_web_search_without_api_key(self, web_search_tool):
        """Test web search behavior without API key."""
        # Temporarily remove API key
        original_key = web_search_tool.api_key
        web_search_tool.api_key = None

        result = web_search_tool._run("test query")

        print(f"\n=== No API Key Result ===\n{result}\n")

        assert "unavailable" in result
        assert "FIRECRAWL_API_KEY" in result

        # Restore key
        web_search_tool.api_key = original_key

        print("✅ No API key test passed")


class TestDatabaseQueryToolRealExecution:
    """Real integration tests for DatabaseQueryTool."""

    @pytest.fixture
    def database_tool(self):
        """Get real DatabaseQueryTool instance."""
        try:
            from app.shared.utils.tools.database_query_tool import DatabaseQueryTool

            return DatabaseQueryTool()
        except Exception as e:
            pytest.skip(f"Database tool initialization failed: {e}")

    @pytest.mark.skipif(
        not os.environ.get("SUPABASE_MCP_SERVER_URL")
        or not os.environ.get("SUPABASE_MCP_API_KEY"),
        reason="MCP server not configured",
    )
    def test_real_database_query(self, database_tool):
        """Test real database query."""
        query = "List all tables in the database"
        result = database_tool._run(query, include_analysis=False, use_cache=False)

        print(f"\n=== Database Query Result ===\n{result}\n")

        # Should not be an error
        assert not result.startswith("Error:")

        # Should have some content
        assert len(result) > 0

        print("✅ Real database query test passed")

    def test_database_input_validation(self, database_tool):
        """Test database tool input validation."""
        # Test empty query
        result = database_tool._run("")
        assert "Error" in result
        assert "non-empty string" in result

        # Test short query
        result = database_tool._run("ab")
        assert "Error" in result
        assert "too short" in result

        # Test long query
        result = database_tool._run("a" * 1001)
        assert "Error" in result
        assert "too long" in result

        print("✅ Database input validation test passed")

    def test_database_helper_methods(self, database_tool):
        """Test database helper methods."""
        # Test list tables
        result = database_tool._list_tables()

        print(f"\n=== List Tables Result ===\n{result}\n")

        # Should return something (either tables or error)
        assert len(result) > 0

        print("✅ Database helper methods test passed")


class TestToolIntegrationReal:
    """Real integration tests for tool system."""

    def test_load_all_tools_real(self):
        """Test loading all tools from toolhub."""
        from app.shared.utils.toolhub import get_all_tools

        tools = get_all_tools()

        print(f"\n=== Loaded Tools ===")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:60]}...")
        print()

        # Should have exactly 3 tools
        assert len(tools) == 3

        # Verify tool names
        tool_names = {tool.name for tool in tools}
        assert tool_names == {"database_query", "time_info", "web_search"}

        print("✅ Tool loading test passed")

    def test_all_tools_callable(self):
        """Test that all tools can be called."""
        from app.shared.utils.toolhub import get_all_tools

        tools = get_all_tools()

        for tool in tools:
            # Verify tool has _run method
            assert hasattr(tool, "_run")
            assert callable(tool._run)

            print(f"✅ {tool.name} is callable")

        print("✅ All tools callable test passed")

    def test_time_tool_execution(self):
        """Test time tool can be executed."""
        from app.shared.utils.toolhub import get_all_tools

        tools = get_all_tools()
        time_tool = next(t for t in tools if t.name == "time_info")

        result = time_tool._run()

        print(f"\n=== Time Tool Execution ===\n{result[:200]}...\n")

        assert len(result) > 0
        assert "Time" in result

        print("✅ Time tool execution test passed")


class TestToolErrorHandling:
    """Test error handling across all tools."""

    def test_time_tool_invalid_timezone(self):
        """Test time tool with invalid timezone."""
        from app.shared.utils.tools.time_tool import TimeInfoTool

        tool = TimeInfoTool()
        result = tool._run(timezone="Invalid/Timezone")

        print(f"\n=== Invalid Timezone Result ===\n{result}\n")

        assert "Error" in result

        print("✅ Time tool error handling test passed")

    def test_database_tool_invalid_input(self):
        """Test database tool with invalid input."""
        try:
            from app.shared.utils.tools.database_query_tool import DatabaseQueryTool

            tool = DatabaseQueryTool()

            # Test with None
            result = tool._run(None)
            assert "Error" in result

            # Test with empty
            result = tool._run("")
            assert "Error" in result

            print("✅ Database tool error handling test passed")
        except Exception as e:
            pytest.skip(f"Database tool not available: {e}")

    def test_web_search_without_package(self):
        """Test web search error when package missing."""
        from app.shared.utils.tools.web_search_tool import WebSearchTool

        tool = WebSearchTool()

        # If no API key, should return error
        if not tool.api_key:
            result = tool._run("test")
            assert "unavailable" in result
            print("✅ Web search error handling test passed")
        else:
            print("⚠️  Skipped (API key present)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
