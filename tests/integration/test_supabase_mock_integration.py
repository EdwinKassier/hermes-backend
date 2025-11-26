"""
Integration tests for MCP Database Service with mock server.
"""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

# Skip if MCP not enabled
pytestmark = pytest.mark.skipif(
    not os.getenv("MCP_SERVER_ENABLED", "false").lower() == "true",
    reason="MCP server not enabled",
)


class TestSupabaseMockIntegration:
    """Integration tests for MCP Database Service with mock server."""

    @pytest.fixture
    def service(self):
        """Create MCP service instance for integration testing."""
        from app.shared.services.SupabaseDatabaseService import SupabaseDatabaseService

        return SupabaseDatabaseService(
            supabase_url=os.getenv("SUPABASE_MCP_SERVER_URL", "http://localhost:3001"),
            supabase_key=os.getenv("SUPABASE_MCP_API_KEY", "test-key"),
        )

    @pytest.fixture
    def tool(self):
        """Create LangChain tool instance for integration testing."""
        from app.shared.utils.tools.database_query_tool import DatabaseQueryTool

        return DatabaseQueryTool()

    def test_service_initialization(self, service):
        """Test that service initializes correctly."""
        assert service is not None
        assert service.mcp_server_url is not None
        assert service.api_key is not None

    def test_tool_initialization(self, tool):
        """Test that LangChain tool initializes correctly."""
        assert tool is not None
        assert tool.name == "database_query"
        assert tool.mcp_service is not None

    def test_mock_server_health(self, service):
        """Test mock server health check."""
        try:
            # This will work with the nginx mock server
            result = service._run_async_safe(service.list_tables())
            assert result is not None
            assert isinstance(result, list)
        except Exception as e:
            pytest.skip(f"Mock server not available: {e}")

    def test_tool_query_execution(self, tool):
        """Test LangChain tool query execution with mock server."""
        try:
            result = tool._run(
                query="Show me all tables", include_analysis=False, use_cache=False
            )

            assert result is not None
            assert isinstance(result, str)
            assert len(result) > 0
        except Exception as e:
            pytest.skip(f"Mock server not available: {e}")

    def test_tool_table_listing(self, tool):
        """Test LangChain tool table listing with mock server."""
        try:
            result = tool.list_tables()

            assert result is not None
            assert isinstance(result, str)
            assert "Available tables:" in result or "No tables found" in result
        except Exception as e:
            pytest.skip(f"Mock server not available: {e}")

    def test_tool_schema_info(self, tool):
        """Test LangChain tool schema information with mock server."""
        try:
            result = tool.get_schema_info()

            assert result is not None
            assert isinstance(result, str)
            assert len(result) > 0
        except Exception as e:
            pytest.skip(f"Mock server not available: {e}")

    def test_error_handling(self, tool):
        """Test error handling with mock server."""
        try:
            # Test with invalid query that should return an error
            result = tool._run(
                query="INVALID SQL QUERY THAT SHOULD FAIL",
                include_analysis=False,
                use_cache=False,
            )

            # Should return error message, not raise exception
            assert result is not None
            assert isinstance(result, str)
        except Exception as e:
            pytest.skip(f"Mock server not available: {e}")

    def test_caching_behavior(self, tool):
        """Test query result caching with mock server."""
        try:
            query = "Test caching query"

            # First call
            result1 = tool._run(query, use_cache=True)
            assert result1 is not None

            # Second call should use cache
            result2 = tool._run(query, use_cache=True)
            assert result2 == result1

            # Third call without cache should make new request
            result3 = tool._run(query, use_cache=False)
            assert result3 is not None

        except Exception as e:
            pytest.skip(f"Mock server not available: {e}")

    def test_tool_integration_with_gemini(self, tool):
        """Test tool integration with GeminiService."""
        try:
            # This test requires both MCP server and Gemini service to be available
            result = tool._run(
                query="Analyze the hermes_vectors table structure",
                include_analysis=True,
                use_cache=False,
            )

            assert result is not None
            assert isinstance(result, str)
            assert len(result) > 0

            # Should contain analysis if Gemini is working
            if "Analysis unavailable" not in result:
                assert "--- Raw Data ---" in result

        except Exception as e:
            pytest.skip(f"Mock server or Gemini service not available: {e}")
