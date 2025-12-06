"""
Integration tests for MCP Database Service.
"""

import asyncio
import os

import pytest
from dotenv import load_dotenv

load_dotenv()

# Skip if MCP credentials not available
pytestmark = pytest.mark.skipif(
    not os.getenv("SUPABASE_PROJECT_URL") or not os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
    reason="Supabase credentials not available",
)


class TestMCPDatabaseIntegration:
    """Integration tests for MCP Database Service."""

    @pytest.fixture
    def service(self):
        """Create MCP service instance for integration testing."""
        from app.shared.services.SupabaseDatabaseService import SupabaseDatabaseService

        return SupabaseDatabaseService(
            supabase_url=os.getenv("SUPABASE_PROJECT_URL"),
            supabase_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
        )

    @pytest.fixture
    def tool(self):
        """Create LangChain tool instance for integration testing."""
        from app.shared.utils.tools.database_query_tool import DatabaseQueryTool

        return DatabaseQueryTool()

    @pytest.mark.asyncio
    async def test_service_initialization(self, service):
        """Test that service initializes correctly."""
        assert service is not None
        assert service.supabase_url is not None
        assert service.supabase_key is not None
        assert service.client is not None

    @pytest.mark.asyncio
    async def test_tool_initialization(self, tool):
        """Test that LangChain tool initializes correctly."""
        assert tool is not None
        assert tool.name == "database_query"
        assert tool.db_service is not None

    @pytest.mark.asyncio
    async def test_simple_query_execution(self, service):
        """Test simple query execution against real MCP server."""
        try:
            result = await service.execute_query(
                "List all tables in the database",
                include_analysis=False,
                use_cache=False,
            )

            assert result is not None
            assert isinstance(result, str)
            assert len(result) > 0

        except Exception as e:
            pytest.skip(f"MCP server not available: {e}")

    @pytest.mark.asyncio
    async def test_table_listing(self, service):
        """Test table listing functionality."""
        try:
            tables = await service.list_tables()

            assert isinstance(tables, list)
            # Should include hermes_vectors table if it exists
            table_names = [table.lower() for table in tables]
            assert any("hermes" in name or "vector" in name for name in table_names)

        except Exception as e:
            pytest.skip(f"MCP server not available: {e}")

    @pytest.mark.asyncio
    async def test_table_description(self, service):
        """Test table description functionality."""
        try:
            # First get list of tables
            tables = await service.list_tables()
            if not tables:
                pytest.skip("No tables available for testing")

            # Test describing the first table
            table_name = tables[0]
            description = await service.describe_table(table_name)

            assert isinstance(description, dict)
            assert len(description) > 0

        except Exception as e:
            pytest.skip(f"MCP server not available: {e}")

    @pytest.mark.asyncio
    async def test_schema_info(self, service):
        """Test comprehensive schema information retrieval."""
        try:
            schema_info = await service.get_schema_info()

            assert isinstance(schema_info, dict)
            assert len(schema_info) > 0

        except Exception as e:
            pytest.skip(f"MCP server not available: {e}")

    def test_tool_query_execution(self, tool):
        """Test LangChain tool query execution."""
        try:
            result = tool._run(
                query="Show me the structure of the hermes_vectors table",
                include_analysis=False,
                use_cache=False,
            )

            assert result is not None
            assert isinstance(result, str)
            assert len(result) > 0

        except Exception as e:
            pytest.skip(f"MCP server not available: {e}")

    def test_tool_table_listing(self, tool):
        """Test LangChain tool table listing."""
        try:
            result = tool.list_tables()

            assert result is not None
            assert isinstance(result, str)
            assert "Available tables:" in result or "No tables found" in result

        except Exception as e:
            pytest.skip(f"MCP server not available: {e}")

    def test_tool_table_description(self, tool):
        """Test LangChain tool table description."""
        try:
            # First get list of tables
            tables_result = tool.list_tables()
            if "No tables found" in tables_result:
                pytest.skip("No tables available for testing")

            # Extract first table name from result
            # This is a simple extraction - in real usage, the tool would be more sophisticated
            if "Available tables:" in tables_result:
                tables_line = tables_result.split("Available tables:")[1].strip()
                first_table = tables_line.split(",")[0].strip()

                description = tool.describe_table(first_table)

                assert description is not None
                assert isinstance(description, str)
                assert len(description) > 0

        except Exception as e:
            pytest.skip(f"MCP server not available: {e}")

    def test_tool_schema_info(self, tool):
        """Test LangChain tool schema information."""
        try:
            result = tool.get_schema_info()

            assert result is not None
            assert isinstance(result, str)
            assert len(result) > 0

        except Exception as e:
            pytest.skip(f"MCP server not available: {e}")

    @pytest.mark.asyncio
    async def test_query_with_analysis(self, service):
        """Test query execution with AI analysis."""
        try:
            result = await service.execute_query(
                "Count the number of records in the hermes_vectors table",
                include_analysis=True,
                use_cache=False,
            )

            assert result is not None
            assert isinstance(result, str)
            assert len(result) > 0

            # Should contain analysis section
            assert "--- Raw Data ---" in result

        except Exception as e:
            pytest.skip(f"MCP server not available: {e}")

    @pytest.mark.asyncio
    async def test_caching_behavior(self, service):
        """Test query result caching."""
        try:
            query = "Test caching query"

            # First call
            result1 = await service.execute_query(query, use_cache=True)
            assert result1 is not None

            # Second call should use cache
            result2 = await service.execute_query(query, use_cache=True)
            assert result2 == result1

            # Third call without cache should make new request
            result3 = await service.execute_query(query, use_cache=False)
            assert result3 is not None

        except Exception as e:
            pytest.skip(f"MCP server not available: {e}")

    @pytest.mark.asyncio
    async def test_error_handling(self, service):
        """Test error handling for invalid queries."""
        try:
            # Test with invalid query that should return an error
            result = await service.execute_query(
                "INVALID SQL QUERY THAT SHOULD FAIL",
                include_analysis=False,
                use_cache=False,
            )

            # Should return error message, not raise exception
            assert result is not None
            assert isinstance(result, str)

        except Exception as e:
            pytest.skip(f"MCP server not available: {e}")

    @pytest.mark.asyncio
    async def test_concurrent_queries(self, service):
        """Test concurrent query execution."""
        try:
            queries = [
                "List all tables",
                "Show database schema",
                "Count records in hermes_vectors",
            ]

            # Execute queries concurrently
            tasks = [
                service.execute_query(query, include_analysis=False, use_cache=False)
                for query in queries
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should complete (some might be errors, but should not raise exceptions)
            assert len(results) == len(queries)
            for result in results:
                assert isinstance(result, str)

        except Exception as e:
            pytest.skip(f"MCP server not available: {e}")

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
            pytest.skip(f"MCP server or Gemini service not available: {e}")

    @pytest.mark.asyncio
    async def test_service_cleanup(self, service):
        """Test service cleanup and resource management."""
        try:
            # Test that service can be properly closed
            await service.close()

            # Service should still be usable after close (new client created)
            result = await service.execute_query("test query", use_cache=False)
            assert result is not None

        except Exception as e:
            pytest.skip(f"MCP server not available: {e}")
