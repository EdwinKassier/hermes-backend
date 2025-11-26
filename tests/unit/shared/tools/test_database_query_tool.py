"""
Tests for Database Query Tool.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.shared.services.SupabaseDatabaseService import (
    SupabaseAuthenticationError,
    SupabaseConnectionError,
    SupabaseTimeoutError,
    SupabaseValidationError,
)
from app.shared.utils.tools.database_query_tool import DatabaseQueryTool


class TestDatabaseQueryTool:
    """Test suite for DatabaseQueryTool."""

    @pytest.fixture
    def mock_mcp_service(self):
        """Create mock MCP service."""
        mock_service = Mock()
        mock_service.execute_query = AsyncMock()
        return mock_service

    @pytest.fixture
    def tool(self, mock_mcp_service):
        """Create DatabaseQueryTool with mocked service."""
        with patch(
            "app.shared.utils.tools.database_query_tool.get_supabase_database_service",
            return_value=mock_mcp_service,
        ):
            return DatabaseQueryTool()

    # === Input Validation Tests ===

    def test_empty_query(self, tool):
        """Test that empty query returns error."""
        result = tool._run("")
        assert "Error" in result
        assert "non-empty string" in result

    def test_none_query(self, tool):
        """Test that None query returns error."""
        result = tool._run(None)
        assert "Error" in result
        assert "non-empty string" in result

    def test_whitespace_only_query(self, tool):
        """Test that whitespace-only query returns error."""
        result = tool._run("   ")
        assert "Error" in result
        assert "too short" in result

    def test_short_query(self, tool):
        """Test that very short query returns error."""
        result = tool._run("ab")
        assert "Error" in result
        assert "too short" in result

    def test_long_query(self, tool):
        """Test that very long query returns error."""
        long_query = "a" * 1001
        result = tool._run(long_query)
        assert "Error" in result
        assert "too long" in result

    def test_non_string_query(self, tool):
        """Test that non-string query returns error."""
        result = tool._run(123)
        assert "Error" in result
        assert "non-empty string" in result

    # === Successful Query Tests ===

    def test_valid_query_success(self, tool, mock_mcp_service):
        """Test successful query execution."""
        mock_mcp_service.execute_query.return_value = "Query results: 10 users found"

        result = tool._run("Show me all users")

        assert "Query results" in result
        assert "10 users found" in result
        mock_mcp_service.execute_query.assert_called_once()

    def test_query_with_analysis(self, tool, mock_mcp_service):
        """Test query with analysis enabled."""
        mock_mcp_service.execute_query.return_value = "Analysis: Data shows growth"

        result = tool._run("Analyze user growth", include_analysis=True)

        assert "Analysis" in result
        mock_mcp_service.execute_query.assert_called_once_with(
            query="Analyze user growth", include_analysis=True, use_cache=True
        )

    def test_query_without_cache(self, tool, mock_mcp_service):
        """Test query with caching disabled."""
        mock_mcp_service.execute_query.return_value = "Fresh results"

        result = tool._run("Get latest data", use_cache=False)

        mock_mcp_service.execute_query.assert_called_once_with(
            query="Get latest data", include_analysis=True, use_cache=False
        )

    # === Error Handling Tests ===

    def test_connection_error(self, tool, mock_mcp_service):
        """Test handling of connection errors."""
        mock_mcp_service.execute_query.side_effect = SupabaseConnectionError(
            "Connection failed"
        )

        result = tool._run("Valid query")

        assert "temporarily unavailable" in result
        assert "Connection failed" in result

    def test_timeout_error(self, tool, mock_mcp_service):
        """Test handling of timeout errors."""
        mock_mcp_service.execute_query.side_effect = SupabaseTimeoutError(
            "Request timed out"
        )

        result = tool._run("Valid query")

        assert "temporarily unavailable" in result
        assert "timed out" in result

    def test_validation_error(self, tool, mock_mcp_service):
        """Test handling of validation errors."""
        mock_mcp_service.execute_query.side_effect = SupabaseValidationError(
            "Invalid query format"
        )

        result = tool._run("Valid query")

        assert "access error" in result
        assert "Invalid query format" in result

    def test_authentication_error(self, tool, mock_mcp_service):
        """Test handling of authentication errors."""
        mock_mcp_service.execute_query.side_effect = SupabaseAuthenticationError(
            "Auth failed"
        )

        result = tool._run("Valid query")

        assert "access error" in result
        assert "Auth failed" in result

    def test_unexpected_error(self, tool, mock_mcp_service):
        """Test handling of unexpected errors."""
        mock_mcp_service.execute_query.side_effect = RuntimeError("Unexpected error")

        result = tool._run("Valid query")

        assert "Database error" in result
        assert "Unexpected error" in result

    # === Helper Methods Tests ===

    def test_list_tables(self, tool, mock_mcp_service):
        """Test listing tables."""
        mock_mcp_service.list_tables = AsyncMock(
            return_value=["users", "sessions", "conversations"]
        )

        result = tool._list_tables()

        assert "users" in result
        assert "sessions" in result
        assert "conversations" in result

    def test_list_tables_empty(self, tool, mock_mcp_service):
        """Test listing tables when none exist."""
        mock_mcp_service.list_tables = AsyncMock(return_value=[])

        result = tool._list_tables()

        assert "No tables found" in result

    def test_describe_table(self, tool, mock_mcp_service):
        """Test describing a table."""
        mock_mcp_service.describe_table = AsyncMock(
            return_value={"name": "users", "columns": ["id", "email", "created_at"]}
        )

        result = tool._describe_table("users")

        assert "users" in result
        assert "id" in result
        assert "email" in result

    def test_get_schema_info(self, tool, mock_mcp_service):
        """Test getting schema info."""
        mock_mcp_service.get_schema_info = AsyncMock(
            return_value={"tables": ["users", "sessions"], "version": "1.0"}
        )

        result = tool._get_schema_info()

        assert "users" in result
        assert "sessions" in result

    # === Edge Cases ===

    def test_query_with_special_characters(self, tool, mock_mcp_service):
        """Test query with special characters."""
        mock_mcp_service.execute_query.return_value = "Results"

        result = tool._run("Show users with email @example.com")

        # Should not error, should process normally
        assert "Error" not in result or "Results" in result

    def test_query_trimming(self, tool, mock_mcp_service):
        """Test that queries are trimmed."""
        mock_mcp_service.execute_query.return_value = "Results"

        result = tool._run("  Show users  ")

        # Should call with trimmed query
        call_args = mock_mcp_service.execute_query.call_args
        assert call_args[1]["query"] == "Show users"

    def test_minimum_valid_query(self, tool, mock_mcp_service):
        """Test minimum valid query length (3 characters)."""
        mock_mcp_service.execute_query.return_value = "Results"

        result = tool._run("abc")

        assert "Error" not in result or "Results" in result

    def test_maximum_valid_query(self, tool, mock_mcp_service):
        """Test maximum valid query length (1000 characters)."""
        mock_mcp_service.execute_query.return_value = "Results"

        max_query = "a" * 1000
        result = tool._run(max_query)

        assert "Error" not in result or "Results" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
