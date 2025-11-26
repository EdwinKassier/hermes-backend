"""
Unit tests for MCP Database Service.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.shared.services.SupabaseDatabaseService import (
    SupabaseAuthenticationError,
    SupabaseConnectionError,
    SupabaseDatabaseService,
    SupabaseDatabaseServiceError,
    SupabaseTimeoutError,
    SupabaseValidationError,
    get_supabase_database_service,
)


class TestSupabaseDatabaseService:
    """Test MCP Database Service."""

    @pytest.fixture
    def mock_httpx_client(self):
        """Mock httpx.AsyncClient."""
        with patch("httpx.AsyncClient") as mock:
            client = AsyncMock()
            client.aclose = AsyncMock()
            mock.return_value.__aenter__.return_value = client
            yield client

    @pytest.fixture
    def service(self, mock_httpx_client):
        """Create MCP service instance with mocked client."""
        return SupabaseDatabaseService(
            mcp_server_url="http://localhost:3001",
            api_key="test-key",
            enable_cache=True,
        )

    @pytest.mark.asyncio
    async def test_execute_query_success(self, service, mock_httpx_client):
        """Test successful query execution."""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "test data"}
        mock_response.raise_for_status.return_value = None
        mock_httpx_client.post.return_value = mock_response

        result = await service.execute_query("test query", include_analysis=False)

        assert result == "test data"
        mock_httpx_client.post.assert_called_once_with(
            "http://localhost:3001/query",
            json={"query": "test query", "format": "json"},
        )

    @pytest.mark.asyncio
    async def test_execute_query_with_analysis(self, service, mock_httpx_client):
        """Test query execution with AI analysis."""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "test data"}
        mock_response.raise_for_status.return_value = None
        mock_httpx_client.post.return_value = mock_response

        # Mock GeminiService
        with patch(
            "app.shared.services.SupabaseDatabaseService.GeminiService"
        ) as mock_gemini:
            mock_gemini_instance = MagicMock()
            mock_gemini_instance.generate_response.return_value = "AI analysis"
            mock_gemini.return_value = mock_gemini_instance

            result = await service.execute_query("test query", include_analysis=True)

            assert "AI analysis" in result
            assert "--- Raw Data ---" in result
            assert "test data" in result

    @pytest.mark.asyncio
    async def test_execute_query_http_error(self, service, mock_httpx_client):
        """Test query execution with HTTP error."""
        # Mock HTTP error
        mock_httpx_client.post.side_effect = httpx.HTTPError("Connection failed")

        with pytest.raises(
            SupabaseDatabaseServiceError, match="Query execution failed"
        ):
            await service.execute_query("test query")

    @pytest.mark.asyncio
    async def test_execute_query_401_error(self, service, mock_httpx_client):
        """Test query execution with 401 authentication error."""
        # Mock 401 error
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_error = httpx.HTTPError("Unauthorized")
        mock_error.response = mock_response
        mock_httpx_client.post.side_effect = mock_error

        with pytest.raises(SupabaseConnectionError, match="Authentication failed"):
            await service.execute_query("test query")

    @pytest.mark.asyncio
    async def test_execute_query_404_error(self, service, mock_httpx_client):
        """Test query execution with 404 server not found error."""
        # Mock 404 error
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_error = httpx.HTTPError("Not Found")
        mock_error.response = mock_response
        mock_httpx_client.post.side_effect = mock_error

        with pytest.raises(SupabaseConnectionError, match="MCP server not found"):
            await service.execute_query("test query")

    @pytest.mark.asyncio
    async def test_list_tables_success(self, service, mock_httpx_client):
        """Test successful table listing."""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"tables": ["users", "conversations"]}
        mock_response.raise_for_status.return_value = None
        mock_httpx_client.get.return_value = mock_response

        tables = await service.list_tables()

        assert tables == ["users", "conversations"]
        mock_httpx_client.get.assert_called_once_with("http://localhost:3001/tables")

    @pytest.mark.asyncio
    async def test_describe_table_success(self, service, mock_httpx_client):
        """Test successful table description."""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"columns": ["id", "name", "email"]}
        mock_response.raise_for_status.return_value = None
        mock_httpx_client.get.return_value = mock_response

        description = await service.describe_table("users")

        assert description == {"columns": ["id", "name", "email"]}
        mock_httpx_client.get.assert_called_once_with(
            "http://localhost:3001/tables/users"
        )

    @pytest.mark.asyncio
    async def test_get_schema_info_success(self, service, mock_httpx_client):
        """Test successful schema info retrieval."""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "tables": {"users": {"columns": ["id", "name"]}}
        }
        mock_response.raise_for_status.return_value = None
        mock_httpx_client.get.return_value = mock_response

        schema_info = await service.get_schema_info()

        assert schema_info == {"tables": {"users": {"columns": ["id", "name"]}}}
        mock_httpx_client.get.assert_called_once_with("http://localhost:3001/schema")

    def test_query_caching(self, service):
        """Test query result caching."""
        # Mock the HTTP client for this test
        with patch.object(service, "_execute_mcp_query") as mock_execute:
            mock_execute.return_value = "cached result"

            # First call
            result1 = asyncio.run(service.execute_query("test query", use_cache=True))
            assert result1 == "cached result"
            assert mock_execute.call_count == 1

            # Second call (should use cache)
            result2 = asyncio.run(service.execute_query("test query", use_cache=True))
            assert result2 == "cached result"
            assert mock_execute.call_count == 1  # No additional call

    def test_cache_ttl_expiry(self, service):
        """Test cache TTL expiry."""
        # Set short TTL for testing
        service.cache_ttl = 0.1  # 100ms

        with patch.object(service, "_execute_mcp_query") as mock_execute:
            mock_execute.return_value = "fresh result"

            # First call
            result1 = asyncio.run(service.execute_query("test query", use_cache=True))
            assert result1 == "fresh result"
            assert mock_execute.call_count == 1

            # Wait for cache to expire
            import time

            time.sleep(0.2)

            # Second call (should not use cache)
            result2 = asyncio.run(service.execute_query("test query", use_cache=True))
            assert result2 == "fresh result"
            assert mock_execute.call_count == 2  # Additional call made

    def test_cache_disabled(self, service):
        """Test query execution with caching disabled."""
        service.enable_cache = False

        with patch.object(service, "_execute_mcp_query") as mock_execute:
            mock_execute.return_value = "result"

            # Multiple calls should not use cache
            asyncio.run(service.execute_query("test query", use_cache=True))
            asyncio.run(service.execute_query("test query", use_cache=True))

            assert mock_execute.call_count == 2

    @pytest.mark.asyncio
    async def test_close_client(self, service, mock_httpx_client):
        """Test client cleanup."""
        await service.close()
        mock_httpx_client.aclose.assert_called_once()

    def test_singleton_pattern(self):
        """Test singleton pattern for get_supabase_database_service."""
        with patch(
            "app.shared.services.SupabaseDatabaseService.get_env"
        ) as mock_get_env:
            mock_get_env.side_effect = lambda key: {
                "SUPABASE_MCP_SERVER_URL": "http://localhost:3001",
                "SUPABASE_MCP_API_KEY": "test-key",
            }.get(key)

            # First call
            service1 = get_supabase_database_service()
            assert service1 is not None

            # Second call should return same instance
            service2 = get_supabase_database_service()
            assert service1 is service2

    def test_missing_env_vars(self):
        """Test error handling for missing environment variables."""
        with patch(
            "app.shared.services.SupabaseDatabaseService.get_env"
        ) as mock_get_env:
            mock_get_env.return_value = None

            with pytest.raises(
                ValueError, match="SUPABASE_MCP_SERVER_URL and SUPABASE_MCP_API_KEY"
            ):
                get_supabase_database_service()

    def test_different_response_formats(self, service, mock_httpx_client):
        """Test handling of different response formats."""
        test_cases = [
            ({"result": "data"}, "data"),
            ({"data": "data"}, "data"),
            ("direct_string", "direct_string"),
            ({"other": "format"}, '{"other": "format"}'),
        ]

        for response_data, expected in test_cases:
            mock_response = MagicMock()
            mock_response.json.return_value = response_data
            mock_response.raise_for_status.return_value = None
            mock_httpx_client.post.return_value = mock_response

            result = asyncio.run(
                service.execute_query("test query", include_analysis=False)
            )
            assert result == expected
