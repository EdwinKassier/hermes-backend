"""
Unit tests for Supabase Database Service.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.shared.services.SupabaseDatabaseService import (
    SupabaseDatabaseService,
    SupabaseDatabaseServiceError,
    SupabaseValidationError,
)


class TestSupabaseDatabaseService:
    """Test Supabase Database Service."""

    @pytest.fixture
    def mock_supabase_client(self):
        """Mock Supabase Client."""
        with patch("app.shared.services.SupabaseDatabaseService.create_client") as mock:
            client = MagicMock()
            mock.return_value = client
            yield client

    @pytest.fixture
    def service(self, mock_supabase_client):
        """Create service instance with mocked client."""
        return SupabaseDatabaseService(
            supabase_url="http://localhost:54321",
            supabase_key="test-key",
        )

    @pytest.fixture
    def mock_llm_service(self):
        """Mock LLMService."""
        with patch("app.shared.utils.service_loader.get_llm_service") as mock_get:
            mock_instance = MagicMock()
            mock_get.return_value = mock_instance
            yield mock_instance

    @pytest.mark.asyncio
    async def test_execute_query_success_select(
        self, service, mock_supabase_client, mock_llm_service
    ):
        """Test successful SELECT query execution."""
        # Mock table list
        with patch.object(service, "_get_table_list", return_value=["users"]):
            # Mock LLM response
            query_plan = {
                "operation": "select",
                "table": "users",
                "columns": "*",
                "limit": 10,
            }
            mock_llm_service.generate_response.return_value = json.dumps(query_plan)

            # Mock Supabase response
            mock_response = MagicMock()
            mock_response.data = [{"id": 1, "name": "Test User"}]
            mock_supabase_client.table.return_value.select.return_value.limit.return_value.execute.return_value = (
                mock_response
            )

            result = await service.execute_query(
                "Show me users", include_analysis=False
            )

            assert "Query Results from 'users'" in result
            assert "Test User" in result

            # Verify LLM service called
            mock_llm_service.generate_response.assert_called_once()

            # Verify Supabase called
            mock_supabase_client.table.assert_called_with("users")

    @pytest.mark.asyncio
    async def test_execute_query_validation_error(self, service):
        """Test query validation."""
        with pytest.raises(SupabaseValidationError, match="Query too short"):
            await service.execute_query("hi")

        with pytest.raises(
            SupabaseValidationError, match="Write operations are not allowed"
        ):
            await service.execute_query("DROP TABLE users")

    @pytest.mark.asyncio
    async def test_list_tables(self, service):
        """Test list_tables operation."""
        with patch.object(service, "_get_table_list", return_value=["users", "posts"]):
            tables = await service.list_tables()
            assert tables == ["users", "posts"]

    @pytest.mark.asyncio
    async def test_describe_table(self, service, mock_supabase_client):
        """Test describe_table operation."""
        # Mock Supabase response for describe (limit 1)
        mock_response = MagicMock()
        mock_response.data = [{"id": 1, "name": "Test"}]
        mock_supabase_client.table.return_value.select.return_value.limit.return_value.execute.return_value = (
            mock_response
        )

        result = await service.describe_table("users")

        assert "schema" in result
        assert "Table: users" in result["schema"]
        assert (
            "id (int)" in result["schema"]
            or "id (int" in result["schema"]
            or "id" in result["schema"]
        )

    @pytest.mark.asyncio
    async def test_execute_query_count(
        self, service, mock_supabase_client, mock_llm_service
    ):
        """Test COUNT query execution."""
        with patch.object(service, "_get_table_list", return_value=["users"]):
            # Mock LLM response
            query_plan = {"operation": "count", "table": "users"}
            mock_llm_service.generate_response.return_value = json.dumps(query_plan)

            # Mock Supabase response
            mock_response = MagicMock()
            mock_response.count = 42
            mock_supabase_client.table.return_value.select.return_value.execute.return_value = (
                mock_response
            )

            result = await service.execute_query("Count users", include_analysis=False)

            assert "42 records" in result
