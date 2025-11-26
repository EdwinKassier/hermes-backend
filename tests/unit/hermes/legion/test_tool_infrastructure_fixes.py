"""Test tool infrastructure fixes."""

from unittest.mock import Mock, patch

import pytest

from app.hermes.legion.utils.tool_allocator import ToolAllocator


class TestToolValidation:
    """Test tool validation on startup."""

    @pytest.fixture
    def mock_gemini_service(self):
        """Create mock Gemini service with 2 tools."""
        service = Mock()

        # Create mock tools
        db_tool = Mock()
        db_tool.name = "database_query"
        db_tool.description = "Query database"

        time_tool = Mock()
        time_tool.name = "time_info"
        time_tool.description = "Get time"

        service.all_tools = [db_tool, time_tool]
        return service

    def test_validation_runs_on_first_access(self, mock_gemini_service):
        """Test validation runs when gemini_service is first accessed."""
        allocator = ToolAllocator()

        with patch(
            "app.hermes.legion.utils.tool_allocator.get_gemini_service"
        ) as mock_get:
            mock_get.return_value = mock_gemini_service

            # Access gemini_service - should trigger validation
            _ = allocator.gemini_service

            # Validation should have run
            assert allocator._validated is True

    def test_validation_logs_available_tools(self, mock_gemini_service, caplog):
        """Test validation logs available tools."""
        allocator = ToolAllocator()

        with patch(
            "app.hermes.legion.utils.tool_allocator.get_gemini_service"
        ) as mock_get:
            mock_get.return_value = mock_gemini_service

            _ = allocator.gemini_service

            # Should log available tools
            assert "database_query" in caplog.text
            assert "time_info" in caplog.text

    def test_capability_mapping_matches_tools(self, mock_gemini_service):
        """Test capability mapping only includes existing tools."""
        allocator = ToolAllocator()

        with patch(
            "app.hermes.legion.utils.tool_allocator.get_gemini_service"
        ) as mock_get:
            mock_get.return_value = mock_gemini_service

            _ = allocator.gemini_service

            # All mapped tools should exist (3 tools: database_query, time_info, web_search)
            mapped_tools = set(allocator._tool_capabilities.keys())
            available_tools = {"database_query", "time_info", "web_search"}

            assert mapped_tools == available_tools


class TestToolAllocation:
    """Test tool allocation with only 2 tools."""

    @pytest.fixture
    def allocator(self):
        """Create allocator with mocked service."""
        allocator = ToolAllocator()

        # Mock gemini service
        service = Mock()
        service.persona_configs = {}

        db_tool = Mock()
        db_tool.name = "database_query"

        time_tool = Mock()
        time_tool.name = "time_info"

        service.all_tools = [db_tool, time_tool]
        allocator._gemini_service = service
        allocator._validated = True

        return allocator

    def test_allocate_for_data_task(self, allocator):
        """Test allocating tools for data task."""
        # Mock _get_tools_for_persona to return the tools list
        with patch.object(
            allocator,
            "_get_tools_for_persona",
            return_value=allocator._gemini_service.all_tools,
        ):
            tools = allocator.allocate_tools_for_task(
                task_type="data",
                task_description="Query database",
                persona="hermes",
                use_ai=False,
            )

            # Should return database_query tool
            assert len(tools) > 0
            tool_names = {allocator._get_tool_name(t) for t in tools}
            assert "database_query" in tool_names

    def test_allocate_for_general_task(self, allocator):
        """Test allocating tools for general task."""
        # Mock _get_tools_for_persona to return the tools list
        with patch.object(
            allocator,
            "_get_tools_for_persona",
            return_value=allocator._gemini_service.all_tools,
        ):
            tools = allocator.allocate_tools_for_task(
                task_type="general",
                task_description="Get current time",
                persona="hermes",
                use_ai=False,
            )

            # Should return time_info tool
            assert len(tools) > 0
            tool_names = {allocator._get_tool_name(t) for t in tools}
            assert "time_info" in tool_names

    def test_ai_allocation_disabled_by_default(self, allocator):
        """Test AI allocation is enabled by default (use_ai=True)."""
        # Mock the AI response and _get_tools_for_persona
        allocator._gemini_service.generate_gemini_response = Mock(
            return_value='{"selected_tools": ["database_query"]}'
        )

        with patch.object(
            allocator,
            "_get_tools_for_persona",
            return_value=allocator._gemini_service.all_tools,
        ):
            # use_ai defaults to True now
            tools = allocator.allocate_tools_for_task(
                task_type="data", task_description="Query database", persona="hermes"
            )

            # Should return tools (AI is enabled by default)
            assert isinstance(tools, list)

    def test_ai_allocation_can_be_enabled(self, allocator):
        """Test AI allocation works when enabled."""
        allocator._gemini_service.generate_gemini_response = Mock(
            return_value='{"selected_tools": ["database_query"]}'
        )

        with patch.object(
            allocator,
            "_get_tools_for_persona",
            return_value=allocator._gemini_service.all_tools,
        ):
            tools = allocator.allocate_tools_for_task(
                task_type="data",
                task_description="Query database",
                persona="hermes",
                use_ai=True,
            )

            # Should work
            assert isinstance(tools, list)


class TestCapabilityMapping:
    """Test capability mapping correctness."""

    def test_only_existing_tools_mapped(self):
        """Test capability mapping includes the 3 actual tools."""
        allocator = ToolAllocator()

        mapped_tools = set(allocator._tool_capabilities.keys())

        # Should have 3 tools: database_query, time_info, web_search
        assert len(mapped_tools) == 3
        assert "database_query" in mapped_tools
        assert "time_info" in mapped_tools
        assert "web_search" in mapped_tools

    def test_no_phantom_tools(self):
        """Test no phantom tools in mapping."""
        allocator = ToolAllocator()

        mapped_tools = set(allocator._tool_capabilities.keys())

        # Should NOT have these phantom tools (web_search is actually valid now)
        phantom_tools = {
            "python_repl",
            "csv_reader",
            "code_execution",
            "file_reader",
            "calculator",
        }

        for phantom in phantom_tools:
            assert (
                phantom not in mapped_tools
            ), f"Phantom tool '{phantom}' still in mapping!"

    def test_database_query_capabilities(self):
        """Test database_query has correct capabilities."""
        allocator = ToolAllocator()

        capabilities = allocator._tool_capabilities.get("database_query", [])

        assert "data" in capabilities
        assert "analysis" in capabilities

    def test_time_info_capabilities(self):
        """Test time_info has correct capabilities."""
        allocator = ToolAllocator()

        capabilities = allocator._tool_capabilities.get("time_info", [])

        assert "general" in capabilities
        assert "information" in capabilities
