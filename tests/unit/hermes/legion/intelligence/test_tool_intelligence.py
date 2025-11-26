"""Unit tests for ToolIntelligence service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.hermes.legion.intelligence.tool_intelligence import ToolIntelligence


class MockTool:
    def __init__(self, name, description):
        self.name = name
        self.description = description


@pytest.fixture
def mock_async_llm_service():
    service = MagicMock()
    service.generate_async = AsyncMock()
    with patch(
        "app.hermes.legion.intelligence.tool_intelligence.get_async_llm_service",
        return_value=service,
    ):
        yield service


@pytest.mark.asyncio
async def test_recommend_tools(mock_async_llm_service):
    # Setup
    intelligence = ToolIntelligence()
    mock_async_llm_service.generate_async.return_value = """
    {
        "selected_tools": ["tool1"]
    }
    """

    tools = [MockTool("tool1", "desc1"), MockTool("tool2", "desc2")]

    # Execute
    selected = await intelligence.recommend_tools("role", "task", tools)

    # Verify
    assert len(selected) == 1
    assert selected[0] == "tool1"


@pytest.mark.asyncio
async def test_recommend_tools_validation(mock_async_llm_service):
    # Setup
    intelligence = ToolIntelligence()
    mock_async_llm_service.generate_async.return_value = """
    {
        "selected_tools": ["tool1", "invalid_tool"]
    }
    """

    tools = [MockTool("tool1", "desc1")]

    # Execute
    selected = await intelligence.recommend_tools("role", "task", tools)

    # Verify
    assert len(selected) == 1
    assert selected[0] == "tool1"


@pytest.mark.asyncio
async def test_rank_tools_by_relevance(mock_async_llm_service):
    # Setup
    intelligence = ToolIntelligence()
    mock_async_llm_service.generate_async.return_value = """
    {
        "tool_relevance": {
            "tool1": 0.9,
            "tool2": 0.1
        }
    }
    """

    tools = [MockTool("tool1", "d"), MockTool("tool2", "d")]

    # Execute
    ranking = await intelligence.rank_tools_by_relevance("task", tools)

    # Verify
    assert ranking["tool1"] == 0.9
    assert ranking["tool2"] == 0.1
