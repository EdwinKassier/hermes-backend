"""Unit tests for QueryAnalyzer service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.hermes.legion.intelligence.query_analyzer import QueryAnalyzer
from app.hermes.legion.models import Domain, QueryComplexity


@pytest.fixture
def mock_async_llm_service():
    service = MagicMock()
    # Make generate_async an AsyncMock
    service.generate_async = AsyncMock()
    with patch(
        "app.hermes.legion.intelligence.query_analyzer.get_async_llm_service",
        return_value=service,
    ):
        yield service


@pytest.mark.asyncio
async def test_analyze_complexity(mock_async_llm_service):
    # Setup
    analyzer = QueryAnalyzer()
    mock_async_llm_service.generate_async.return_value = """
    {
        "score": 0.8,
        "dimensions": {
            "technical": 0.9,
            "creative": 0.2,
            "reasoning": 0.7,
            "context": 0.5
        },
        "suggested_workers": 3,
        "estimated_time_seconds": 60.0
    }
    """

    # Execute
    complexity = await analyzer.analyze_complexity("Complex coding task")

    # Verify
    assert isinstance(complexity, QueryComplexity)
    assert complexity.score == 0.8
    assert complexity.suggested_workers == 3
    assert complexity.dimensions["technical"] == 0.9


@pytest.mark.asyncio
async def test_identify_domains(mock_async_llm_service):
    # Setup
    analyzer = QueryAnalyzer()
    mock_async_llm_service.generate_async.return_value = """
    {
        "domains": ["coding", "research"]
    }
    """

    # Execute
    domains = await analyzer.identify_domains("Research and code this")

    # Verify
    assert len(domains) == 2
    assert Domain.CODING in domains
    assert Domain.RESEARCH in domains


@pytest.mark.asyncio
async def test_analyze_complexity_error_handling(mock_async_llm_service):
    # Setup
    analyzer = QueryAnalyzer()
    mock_async_llm_service.generate_async.side_effect = Exception("API Error")

    # Execute
    complexity = await analyzer.analyze_complexity("Task")

    # Verify fallback
    assert complexity.score == 0.5
    assert complexity.suggested_workers == 2
