"""Unit tests for AdaptiveSynthesizer service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.hermes.legion.intelligence.adaptive_synthesizer import AdaptiveSynthesizer
from app.hermes.legion.models import QualityMetrics


@pytest.fixture
def mock_async_llm_service():
    service = MagicMock()
    service.generate_async = AsyncMock()
    with patch(
        "app.hermes.legion.intelligence.adaptive_synthesizer.get_async_llm_service",
        return_value=service,
    ):
        yield service


@pytest.mark.asyncio
async def test_assess_result_quality(mock_async_llm_service):
    # Setup
    synthesizer = AdaptiveSynthesizer()
    mock_async_llm_service.generate_async.return_value = """
    {
        "completeness": 0.8,
        "coherence": 0.9,
        "relevance": 0.9,
        "confidence": 0.8,
        "agreement": 0.7
    }
    """

    results = {"worker1": {"result": "res1", "status": "success"}}

    # Execute
    quality = await synthesizer.assess_result_quality(results)

    # Verify
    assert isinstance(quality, QualityMetrics)
    assert quality.completeness == 0.8


@pytest.mark.asyncio
async def test_synthesize_adaptively(mock_async_llm_service):
    # Setup
    synthesizer = AdaptiveSynthesizer()
    mock_async_llm_service.generate_async.return_value = "Final Answer"

    results = {"worker1": {"result": "res1"}}
    quality = QualityMetrics(
        completeness=0.9, coherence=0.9, relevance=0.9, confidence=0.9, agreement=0.9
    )

    # Execute
    response = await synthesizer.synthesize_adaptively(
        "query", results, quality, "council", "hermes"
    )

    # Verify
    assert response == "Final Answer"
    # Verify prompt contains instructions
    call_args = mock_async_llm_service.generate_async.call_args
    assert "Synthesize a comprehensive" in call_args[0][0]


@pytest.mark.asyncio
async def test_synthesize_adaptively_low_quality(mock_async_llm_service):
    # Setup
    synthesizer = AdaptiveSynthesizer()
    mock_async_llm_service.generate_async.return_value = "Final Answer"

    results = {"worker1": {"result": "res1"}}
    quality = QualityMetrics(
        completeness=0.2, coherence=0.9, relevance=0.9, confidence=0.9, agreement=0.9
    )

    # Execute
    await synthesizer.synthesize_adaptively(
        "query", results, quality, "council", "hermes"
    )

    # Verify prompt contains warning
    call_args = mock_async_llm_service.generate_async.call_args
    assert "WARNING: Results are incomplete" in call_args[0][0]
