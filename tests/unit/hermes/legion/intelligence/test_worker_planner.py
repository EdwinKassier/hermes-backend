"""Unit tests for WorkerPlanner service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.hermes.legion.intelligence.worker_planner import IntelligentWorkerPlanner
from app.hermes.legion.models import QueryComplexity, WorkerPlan


@pytest.fixture
def mock_async_llm_service():
    service = MagicMock()
    service.generate_async = AsyncMock()
    with patch(
        "app.hermes.legion.intelligence.worker_planner.get_async_llm_service",
        return_value=service,
    ):
        yield service


@pytest.mark.asyncio
async def test_plan_workers(mock_async_llm_service):
    # Setup
    planner = IntelligentWorkerPlanner()
    mock_async_llm_service.generate_async.return_value = """
    {
        "workers": [
            {
                "role": "researcher",
                "specialization": "history",
                "task_description": "Research history",
                "priority": 1,
                "estimated_duration": 30.0
            },
            {
                "role": "coder",
                "specialization": "python",
                "task_description": "Write code",
                "priority": 2,
                "estimated_duration": 45.0
            }
        ]
    }
    """

    complexity = QueryComplexity(
        score=0.6,
        dimensions={"technical": 0.5},
        suggested_workers=2,
        estimated_time_seconds=60.0,
    )

    # Execute
    plans = await planner.plan_workers("Task", complexity)

    # Verify
    assert len(plans) == 2
    assert isinstance(plans[0], WorkerPlan)
    assert plans[0].role == "researcher"
    assert plans[1].role == "coder"


@pytest.mark.asyncio
async def test_optimize_worker_count():
    planner = IntelligentWorkerPlanner()

    # Low complexity
    c1 = QueryComplexity(
        score=0.2, dimensions={}, suggested_workers=5, estimated_time_seconds=10
    )
    assert planner._optimize_worker_count(c1) <= 2

    # High complexity
    c2 = QueryComplexity(
        score=0.9, dimensions={}, suggested_workers=1, estimated_time_seconds=10
    )
    assert planner._optimize_worker_count(c2) >= 3
