"""Integration test for Intelligent Strategy."""

from unittest.mock import MagicMock, patch

import pytest

from app.hermes.legion.nodes.legion_orchestrator import (
    LegionWorkerState,
    legion_orchestrator_node,
    legion_synthesis_node,
    legion_worker_node,
)


async def async_return(val):
    return val


@pytest.fixture
def mock_gemini_service():
    service = MagicMock()

    # Patch all locations
    with (
        patch(
            "app.hermes.legion.nodes.legion_orchestrator.get_gemini_service",
            return_value=service,
        ),
        patch(
            "app.hermes.legion.intelligence.query_analyzer.get_gemini_service",
            return_value=service,
        ),
        patch(
            "app.hermes.legion.intelligence.worker_planner.get_gemini_service",
            return_value=service,
        ),
        patch(
            "app.hermes.legion.intelligence.tool_intelligence.get_gemini_service",
            return_value=service,
        ),
        patch(
            "app.hermes.legion.intelligence.adaptive_synthesizer.get_gemini_service",
            return_value=service,
        ),
        patch(
            "app.hermes.legion.utils.tool_allocator.get_gemini_service",
            return_value=service,
        ),
        patch(
            "app.hermes.legion.agents.factory.get_gemini_service", return_value=service
        ),
    ):
        yield service


@pytest.mark.asyncio
async def test_intelligent_strategy_flow(mock_gemini_service):
    """Test end-to-end intelligent strategy orchestration."""

    # Setup mock responses
    mock_gemini_service.generate_gemini_response.side_effect = [
        # 1. QueryAnalyzer - analyze_complexity
        """{
            "score": 0.7,
            "dimensions": {"technical": 0.8, "creative": 0.3, "reasoning": 0.6, "context": 0.4},
            "suggested_workers": 3,
            "estimated_time_seconds": 45.0
        }""",
        # 2. WorkerPlanner - plan_workers
        """{
            "workers": [
                {"role": "researcher", "specialization": "technical", "task_description": "Research technical aspects", "priority": 1, "estimated_duration": 20.0},
                {"role": "analyzer", "specialization": "data", "task_description": "Analyze data", "priority": 2, "estimated_duration": 15.0}
            ]
        }""",
        # 3. ToolIntelligence - recommend_tools for worker 1
        """{"selected_tools": ["web_search"]}""",
        # 4. ToolIntelligence - recommend_tools for worker 2
        """{"selected_tools": ["database_query"]}""",
        # 5. Tool allocation for worker 1 (executor)
        """{"selected_tools": []}""",
        # 6. Tool allocation for worker 2 (executor)
        """{"selected_tools": []}""",
        # 7. AdaptiveSynthesizer - assess_result_quality
        """{
            "completeness": 0.9,
            "coherence": 0.85,
            "relevance": 0.9,
            "confidence": 0.8,
            "agreement": 0.75
        }""",
        # 8. AdaptiveSynthesizer - synthesize_adaptively
        "Intelligent synthesis of results",
    ]

    # Mock agent execution
    with patch(
        "app.hermes.legion.agents.factory.AgentFactory.create_agent"
    ) as mock_create_agent:
        mock_agent = MagicMock()
        mock_agent.execute_task_async = MagicMock(
            side_effect=[
                async_return("Researcher result"),
                async_return("Analyzer result"),
            ]
        )
        mock_create_agent.return_value = mock_agent

        # 1. Orchestrator
        state = {
            "messages": [{"content": "Complex technical query", "role": "user"}],
            "user_id": "test_user",
            "persona": "hermes",
            "legion_strategy": "intelligent",
            "metadata": {},
            "collected_info": {},
        }

        orchestrator_result = await legion_orchestrator_node(state)

        # Verify intelligent planning
        assert orchestrator_result["legion_strategy"] == "intelligent"
        workers = orchestrator_result["metadata"]["legion_worker_plans"]
        assert len(workers) == 2
        # Verify workers have required fields and metadata
        assert "role" in workers[0]
        assert "task_description" in workers[0]
        assert "metadata" in workers[0]
        assert "complexity_score" in workers[0]["metadata"]
        # Verify complexity score is valid (0.0 to 1.0)
        assert 0.0 <= workers[0]["metadata"]["complexity_score"] <= 1.0

        # 2. Execute Workers
        worker_results = {}
        for worker in workers:
            worker_state = LegionWorkerState(
                worker_id=worker["worker_id"],
                role=worker["role"],
                task_description=worker["task_description"],
                tools=worker["tools"],
                user_id=state["user_id"],
                persona=state["persona"],
                context=state.get("collected_info", {}),
            )

            result = await legion_worker_node(worker_state)
            worker_results.update(result["legion_results"])

        # Verify execution
        assert len(worker_results) == 2

        # 3. Synthesis
        state["legion_results"] = worker_results
        synthesis_result = await legion_synthesis_node(state)

        # Verify adaptive synthesis
        assert "Intelligent synthesis" in synthesis_result["messages"][-1]["content"]
