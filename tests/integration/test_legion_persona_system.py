"""Integration tests for the complete Legion persona system."""

from unittest.mock import AsyncMock, patch

import pytest

from app.hermes.legion.strategies.council import CouncilStrategy
from app.hermes.legion.strategies.intelligent import IntelligentStrategy
from app.hermes.legion.strategies.parallel import ParallelStrategy
from app.hermes.legion.utils.persona_generator import LegionPersonaGenerator


class TestLegionPersonaSystemIntegration:
    """Integration tests for the complete persona system."""

    @pytest.fixture
    def mock_llm_service(self):
        """Mock LLM service for testing."""
        mock_service = AsyncMock()
        mock_service.generate_async.return_value = '{"workers": [{"role": "researcher", "task_description": "Research AI", "priority": 1, "estimated_duration": 30}]}'
        return mock_service

    @pytest.mark.asyncio
    async def test_parallel_strategy_persona_generation(self, mock_llm_service):
        """Test that parallel strategy generates personas correctly."""
        with (
            patch(
                "app.hermes.legion.strategies.parallel.get_async_llm_service",
                return_value=mock_llm_service,
            ),
            patch(
                "app.hermes.legion.utils.persona_generator.get_async_llm_service",
                return_value=mock_llm_service,
            ),
        ):

            strategy = ParallelStrategy()
            query = "Research artificial intelligence"

            # Mock the decomposer to return a simple task
            with (
                patch.object(strategy.decomposer, "decompose_task") as mock_decompose,
                patch.object(
                    strategy.decomposer, "analyze_task_dependencies"
                ) as mock_analyze,
            ):

                mock_decompose.return_value = [
                    {
                        "description": "Research AI fundamentals",
                        "agent_type": "research",
                        "keywords": ["research", "AI"],
                    }
                ]
                mock_analyze.return_value = {
                    "execution_levels": [["task_0"]],
                    "tasks": {
                        "task_0": type(
                            "Task",
                            (),
                            {
                                "description": "Research AI fundamentals",
                                "agent_type": "research",
                                "keywords": ["research", "AI"],
                                "dependencies": [],
                            },
                        )()
                    },
                    "is_sequential": False,
                }

                workers = await strategy.generate_workers(query, {})

                assert len(workers) > 0
                for worker in workers:
                    assert "persona" in worker
                    assert isinstance(worker["persona"], str)
                    assert len(worker["persona"]) > 0

    @pytest.mark.asyncio
    async def test_intelligent_strategy_persona_generation(self, mock_llm_service):
        """Test that intelligent strategy generates personas correctly."""
        with (
            patch(
                "app.hermes.legion.strategies.intelligent.get_async_llm_service",
                return_value=mock_llm_service,
            ),
            patch(
                "app.hermes.legion.utils.persona_generator.get_async_llm_service",
                return_value=mock_llm_service,
            ),
            patch(
                "app.hermes.legion.strategies.intelligent.get_all_tools",
                return_value=[],
            ),
        ):

            strategy = IntelligentStrategy()
            query = "Analyze data patterns"

            # Mock the complex intelligence pipeline
            with (
                patch.object(
                    strategy.query_analyzer, "analyze_complexity"
                ) as mock_analyze,
                patch.object(strategy.worker_planner, "plan_workers") as mock_plan,
                patch.object(
                    strategy.tool_intelligence, "recommend_tools"
                ) as mock_recommend,
                patch.object(
                    strategy.feedback_learner,
                    "get_optimal_worker_count",
                    return_value=2,
                ),
                patch.object(
                    strategy.cost_optimizer, "should_reduce_workers", return_value=2
                ),
            ):

                # Mock complexity analysis
                mock_analyze.return_value = type(
                    "Complexity",
                    (),
                    {
                        "score": 0.7,
                        "dimensions": {"technical": 0.8},
                        "suggested_workers": 2,
                    },
                )()

                # Mock worker planning
                mock_plan.return_value = [
                    type(
                        "WorkerPlan",
                        (),
                        {
                            "worker_id": "worker_0_research",
                            "role": "research",
                            "specialization": "analysis",
                            "task_description": "Analyze data patterns",
                            "tools": [],
                            "priority": 1,
                            "estimated_duration": 30.0,
                        },
                    )()
                ]

                # Mock tool recommendation
                mock_recommend.return_value = []

                workers = await strategy.generate_workers(query, {})

                assert len(workers) > 0
                for worker in workers:
                    assert "persona" in worker
                    assert isinstance(worker["persona"], str)

    @pytest.mark.asyncio
    async def test_council_strategy_persona_assignment(self, mock_llm_service):
        """Test that council strategy assigns personas correctly."""
        with patch(
            "app.hermes.legion.strategies.council.get_async_llm_service",
            return_value=mock_llm_service,
        ):

            strategy = CouncilStrategy()
            query = "Should I invest in AI stocks?"

            # Mock the LLM response for persona generation
            mock_llm_service.generate_async.return_value = """{
                "personas": [
                    {"name": "optimist", "description": "Positive outlook", "perspective": "best case"},
                    {"name": "pessimist", "description": "Risk-focused", "perspective": "worst case"}
                ]
            }"""

            workers = await strategy.generate_workers(query, {})

            assert len(workers) == 2
            persona_names = [worker["persona"] for worker in workers]
            assert "optimist" in persona_names
            assert "pessimist" in persona_names

    @pytest.mark.asyncio
    async def test_persona_generator_role_templates_coverage(self):
        """Test that all expected roles have persona templates."""
        generator = LegionPersonaGenerator()

        # Test that all roles from agent factory are covered
        expected_roles = ["research", "code", "analysis", "data", "general"]

        for role in expected_roles:
            assert role in generator.role_persona_templates
            templates = generator.role_persona_templates[role]
            assert len(templates) == 5  # Should have 5 templates per role

    @pytest.mark.asyncio
    async def test_persona_system_error_recovery(self, mock_llm_service):
        """Test error recovery in the persona system."""
        # Mock LLM service to fail
        mock_llm_service.generate_async.side_effect = Exception(
            "LLM Service Unavailable"
        )

        generator = LegionPersonaGenerator()

        # Should still generate fallback personas
        workers = [
            {
                "role": "research",
                "task_description": "Research topic",
                "worker_id": "worker1",
            },
            {"role": "code", "task_description": "Code task", "worker_id": "worker2"},
        ]

        result = await generator.generate_personas_for_workers(workers)

        assert len(result) == 2
        assert result[0]["persona"] == "research_specialist"  # Fallback
        assert result[1]["persona"] == "code_specialist"  # Fallback

    def test_persona_context_isolation(self):
        """Test that persona contexts are properly isolated."""
        from app.hermes.legion.utils.persona_context import (
            LegionPersonaContext,
            get_current_legion_persona,
        )

        # Test nested contexts
        assert get_current_legion_persona() == "legion"

        with LegionPersonaContext("strategy_persona"):
            assert get_current_legion_persona() == "strategy_persona"

            with LegionPersonaContext("worker_persona"):
                assert get_current_legion_persona() == "worker_persona"

            assert get_current_legion_persona() == "strategy_persona"

        assert get_current_legion_persona() == "legion"

    @pytest.mark.asyncio
    async def test_end_to_end_persona_flow(self, mock_llm_service):
        """Test complete persona flow from strategy to worker."""
        with patch(
            "app.hermes.legion.utils.persona_generator.get_async_llm_service",
            return_value=mock_llm_service,
        ):

            # Test persona generation
            generator = LegionPersonaGenerator()
            workers = [{"role": "research", "task_description": "Research AI"}]

            result = await generator.generate_personas_for_workers(workers)

            assert len(result) == 1
            assert "persona" in result[0]
            assert isinstance(result[0]["persona"], str)

            # Verify the persona is from the expected templates
            research_templates = generator.role_persona_templates["research"]
            assert result[0]["persona"] in research_templates
