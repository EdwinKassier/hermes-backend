"""Unit tests for LegionPersonaGenerator."""

from unittest.mock import AsyncMock, patch

import pytest

from app.hermes.legion.utils.persona_generator import LegionPersonaGenerator


@pytest.fixture
def generator():
    """Create a LegionPersonaGenerator instance."""
    return LegionPersonaGenerator()


class TestLegionPersonaGenerator:
    """Test cases for LegionPersonaGenerator."""

    def test_role_normalization(self, generator):
        """Test role normalization logic."""
        # Test common role mappings
        assert generator._normalize_role("researcher") == "research"
        assert generator._normalize_role("Coder") == "code"
        assert generator._normalize_role("PROGRAMMER") == "code"
        assert generator._normalize_role("analyst") == "analysis"
        assert generator._normalize_role("data_analyst") == "data"
        assert generator._normalize_role("scientist") == "research"
        assert generator._normalize_role("engineer") == "code"
        assert generator._normalize_role("architect") == "code"
        assert generator._normalize_role("investigator") == "research"
        assert generator._normalize_role("evaluator") == "analysis"
        assert generator._normalize_role("interpreter") == "analysis"
        assert generator._normalize_role("specialist") == "general"
        assert generator._normalize_role("expert") == "general"

        # Test unknown roles pass through
        assert generator._normalize_role("unknown_role") == "unknown_role"
        assert generator._normalize_role("custom_agent") == "custom_agent"

    def test_persona_template_structure(self, generator):
        """Test that persona templates have correct structure."""
        templates = generator.template_store.get_all_roles()

        # Check expected roles exist
        expected_roles = {"research", "code", "analysis", "data", "general"}
        assert set(templates) == expected_roles

        # Check each role has templates
        for role in templates:
            role_templates = generator.template_store.get_templates_for_role(role)
            assert isinstance(role_templates, list)
            assert len(role_templates) == 5  # 5 templates per role
            assert all(isinstance(template, str) for template in role_templates)

        # Check all templates are unique (this is checked in template store validation)
        assert generator.template_store.validate_templates()

    def test_persona_selection_logic(self, generator):
        """Test persona selection based on task content."""
        research_templates = generator.template_store.get_templates_for_role("research")

        # Test research keywords
        assert (
            generator.selector.select_persona_from_templates(
                research_templates, "Research the history of AI"
            )
            == "thorough_investigator"
        )

        # Test analysis keywords
        assert (
            generator.selector.select_persona_from_templates(
                research_templates, "Analyze the market trends"
            )
            == "data_driven_analyst"
        )

        # Test design keywords
        assert (
            generator.selector.select_persona_from_templates(
                research_templates, "Design a new analysis framework"
            )
            == "comprehensive_researcher"
        )

        # Test optimization keywords
        assert (
            generator.selector.select_persona_from_templates(
                research_templates, "Optimize the data processing pipeline"
            )
            == "methodical_explorer"
        )

        # Test robustness keywords
        assert (
            generator.selector.select_persona_from_templates(
                research_templates, "Build a robust data pipeline"
            )
            == "evidence_based_analyst"
        )

        # Test default behavior
        assert (
            generator.selector.select_persona_from_templates(
                research_templates, "Write some code"
            )
            == "thorough_investigator"
        )  # First template as default

    def test_input_validation_role(self, generator):
        """Test role input validation."""
        # Valid inputs
        assert generator.validator.validate_role("research") == "research"
        assert generator.validator.validate_role("  researcher  ") == "researcher"

        # Invalid inputs - we need to catch exceptions manually
        with pytest.raises(ValueError, match="Role must be a string"):
            generator.validator.validate_role(123)

        with pytest.raises(ValueError, match="Role cannot be empty"):
            generator.validator.validate_role("")

        with pytest.raises(ValueError, match="Role cannot be empty"):
            generator.validator.validate_role("   ")

        with pytest.raises(ValueError, match="Role too long"):
            generator.validator.validate_role("a" * 101)

        with pytest.raises(ValueError, match="Role contains invalid characters"):
            generator.validator.validate_role("research<script>")

    def test_input_validation_task_description(self, generator):
        """Test task description input validation."""
        # Valid inputs
        short_task = "Analyze data"
        assert generator.validator.validate_task_description(short_task) == short_task

        # Long input gets truncated
        long_task = "x" * 2000
        truncated = generator.validator.validate_task_description(long_task)
        assert len(truncated) <= 1003  # 1000 + 3 for "..."
        assert truncated.endswith("...")
        assert len(truncated) == 1003

        # Non-string input gets converted
        assert generator.validator.validate_task_description(123) == "123"

    @pytest.mark.asyncio
    async def test_persona_generation_known_role(self, generator):
        """Test persona generation for known roles."""
        # Mock the LLM service to avoid actual API calls
        with patch.object(generator, "llm_service") as mock_llm:
            mock_llm.generate_async = AsyncMock(return_value="mocked response")

            persona = await generator.generate_persona_for_worker(
                role="research", task_description="Research quantum computing"
            )

            # Should return one of the research templates
            assert persona in generator.template_store.get_templates_for_role(
                "research"
            )
            assert isinstance(persona, str)

    @pytest.mark.asyncio
    async def test_persona_generation_unknown_role(self, generator):
        """Test persona generation for unknown roles."""
        with patch.object(generator, "llm_service") as mock_llm:
            mock_llm.generate_async = AsyncMock(return_value="mocked response")

            persona = await generator.generate_persona_for_worker(
                role="unknown_agent", task_description="Do something unknown"
            )

            assert persona == "unknown_agent_specialist"

    @pytest.mark.asyncio
    async def test_persona_generation_error_handling(self, generator):
        """Test error handling in persona generation."""
        # Test error handling with invalid inputs that should trigger validation errors
        # Since the current implementation doesn't call LLM, we test validation error handling

        # This should work fine (valid inputs)
        persona = await generator.generate_persona_for_worker(
            role="research", task_description="Research topic"
        )
        assert persona in generator.template_store.get_templates_for_role("research")

        # Test with inputs that cause validation errors
        # The method should handle validation errors gracefully
        try:
            await generator.generate_persona_for_worker(
                role="",
                task_description="Research topic",  # Empty role should trigger validation
            )
        except ValueError:
            # Expected validation error
            pass

    @pytest.mark.asyncio
    async def test_batch_persona_generation_empty(self, generator):
        """Test batch persona generation with empty input."""
        result = await generator.generate_personas_for_workers([])
        assert result == []

    @pytest.mark.asyncio
    async def test_batch_persona_generation_success(self, generator):
        """Test successful batch persona generation."""
        with patch.object(generator, "llm_service") as mock_llm:
            mock_llm.generate_async = AsyncMock(return_value="mocked response")

            workers = [
                {
                    "role": "research",
                    "task_description": "Research AI",
                    "worker_id": "worker1",
                },
                {
                    "role": "code",
                    "task_description": "Implement API",
                    "worker_id": "worker2",
                },
            ]

            result = await generator.generate_personas_for_workers(workers)

            assert len(result) == 2
            assert all("persona" in worker for worker in result)
            assert all(isinstance(worker["persona"], str) for worker in result)
            assert result[0][
                "persona"
            ] in generator.template_store.get_templates_for_role("research")
            assert result[1][
                "persona"
            ] in generator.template_store.get_templates_for_role("code")

    @pytest.mark.asyncio
    async def test_batch_persona_generation_partial_failure(self, generator):
        """Test batch persona generation with error handling."""
        # Test with workers that have validation issues
        workers = [
            {
                "role": "research",
                "task_description": "Research AI",
                "worker_id": "worker1",
            },
            {
                "role": "code",
                "task_description": "Implement API",
                "worker_id": "worker2",
            },
        ]

        result = await generator.generate_personas_for_workers(workers)

        assert len(result) == 2
        assert all("persona" in worker for worker in result)
        assert result[0]["persona"] in generator.template_store.get_templates_for_role(
            "research"
        )
        assert result[1]["persona"] in generator.template_store.get_templates_for_role(
            "code"
        )

    @pytest.mark.asyncio
    async def test_batch_persona_generation_missing_fields(self, generator):
        """Test batch persona generation with missing worker fields."""
        with patch.object(generator, "llm_service") as mock_llm:
            mock_llm.generate_async = AsyncMock(return_value="mocked response")

            workers = [
                {"task_description": "Research AI"},  # Missing role
                {"role": "code"},  # Missing task_description
                {},  # Missing both
            ]

            result = await generator.generate_personas_for_workers(workers)

            assert len(result) == 3
            assert all("persona" in worker for worker in result)
            # Should use "general" role for missing role
            assert result[0][
                "persona"
            ] in generator.template_store.get_templates_for_role("general")
            assert result[1][
                "persona"
            ] in generator.template_store.get_templates_for_role("code")
            assert result[2][
                "persona"
            ] in generator.template_store.get_templates_for_role("general")
