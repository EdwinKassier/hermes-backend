"""Unit tests for LegionPersonaGenerator."""

import asyncio
from unittest.mock import AsyncMock, patch

from app.hermes.legion.utils.persona_generator import LegionPersonaGenerator


def run_async_test(coro):
    """Helper to run async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestLegionPersonaGenerator:
    """Test cases for LegionPersonaGenerator."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = LegionPersonaGenerator()

    def test_role_normalization(self):
        """Test role normalization logic."""
        # Test common role mappings
        assert self.generator._normalize_role("researcher") == "research"
        assert self.generator._normalize_role("Coder") == "code"
        assert self.generator._normalize_role("PROGRAMMER") == "code"
        assert self.generator._normalize_role("analyst") == "analysis"
        assert self.generator._normalize_role("data_analyst") == "data"
        assert self.generator._normalize_role("scientist") == "research"
        assert self.generator._normalize_role("engineer") == "code"
        assert self.generator._normalize_role("architect") == "code"
        assert self.generator._normalize_role("investigator") == "research"
        assert self.generator._normalize_role("evaluator") == "analysis"
        assert self.generator._normalize_role("interpreter") == "analysis"
        assert self.generator._normalize_role("specialist") == "general"
        assert self.generator._normalize_role("expert") == "general"

        # Test unknown roles pass through
        assert self.generator._normalize_role("unknown_role") == "unknown_role"
        assert self.generator._normalize_role("custom_agent") == "custom_agent"

    def test_persona_template_structure(self):
        """Test that persona templates have correct structure."""
        templates = self.generator.role_persona_templates

        # Check expected roles exist
        expected_roles = {"research", "code", "analysis", "data", "general"}
        assert set(templates.keys()) == expected_roles

        # Check each role has templates
        for role, role_templates in templates.items():
            assert isinstance(role_templates, list)
            assert len(role_templates) == 5  # 5 templates per role
            assert all(isinstance(template, str) for template in role_templates)

        # Check all templates are unique
        all_templates = []
        for role_templates in templates.values():
            all_templates.extend(role_templates)
        assert len(all_templates) == len(set(all_templates))

    def test_persona_selection_logic(self):
        """Test persona selection based on task content."""
        research_templates = self.generator.role_persona_templates["research"]

        # Test research keywords
        assert (
            self.generator._select_persona_from_templates(
                research_templates, "Research the history of AI"
            )
            == "thorough_investigator"
        )

        # Test analysis keywords
        assert (
            self.generator._select_persona_from_templates(
                research_templates, "Analyze the market trends"
            )
            == "data_driven_analyst"
        )

        # Test design keywords
        assert (
            self.generator._select_persona_from_templates(
                research_templates, "Design a new research methodology"
            )
            == "comprehensive_researcher"
        )

        # Test optimization keywords
        assert (
            self.generator._select_persona_from_templates(
                research_templates, "Optimize the research process"
            )
            == "methodical_explorer"
        )

        # Test robustness keywords
        assert (
            self.generator._select_persona_from_templates(
                research_templates, "Build a robust research framework"
            )
            == "evidence_based_analyst"
        )

        # Test default behavior
        assert (
            self.generator._select_persona_from_templates(
                research_templates, "Write some code"
            )
            == "thorough_investigator"
        )  # First template as default

    def test_input_validation_role(self):
        """Test role input validation."""
        # Valid inputs
        assert self.generator._validate_role("research") == "research"
        assert self.generator._validate_role("  researcher  ") == "researcher"

        # Invalid inputs - we need to catch exceptions manually
        try:
            self.generator._validate_role(123)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Role must be a string" in str(e)

        try:
            self.generator._validate_role("")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Role cannot be empty" in str(e)

        try:
            self.generator._validate_role("   ")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Role cannot be empty" in str(e)

        try:
            self.generator._validate_role("a" * 101)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Role too long" in str(e)

        try:
            self.generator._validate_role("research<script>")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Role contains invalid characters" in str(e)

    def test_input_validation_task_description(self):
        """Test task description input validation."""
        # Valid inputs
        short_task = "Analyze data"
        assert self.generator._validate_task_description(short_task) == short_task

        # Long input gets truncated
        long_task = "x" * 2000
        truncated = self.generator._validate_task_description(long_task)
        assert len(truncated) <= 1000
        assert truncated.endswith("...")

        # Non-string input gets converted
        assert self.generator._validate_task_description(123) == "123"

    def test_persona_generation_known_role(self):
        """Test persona generation for known roles."""
        # Mock the LLM service to avoid actual API calls
        with patch.object(self.generator, "llm_service") as mock_llm:
            mock_llm.generate_async = AsyncMock(return_value="mocked response")

            persona = run_async_test(
                self.generator.generate_persona_for_worker(
                    role="research", task_description="Research quantum computing"
                )
            )

            # Should return one of the research templates
            assert persona in self.generator.role_persona_templates["research"]
            assert isinstance(persona, str)

    def test_persona_generation_unknown_role(self):
        """Test persona generation for unknown roles."""
        with patch.object(self.generator, "llm_service") as mock_llm:
            mock_llm.generate_async = AsyncMock(return_value="mocked response")

            persona = run_async_test(
                self.generator.generate_persona_for_worker(
                    role="unknown_agent", task_description="Do something unknown"
                )
            )

            assert persona == "unknown_agent_specialist"

    def test_persona_generation_error_handling(self):
        """Test error handling in persona generation."""
        # Mock the LLM service to raise an exception
        with patch.object(self.generator, "llm_service") as mock_llm:
            mock_llm.generate_async = AsyncMock(side_effect=Exception("LLM Error"))

            persona = run_async_test(
                self.generator.generate_persona_for_worker(
                    role="research", task_description="Research topic"
                )
            )

            # Should return fallback persona
            assert persona == "research_specialist"

    def test_batch_persona_generation_empty(self):
        """Test batch persona generation with empty input."""
        result = run_async_test(self.generator.generate_personas_for_workers([]))
        assert result == []

    def test_batch_persona_generation_success(self):
        """Test successful batch persona generation."""
        with patch.object(self.generator, "llm_service") as mock_llm:
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

            result = run_async_test(
                self.generator.generate_personas_for_workers(workers)
            )

            assert len(result) == 2
            assert all("persona" in worker for worker in result)
            assert all(isinstance(worker["persona"], str) for worker in result)
            assert (
                result[0]["persona"]
                in self.generator.role_persona_templates["research"]
            )
            assert result[1]["persona"] in self.generator.role_persona_templates["code"]

    def test_batch_persona_generation_partial_failure(self):
        """Test batch persona generation with partial failures."""
        # Mock to fail on first call, succeed on second
        with patch.object(self.generator, "llm_service") as mock_llm:
            mock_llm.generate_async = AsyncMock(
                side_effect=[Exception("First call fails"), "mocked response"]
            )

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

            result = run_async_test(
                self.generator.generate_personas_for_workers(workers)
            )

            assert len(result) == 2
            assert result[0]["persona"] == "research_specialist"  # Fallback
            assert (
                result[1]["persona"] in self.generator.role_persona_templates["code"]
            )  # Success

    def test_batch_persona_generation_missing_fields(self):
        """Test batch persona generation with missing worker fields."""
        with patch.object(self.generator, "llm_service") as mock_llm:
            mock_llm.generate_async = AsyncMock(return_value="mocked response")

            workers = [
                {"task_description": "Research AI"},  # Missing role
                {"role": "code"},  # Missing task_description
                {},  # Missing both
            ]

            result = run_async_test(
                self.generator.generate_personas_for_workers(workers)
            )

            assert len(result) == 3
            assert all("persona" in worker for worker in result)
            # Should use "general" role for missing role
            assert (
                result[0]["persona"] in self.generator.role_persona_templates["general"]
            )
            assert result[1]["persona"] in self.generator.role_persona_templates["code"]
            assert (
                result[2]["persona"] in self.generator.role_persona_templates["general"]
            )
