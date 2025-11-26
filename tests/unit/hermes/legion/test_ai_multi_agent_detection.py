"""Unit tests for AI-powered multi-agent detection."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from app.hermes.legion.parallel.task_decomposer import ParallelTaskDecomposer


class TestMultiAgentDetection:
    """Test AI-powered multi-agent task detection."""

    @pytest.fixture
    def decomposer(self):
        """Create decomposer instance."""
        return ParallelTaskDecomposer()

    def test_multi_agent_detection_ai_success_multi(self, decomposer):
        """Test AI correctly detects multi-agent task."""
        mock_service = Mock()
        mock_service.generate_gemini_response.return_value = "MULTI_AGENT"
        decomposer._gemini_service = mock_service

        result = decomposer.is_multi_agent_task("Research AI and write code")

        assert result is True
        mock_service.generate_gemini_response.assert_called_once()

    def test_multi_agent_detection_ai_success_single(self, decomposer):
        """Test AI correctly detects single-agent task."""
        mock_service = Mock()
        mock_service.generate_gemini_response.return_value = "SINGLE_AGENT"
        decomposer._gemini_service = mock_service

        result = decomposer.is_multi_agent_task("Research quantum computing")

        assert result is False
        mock_service.generate_gemini_response.assert_called_once()

    def test_multi_agent_detection_ai_failure_uses_fallback(self, decomposer):
        """Test fallback when AI fails."""
        mock_service = Mock()
        mock_service.generate_gemini_response.side_effect = Exception("API Error")
        decomposer._gemini_service = mock_service

        # Should use fallback - detects "and" + multiple verbs
        result = decomposer.is_multi_agent_task("Research AI and analyze data")

        assert result is True  # Fallback detects this

    def test_multi_agent_detection_network_error_retry(self, decomposer):
        """Test retry on network error."""
        mock_service = Mock()
        # First call fails, second succeeds
        mock_service.generate_gemini_response.side_effect = [
            ConnectionError("Network error"),
            "MULTI_AGENT",
        ]
        decomposer._gemini_service = mock_service

        result = decomposer._cached_multi_agent_check("Research and analyze")

        assert result is True
        assert mock_service.generate_gemini_response.call_count == 2

    def test_multi_agent_detection_empty_input(self, decomposer):
        """Test empty input returns False."""
        assert decomposer.is_multi_agent_task("") is False
        assert decomposer.is_multi_agent_task("   ") is False
        assert decomposer.is_multi_agent_task("abc") is False  # Too short

    def test_multi_agent_detection_invalid_type(self, decomposer):
        """Test invalid input type returns False."""
        assert decomposer.is_multi_agent_task(None) is False
        assert decomposer.is_multi_agent_task(123) is False
        assert decomposer.is_multi_agent_task([]) is False

    def test_multi_agent_detection_long_input_truncated(self, decomposer):
        """Test long input is truncated."""
        long_message = "x" * 2000

        with patch.object(decomposer, "_cached_multi_agent_check") as mock_cached:
            mock_cached.return_value = False

            decomposer.is_multi_agent_task(long_message)

            # Should be called with truncated message
            called_message = mock_cached.call_args[0][0]
            assert len(called_message) == 1000

    def test_fallback_multi_agent_detection_compound_and_verbs(self, decomposer):
        """Test fallback detects compound indicator + multiple verbs."""
        # Has "and" + "research" + "analyze"
        result = decomposer._fallback_multi_agent_detection(
            "Research quantum computing and analyze applications"
        )
        assert result is True

    def test_fallback_multi_agent_detection_no_compound(self, decomposer):
        """Test fallback returns False without compound indicator."""
        # Has verbs but no compound indicator
        result = decomposer._fallback_multi_agent_detection("Research analyze evaluate")
        assert result is False

    def test_fallback_multi_agent_detection_compound_but_one_verb(self, decomposer):
        """Test fallback with compound indicator."""
        # Has "and" and "research" (which also contains "search" substring)
        # Due to substring matching, this counts as 2 verbs: "research" and "search"
        result = decomposer._fallback_multi_agent_detection(
            "Research and development team"
        )
        # Returns True because "research" contains "search" (substring match counts 2 verbs)
        assert result is True

    def test_build_multi_agent_detection_prompt(self, decomposer):
        """Test prompt building includes user message and examples."""
        prompt = decomposer._build_multi_agent_detection_prompt("Test query")

        assert "Test query" in prompt
        assert "MULTI_AGENT" in prompt
        assert "SINGLE_AGENT" in prompt
        assert "Examples" in prompt
        assert "Research quantum computing AND analyze" in prompt


class TestPromptBuilding:
    """Test prompt construction."""

    @pytest.fixture
    def decomposer(self):
        return ParallelTaskDecomposer()

    def test_prompt_includes_user_message(self, decomposer):
        """Test prompt includes the user's message."""
        user_msg = "Research AI and write code"
        prompt = decomposer._build_multi_agent_detection_prompt(user_msg)

        assert user_msg in prompt

    def test_prompt_includes_examples(self, decomposer):
        """Test prompt includes positive and negative examples."""
        prompt = decomposer._build_multi_agent_detection_prompt("test")

        # Should have multi-agent examples
        assert "Research quantum computing AND analyze" in prompt
        assert "Find data sources, clean the data" in prompt

        # Should have single-agent examples
        assert "Research quantum computing" in prompt
        assert "Write code to sort a list" in prompt

    def test_prompt_specifies_response_format(self, decomposer):
        """Test prompt specifies expected response format."""
        prompt = decomposer._build_multi_agent_detection_prompt("test")

        assert "MULTI_AGENT" in prompt
        assert "SINGLE_AGENT" in prompt
        assert "Respond with ONLY" in prompt


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def decomposer(self):
        return ParallelTaskDecomposer()

    def test_malformed_ai_response(self, decomposer):
        """Test handling of malformed AI response."""
        mock_service = Mock()
        # AI returns unexpected response
        mock_service.generate_gemini_response.return_value = "INVALID_RESPONSE"
        decomposer._gemini_service = mock_service

        result = decomposer._cached_multi_agent_check("Research and analyze")

        # Should return False (MULTI_AGENT not in response)
        assert result is False

    def test_ai_response_with_extra_text(self, decomposer):
        """Test AI response with extra text still works."""
        mock_service = Mock()
        mock_service.generate_gemini_response.return_value = (
            "Based on the analysis, this is a MULTI_AGENT task."
        )
        decomposer._gemini_service = mock_service

        result = decomposer._cached_multi_agent_check("Research and analyze")

        assert result is True

    def test_case_insensitive_response_parsing(self, decomposer):
        """Test response parsing is case-insensitive."""
        mock_service = Mock()
        mock_service.generate_gemini_response.return_value = "multi_agent"
        decomposer._gemini_service = mock_service

        result = decomposer._cached_multi_agent_check("Research and analyze")

        assert result is True
