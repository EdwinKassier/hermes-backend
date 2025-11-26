"""Unit tests for ParallelTaskDecomposer."""

from unittest.mock import MagicMock, patch

import pytest

from app.hermes.legion.parallel.task_decomposer import ParallelTaskDecomposer


class TestParallelTaskDecomposer:
    """Test suite for task decomposition logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.decomposer = ParallelTaskDecomposer()

    def test_is_multi_agent_task_with_compound_query(self):
        """Test detection of multi-agent tasks with 'and' conjunction."""
        query = "Research quantum computing and analyze its applications"
        result = self.decomposer.is_multi_agent_task(query)
        assert result is True

    def test_is_multi_agent_task_with_multiple_verbs(self):
        """Test detection based on multiple action verbs."""
        query = "Find data sources, analyze trends, write report"
        result = self.decomposer.is_multi_agent_task(query)
        assert result is True

    def test_is_multi_agent_task_single_task(self):
        """Test that single tasks are not detected as multi-agent."""
        query = "Write Python code to sort a list"
        result = self.decomposer.is_multi_agent_task(query)
        assert result is False

    def test_is_multi_agent_task_simple_question(self):
        """Test that simple questions are not multi-agent."""
        query = "What is machine learning?"
        result = self.decomposer.is_multi_agent_task(query)
        assert result is False

    @patch("app.hermes.legion.parallel.task_decomposer.get_gemini_service")
    def test_decompose_task_success(self, mock_get_gemini):
        """Test successful task decomposition."""
        # Mock Gemini response
        mock_service = MagicMock()
        mock_service.generate_gemini_response.return_value = """
        {
            "subtasks": [
                {
                    "description": "Research quantum computing",
                    "agent_type": "research",
                    "keywords": ["quantum", "computing", "research"]
                },
                {
                    "description": "Analyze applications",
                    "agent_type": "analysis",
                    "keywords": ["analyze", "applications"]
                }
            ]
        }
        """
        mock_get_gemini.return_value = mock_service

        query = "Research quantum computing and analyze its applications"
        result = self.decomposer.decompose_task(query, skip_check=True)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["agent_type"] == "research"
        assert result[1]["agent_type"] == "analysis"

    @patch("app.hermes.legion.parallel.task_decomposer.get_gemini_service")
    def test_decompose_task_fallback(self, mock_get_gemini):
        """Test fallback when AI decomposition fails."""
        # Mock Gemini to raise exception
        mock_service = MagicMock()
        mock_service.generate_gemini_response.side_effect = Exception("API Error")
        mock_get_gemini.return_value = mock_service

        query = "Research X and analyze Y"
        result = self.decomposer.decompose_task(query, skip_check=True)

        # Should return None when AI fails
        assert result is None

    def test_empty_query(self):
        """Test handling of empty query."""
        result = self.decomposer.is_multi_agent_task("")
        assert result is False

    def test_none_query(self):
        """Test handling of None query."""
        result = self.decomposer.is_multi_agent_task(None)
        assert result is False


class TestTaskDecompositionEdgeCases:
    """Test edge cases in task decomposition."""

    def setup_method(self):
        """Set up test fixtures."""
        self.decomposer = ParallelTaskDecomposer()

    def test_very_long_query(self):
        """Test handling of very long queries."""
        query = "Research " + " and ".join([f"topic_{i}" for i in range(10)])
        result = self.decomposer.is_multi_agent_task(query)
        assert isinstance(result, bool)

    def test_query_with_special_characters(self):
        """Test queries with special characters."""
        query = "Research AI & ML, analyze data, write code!"
        result = self.decomposer.is_multi_agent_task(query)
        assert isinstance(result, bool)

    def test_ambiguous_query(self):
        """Test handling of ambiguous queries."""
        query = "Do research and stuff"
        result = self.decomposer.is_multi_agent_task(query)
        assert isinstance(result, bool)
