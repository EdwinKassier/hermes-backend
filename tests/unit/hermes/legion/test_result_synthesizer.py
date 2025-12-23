"""Unit tests for ResultSynthesizer."""

from unittest.mock import MagicMock, patch

import pytest

from app.hermes.legion.parallel.result_synthesizer import ResultSynthesizer


class TestResultSynthesizer:
    """Test suite for result synthesis logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.synthesizer = ResultSynthesizer()

    @patch("app.hermes.legion.parallel.result_synthesizer.get_llm_service")
    def test_synthesize_results_with_two_agents(self, mock_get_llm):
        """Test synthesis of results from two agents."""
        mock_service = MagicMock()
        mock_service.generate_response.return_value = (
            "Based on the research on quantum computing and analysis of applications..."
        )
        mock_get_llm.return_value = mock_service

        results = {
            "research_1": {
                "agent_type": "research",
                "result": "Quantum computing uses qubits...",
                "task_description": "Research quantum computing",
                "status": "success",
            },
            "analysis_1": {
                "agent_type": "analysis",
                "result": "Applications include cryptography...",
                "task_description": "Analyze applications",
                "status": "success",
            },
        }
        query = "Research quantum computing and analyze applications"

        result = self.synthesizer.synthesize_results(query, results)

        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain elements from both responses
        assert "quantum" in result.lower() or "qubits" in result.lower()

    def test_synthesize_single_result_no_synthesis_needed(self):
        """Test that single result doesn't need synthesis."""
        results = {
            "code_1": {
                "agent_type": "code",
                "result": "Here's the Python code...",
                "task_description": "Write Python code",
                "status": "success",
            }
        }
        query = "Write Python code"

        result = self.synthesizer.synthesize_results(query, results)

        # Should return the single response directly
        assert "Here's the Python code" in result

    def test_synthesize_empty_results(self):
        """Test handling of empty results list."""
        results = {}
        query = "Some query"

        result = self.synthesizer.synthesize_results(query, results)

        assert isinstance(result, str)
        # Should have some fallback message
        assert len(result) > 0

    @patch("app.hermes.legion.parallel.result_synthesizer.get_llm_service")
    def test_synthesize_with_failed_agent(self, mock_get_llm):
        """Test synthesis when one agent failed."""
        mock_service = MagicMock()
        mock_service.generate_response.return_value = "Based on the research results..."
        mock_get_llm.return_value = mock_service

        results = {
            "research_1": {
                "agent_type": "research",
                "result": "Research results here...",
                "task_description": "Research task",
                "status": "success",
            },
            "analysis_1": {
                "agent_type": "analysis",
                "result": "",
                "task_description": "Analysis task",
                "status": "failed",
                "error": "API timeout",
            },
        }
        query = "Research and analyze"

        result = self.synthesizer.synthesize_results(query, results)

        assert isinstance(result, str)
        # Should still produce output from successful agent
        assert "research" in result.lower()

    @patch("app.hermes.legion.parallel.result_synthesizer.get_llm_service")
    def test_synthesize_results_success(self, mock_get_llm):
        """Test successful result synthesis."""
        mock_service = MagicMock()
        mock_get_llm.return_value = mock_service
        mock_service.generate_response.return_value = "Combined synthesis..."

        results = {
            "agent1": {
                "agent_type": "research",
                "result": "Result 1",
                "task_description": "Task 1",
                "status": "success",
            },
            "agent2": {
                "agent_type": "analysis",
                "result": "Result 2",
                "task_description": "Task 2",
                "status": "success",
            },
        }

        result = self.synthesizer.synthesize_results("test query", results)

        # Should have called LLM service
        mock_service.generate_response.assert_called_once()
        assert result == "Combined synthesis..."

    @patch("app.hermes.legion.parallel.result_synthesizer.get_llm_service")
    def test_synthesize_uses_ai_when_multiple_results(self, mock_get_llm):
        """Test that AI synthesis is used for multiple results."""
        mock_service = MagicMock()
        mock_service.generate_response.return_value = "Combined synthesis..."
        mock_get_llm.return_value = mock_service

        results = {
            "agent1": {
                "agent_type": "research",
                "result": "Result 1",
                "task_description": "Task 1",
                "status": "success",
            },
            "agent2": {
                "agent_type": "analysis",
                "result": "Result 2",
                "task_description": "Task 2",
                "status": "success",
            },
        }

        result = self.synthesizer.synthesize_results("test query", results)

        # Should have called AI service
        mock_service.generate_response.assert_called_once()

    @patch("app.hermes.legion.parallel.result_synthesizer.get_llm_service")
    def test_fallback_concatenation_on_ai_failure(self, mock_get_llm):
        """Test fallback to concatenation when AI fails."""
        mock_service = MagicMock()
        mock_service.generate_response.side_effect = Exception("AI Error")
        mock_get_llm.return_value = mock_service

        results = {
            "agent1": {
                "agent_type": "research",
                "result": "Result A",
                "task_description": "Task A",
                "status": "success",
            },
            "agent2": {
                "agent_type": "analysis",
                "result": "Result B",
                "task_description": "Task B",
                "status": "success",
            },
        }

        result = self.synthesizer.synthesize_results("test", results)

        # Should fallback to concatenation
        assert "Result A" in result
        assert "Result B" in result

    def test_generate_clarifying_questions(self):
        """Test generation of clarifying questions from gaps."""
        gaps = [
            "Missing result from code agent",
            "Incomplete result from research agent",
        ]
        partial_results = {
            "research_1": {
                "agent_type": "research",
                "result": "Partial info",
                "task_description": "Research",
                "status": "success",
            }
        }

        questions = self.synthesizer.generate_clarifying_questions(
            gaps, "test query", partial_results
        )

        assert isinstance(questions, list)

    def test_detect_information_gaps(self):
        """Test detection of information gaps in results."""
        results = {
            "agent1": {
                "agent_type": "research",
                "result": "Partial information...",
                "task_description": "Task",
                "status": "success",
            }
        }
        expected_agents = ["agent1", "agent2"]

        gaps = self.synthesizer.detect_gaps(results, expected_agents)

        # Should detect missing agent2
        assert isinstance(gaps, list)
        assert len(gaps) > 0


class TestSynthesisQuality:
    """Test quality of synthesis output."""

    def setup_method(self):
        """Set up test fixtures."""
        self.synthesizer = ResultSynthesizer()

    @patch("app.hermes.legion.parallel.result_synthesizer.get_llm_service")
    def test_synthesis_is_coherent(self, mock_get_llm):
        """Test that synthesis produces coherent output."""
        mock_service = MagicMock()
        mock_service.generate_response.return_value = """
        Based on the research and analysis, quantum computing represents
        a paradigm shift in computation with applications in cryptography.
        """
        mock_get_llm.return_value = mock_service

        results = {
            "agent1": {
                "agent_type": "research",
                "result": "Quantum info",
                "task_description": "Research",
                "status": "success",
            },
            "agent2": {
                "agent_type": "analysis",
                "result": "Crypto applications",
                "task_description": "Analysis",
                "status": "success",
            },
        }

        result = self.synthesizer.synthesize_results("test", results)

        # Should be a reasonably long, coherent response
        assert len(result) > 50
        assert not result.startswith("Error")
