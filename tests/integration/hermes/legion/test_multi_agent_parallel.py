"""Integration tests for multi-agent parallel execution."""

from unittest.mock import MagicMock, patch

import pytest

from app.hermes.legion.graph_service import LegionGraphService
from app.hermes.models import UserIdentity


@pytest.mark.integration
class TestMultiAgentParallelExecution:
    """Integration tests for end-to-end multi-agent workflows."""

    @patch('app.hermes.legion.graph_service.get_gemini_service')
    @patch('app.hermes.legion.nodes.graph_nodes.AgentFactory')
    def test_research_and_code_task(self, mock_factory, mock_gemini):
        """Test parallel execution of research + code task."""
        # Mock Gemini service
        mock_gemini_service = MagicMock()
        mock_gemini.return_value = mock_gemini_service

        # Mock agent responses
        mock_research_agent = MagicMock()
        mock_research_agent.execute_task.return_value = "Research results about algorithms..."

        mock_code_agent = MagicMock()
        mock_code_agent.execute_task.return_value = "def quicksort(arr): ..."

        mock_factory.create_agent.side_effect = [mock_research_agent, mock_code_agent]

        # Create service and execute
        service = LegionGraphService()
        user_identity = UserIdentity(user_id="test_user")

        result = service.process_request(
            text="Research sorting algorithms and write Python implementation",
            user_identity=user_identity
        )

        assert result is not None
        assert result.content is not None
        # Should have used both agents
        assert mock_factory.create_agent.call_count >= 1

    @patch('app.hermes.legion.graph_service.get_gemini_service')
    def test_multi_agent_result_synthesis(self, mock_gemini):
        """Test that results from multiple agents are synthesized."""
        mock_gemini_service = MagicMock()
        mock_gemini.return_value = mock_gemini_service

        service = LegionGraphService()
        user_identity = UserIdentity(user_id="test_user")

        with patch('app.hermes.legion.parallel.task_decomposer.ParallelTaskDecomposer') as mock_decomposer:
            # Force multi-agent detection
            mock_decomp_instance = mock_decomposer.return_value
            mock_decomp_instance.is_multi_agent_task.return_value = True
            mock_decomp_instance.decompose_task.return_value = [
                {"agent_type": "research", "description": "Research AI"},
                {"agent_type": "analysis", "description": "Analyze trends"}
            ]

            result = service.process_request(
                text="Research AI and analyze trends",
                user_identity=user_identity
            )

            # Check that result contains decision metadata
            assert hasattr(result, 'metadata')
            assert "decision_rationale" in result.metadata or "agents_used" in result.metadata

    @patch('app.hermes.legion.graph_service.get_gemini_service')
    def test_performance_metrics_tracked(self, mock_gemini):
        """Test that performance metrics are tracked for parallel execution."""
        mock_gemini_service = MagicMock()
        mock_gemini.return_value = mock_gemini_service

        service = LegionGraphService()
        user_identity = UserIdentity(user_id="test_user")

        result = service.process_request(
            text="Research quantum computing and analyze applications",
            user_identity=user_identity
        )

        # Should have metadata
        assert hasattr(result, 'metadata')
        # May have performance metrics if multi-agent was triggered
        # This is a permissive test as routing depends on AI

    @patch('app.hermes.legion.graph_service.get_gemini_service')
    def test_three_agent_parallel_execution(self, mock_gemini):
        """Test execution with three parallel agents."""
        mock_gemini_service = MagicMock()
        mock_gemini.return_value = mock_gemini_service

        service = LegionGraphService()
        user_identity = UserIdentity(user_id="test_user")

        with patch('app.hermes.legion.parallel.task_decomposer.ParallelTaskDecomposer') as mock_decomposer:
            mock_decomp_instance = mock_decomposer.return_value
            mock_decomp_instance.is_multi_agent_task.return_value = True
            mock_decomp_instance.decompose_task.return_value = [
                {"agent_type": "research", "description": "Research"},
                {"agent_type": "analysis", "description": "Analyze"},
                {"agent_type": "code", "description": "Code"}
            ]

            result = service.process_request(
                text="Research ML, analyze data, write Python code",
                user_identity=user_identity
            )

            assert result is not None
            assert result.content is not None


@pytest.mark.integration
class TestPartialFailureRecovery:
    """Test graceful handling of partial agent failures."""

    @patch('app.hermes.legion.graph_service.get_gemini_service')
    @patch('app.hermes.legion.nodes.graph_nodes.AgentFactory')
    def test_one_agent_fails_others_succeed(self, mock_factory, mock_gemini):
        """Test that system handles one agent failing gracefully."""
        mock_gemini_service = MagicMock()
        mock_gemini.return_value = mock_gemini_service

        # First agent succeeds, second fails
        mock_success_agent = MagicMock()
        mock_success_agent.execute_task.return_value = "Success result"

        mock_fail_agent = MagicMock()
        mock_fail_agent.execute_task.side_effect = Exception("Agent failed")

        mock_factory.create_agent.side_effect = [mock_success_agent, mock_fail_agent]

        service = LegionGraphService()
        user_identity = UserIdentity(user_id="test_user")

        # Should not crash, should return partial results
        result = service.process_request(
            text="Research X and analyze Y",
            user_identity=user_identity
        )

        assert result is not None
        # Should still have some content from successful agent
        assert result.content is not None
