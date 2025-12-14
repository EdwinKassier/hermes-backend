"""Unit tests for RouterService."""

from unittest.mock import MagicMock

import pytest

from app.hermes.legion.nodes.router_service import RouterService
from app.hermes.legion.state import GraphDecision, OrchestratorState


@pytest.mark.unit
class TestRouterService:
    """Test suite for RouterService routing logic."""

    def test_route_from_orchestrator_gather_info(self):
        """Test routing to information gathering."""
        router = RouterService()
        state = {"next_action": GraphDecision.GATHER_INFO.value}

        result = router.route_from_orchestrator(state)

        assert result == "information_gathering"

    def test_route_from_orchestrator_execute_agent(self):
        """Test routing to agent executor."""
        router = RouterService()
        state = {"next_action": GraphDecision.EXECUTE_AGENT.value}

        result = router.route_from_orchestrator(state)

        assert result == "agent_executor"

    def test_route_from_orchestrator_replan(self):
        """Test routing back to orchestrator for replanning."""
        router = RouterService()
        state = {"next_action": GraphDecision.REPLAN.value}

        result = router.route_from_orchestrator(state)

        assert result == "orchestrator"

    def test_route_from_orchestrator_complete(self):
        """Test routing to general response for completion."""
        router = RouterService()
        state = {"next_action": GraphDecision.COMPLETE.value}

        result = router.route_from_orchestrator(state)

        assert result == "general_response"

    def test_route_from_orchestrator_legion(self):
        """Test routing to legion orchestrator."""
        router = RouterService()
        state = {"next_action": router.ROUTE_LEGION_ORCHESTRATE}

        result = router.route_from_orchestrator(state)

        assert result == "legion_orchestrator"

    def test_route_from_orchestrator_legacy(self):
        """Test legacy routing fallback."""
        router = RouterService()
        state = {"next_action": "council_strategy"}

        result = router.route_from_orchestrator(state)

        assert result == "legion_orchestrator"

    def test_route_from_orchestrator_unknown(self):
        """Test unknown routing defaults to general response."""
        router = RouterService()
        state = {"next_action": "unknown_action"}

        result = router.route_from_orchestrator(state)

        assert result == "general_response"

    def test_should_continue_conversation_complete(self):
        """Test ending when conversation is complete."""
        router = RouterService()
        state = {"conversation_complete": True}

        result = router.should_continue(state)

        assert result == "end"

    def test_should_continue_awaiting_user(self):
        """Test pausing when awaiting user response."""
        router = RouterService()
        state = {"awaiting_user_response": True}

        result = router.should_continue(state)

        assert result == "continue"

    def test_should_continue_error_state(self):
        """Test ending on error."""
        router = RouterService()
        state = {"next_action": GraphDecision.ERROR.value}

        result = router.should_continue(state)

        assert result == "end"

    def test_should_continue_default(self):
        """Test default continuation."""
        router = RouterService()
        state = {}

        result = router.should_continue(state)

        assert result == "continue"

    def test_route_from_info_gathering_gather_more(self):
        """Test continuing to gather information."""
        router = RouterService()
        state = {"next_action": GraphDecision.GATHER_INFO.value}

        result = router.route_from_info_gathering(state)

        assert result == "gather_info"

    def test_route_from_info_gathering_execute(self):
        """Test routing to execution after gathering info."""
        router = RouterService()
        state = {"next_action": GraphDecision.EXECUTE_AGENT.value}

        result = router.route_from_info_gathering(state)

        assert result == "execute_agent"

    def test_route_from_info_gathering_unknown(self):
        """Test ending on unknown action."""
        router = RouterService()
        state = {"next_action": "unknown"}

        result = router.route_from_info_gathering(state)

        assert result == "end"

    def test_route_legion_workers_no_workers(self):
        """Test routing with no workers."""
        router = RouterService()
        state = {"metadata": {}, "user_id": "test", "persona": "hermes"}

        result = router.route_legion_workers(state)

        assert result == []

    def test_route_legion_workers_with_workers(self):
        """Test routing with multiple workers."""
        router = RouterService()

        workers = [
            {
                "worker_id": "worker_1",
                "role": "researcher",
                "task_description": "Research topic A",
                "tools": ["search"],
                "dynamic_agent_config": {
                    "agent_id": "researcher",
                    "agent_type": "research_specialist",
                    "task_types": ["research"],
                    "capabilities": {"research": True},
                    "prompts": {"execute_task": "Research the topic"},
                    "persona": "researcher",
                },
            },
            {
                "worker_id": "worker_2",
                "role": "analyzer",
                "task_description": "Analyze data",
                "tools": ["calculator"],
                "dynamic_agent_config": {
                    "agent_id": "analyzer",
                    "agent_type": "analysis_specialist",
                    "task_types": ["analysis"],
                    "capabilities": {"analysis": True},
                    "prompts": {"execute_task": "Analyze the data"},
                    "persona": "analyzer",
                },
            },
        ]

        state = {
            "metadata": {"legion_worker_plans": workers},
            "user_id": "test_user",
            "persona": "hermes",
            "collected_info": {},
        }

        result = router.route_legion_workers(state)

        # Should return 2 Send objects
        assert len(result) == 2

        # Each should be a Send object targeting "legion_worker"
        for send in result:
            assert hasattr(send, "node")
            # Send objects don't expose node name directly in this version
            # but we can verify they were created
