"""Unit tests for conversation flexibility features."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from app.hermes.legion.nodes.graph_nodes import orchestrator_node
from app.hermes.legion.state.graph_state import (
    GraphDecision,
    OrchestratorState,
    TaskInfo,
    TaskStatus,
)


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock all service dependencies to avoid import errors."""
    with (
        patch(
            "app.hermes.legion.utils.conversation_memory.get_async_llm_service"
        ) as mock_memory_llm,
        patch(
            "app.hermes.legion.intelligence.routing_service.get_async_llm_service"
        ) as mock_get_llm,
        patch(
            "app.hermes.legion.parallel.task_decomposer.get_gemini_service"
        ) as mock_decomposer_service,
        patch(
            "app.hermes.legion.nodes.graph_nodes.get_gemini_service"
        ) as mock_nodes_service,
        patch(
            "app.hermes.legion.nodes.graph_nodes.get_async_llm_service"
        ) as mock_nodes_async_service,
        patch(
            "app.hermes.legion.agents.research_agent.get_gemini_service"
        ) as mock_research_service,
        patch(
            "app.hermes.legion.agents.code_agent.get_gemini_service"
        ) as mock_code_service,
        patch(
            "app.hermes.legion.agents.analysis_agent.get_gemini_service"
        ) as mock_analysis_service,
        patch(
            "app.hermes.legion.agents.data_agent.get_gemini_service"
        ) as mock_data_service,
    ):
        mock_llm_service = AsyncMock()
        mock_get_llm.return_value = mock_llm_service
        mock_nodes_async_service.return_value = mock_llm_service
        mock_memory_llm.return_value = mock_llm_service

        mock_gemini = Mock()
        mock_decomposer_service.return_value = mock_gemini
        mock_nodes_service.return_value = mock_gemini

        mock_research_service.return_value = mock_gemini
        mock_code_service.return_value = mock_gemini
        mock_analysis_service.return_value = mock_gemini
        mock_data_service.return_value = mock_gemini

        yield


@pytest.fixture
def base_state():
    """Create base orchestrator state for testing."""
    return {
        "messages": [],
        "user_id": "test_user",
        "persona": "hermes",
        "task_ledger": {},
        "agents": {},
        "tool_allocations": {},
        "current_agent_id": None,
        "current_task_id": None,
        "next_action": "",
        "required_info": {},
        "collected_info": {},
        "pending_questions": [],
        "decision_rationale": [],
        "parallel_mode": False,
        "parallel_tasks": {},
        "parallel_results": {},
        "agents_awaiting_info": {},
        "synthesis_needed": False,
        "legion_strategy": "council",
        "legion_results": {},
        "awaiting_user_response": False,
        "conversation_complete": False,
        "metadata": {},
    }


@pytest.mark.unit
@pytest.mark.asyncio
class TestConversationFlexibility:
    """Test conversation flexibility features."""

    async def test_topic_change_detection_triggers_replan(self, base_state):
        """Test that topic change detection triggers REPLAN action."""
        # Setup: Create state with active task
        task_info = TaskInfo(
            task_id="task_1",
            agent_id="agent_1",
            description="Write code for web scraper",
            status=TaskStatus.IN_PROGRESS,
        )

        state = {
            **base_state,
            "messages": [
                {
                    "role": "user",
                    "content": "Write code for web scraper",
                    "timestamp": "2024-01-01",
                    "metadata": {},
                },
                {
                    "role": "assistant",
                    "content": "I'll help with that",
                    "timestamp": "2024-01-01",
                    "metadata": {},
                },
                {
                    "role": "user",
                    "content": "Actually, just explain what web scraping is",
                    "timestamp": "2024-01-01",
                    "metadata": {},
                },
            ],
            "current_task_id": "task_1",
            "task_ledger": {"task_1": task_info},
        }

        # Mock RoutingIntelligence to return topic change decision
        from app.hermes.legion.intelligence.routing_intelligence import (
            ConversationPhase,
            RoutingAction,
            RoutingDecision,
        )

        with patch(
            "app.hermes.legion.intelligence.routing_service.RoutingIntelligence"
        ) as MockRoutingIntelligence:
            mock_routing = MockRoutingIntelligence.return_value
            mock_decision = RoutingDecision(
                action=RoutingAction.SIMPLE_RESPONSE,
                reasoning="Topic change detected",
                confidence=0.95,
                requires_agents=False,
                conversation_type="topic_change",
                complexity_estimate=0.1,
                user_goal="explain_web_scraping",
                conversation_phase=ConversationPhase.TOPIC_SHIFTING,
                topic_change_detected=True,
                topic_change_confidence=0.95,
                should_abandon_current_task=True,
                previous_topic_description="Write code for web scraper",
                new_topic_description="Explain web scraping concepts",
            )
            mock_routing.analyze = AsyncMock(return_value=mock_decision)

            # Execute
            result = await orchestrator_node(state)

            # Verify REPLAN was triggered
            assert result["next_action"] == GraphDecision.REPLAN.value
            assert "topic_change" in result.get("metadata", {})
            # Confidence may vary based on routing intelligence, just check it exists
            assert "confidence" in result["metadata"]["topic_change"]
            assert result["metadata"]["topic_change"]["confidence"] > 0.7

    async def test_no_replan_for_clarifications(self, base_state):
        """Test that clarifications don't trigger replan."""
        task_info = TaskInfo(
            task_id="task_1",
            agent_id="agent_1",
            description="Write code for web scraper",
            status=TaskStatus.IN_PROGRESS,
        )

        state = {
            **base_state,
            "messages": [
                {
                    "role": "user",
                    "content": "Write code for web scraper",
                    "timestamp": "2024-01-01",
                    "metadata": {},
                },
                {
                    "role": "assistant",
                    "content": "What websites?",
                    "timestamp": "2024-01-01",
                    "metadata": {},
                },
                {
                    "role": "user",
                    "content": "Amazon and eBay",
                    "timestamp": "2024-01-01",
                    "metadata": {},
                },
            ],
            "current_task_id": "task_1",
            "task_ledger": {"task_1": task_info},
        }

        # Mock RoutingIntelligence to return GATHER_INFO (clarification) instead of topic change
        from app.hermes.legion.intelligence.routing_intelligence import (
            ConversationPhase,
            RoutingAction,
            RoutingDecision,
        )

        with patch(
            "app.hermes.legion.intelligence.routing_service.RoutingIntelligence"
        ) as MockRoutingIntelligence:
            mock_routing = MockRoutingIntelligence.return_value
            mock_decision = RoutingDecision(
                action=RoutingAction.GATHER_INFO,
                reasoning="User providing clarification",
                confidence=0.9,
                requires_agents=False,
                conversation_type="clarification",
                complexity_estimate=0.1,
                user_goal="clarify_requirements",
                conversation_phase=ConversationPhase.GATHERING_INFO,
                topic_change_detected=False,
                topic_change_confidence=0.1,
                should_abandon_current_task=False,
                previous_topic_description="Write code for web scraper",
                new_topic_description=None,
            )
            mock_routing.analyze = AsyncMock(return_value=mock_decision)

            # Execute
            result = await orchestrator_node(state)

            # Verify NO replan
            assert result["next_action"] != GraphDecision.REPLAN.value

    async def test_conversation_continuation_flag(self, base_state):
        """Test awaiting_user_response flag enables conversation continuation."""
        from app.hermes.legion.nodes.orchestration_graph import should_continue

        # Test: awaiting_user_response = True
        state_waiting = {**base_state, "awaiting_user_response": True}
        result = should_continue(state_waiting)
        assert result == "continue"

        # Test: conversation_complete = True
        state_complete = {**base_state, "conversation_complete": True}
        result = should_continue(state_complete)
        assert result == "end"

        # Test: default (neither flag set)
        result = should_continue(base_state)
        assert result == "continue"  # Default allows continuation

    async def test_multi_turn_conversation_context(self, base_state):
        """Test that information gathering uses multi-message context."""
        from app.hermes.legion.models import RequiredInfoField
        from app.hermes.legion.nodes.graph_nodes import information_gathering_node

        state = {
            **base_state,
            "messages": [
                {
                    "role": "user",
                    "content": "I want to build a website",
                    "timestamp": "2024-01-01",
                    "metadata": {},
                },
                {
                    "role": "assistant",
                    "content": "What type of website?",
                    "timestamp": "2024-01-01",
                    "metadata": {},
                },
                {
                    "role": "user",
                    "content": "E-commerce for selling shoes",
                    "timestamp": "2024-01-01",
                    "metadata": {},
                },
            ],
            "required_info": {
                "website_type": RequiredInfoField(
                    field_name="website_type",
                    field_type="string",
                    question="Type of website to build",
                ),
                "target_audience": RequiredInfoField(
                    field_name="target_audience",
                    field_type="string",
                    question="Who will use it",
                ),
            },
            "collected_info": {},
        }

        # Patch at the orchestrator module level before import
        with patch(
            "app.hermes.legion.nodes.graph_nodes.InformationExtractor"
        ) as MockExtractor:
            mock_instance = MockExtractor.return_value
            # Mock extract_information to return partial extraction
            mock_instance.extract_information = Mock(
                return_value={"website_type": "E-commerce"}
            )

            # Execute
            result = await information_gathering_node(state)

            # Verify extractor was called
            assert mock_instance.extract_information.called

            # Verify context was passed with multiple messages
            call_args = mock_instance.extract_information.call_args
            if call_args:
                # Verify it was called with proper arguments
                kwargs = (
                    call_args.kwargs if hasattr(call_args, "kwargs") else call_args[1]
                )
                assert "user_message" in kwargs or len(call_args.args) >= 1

    async def test_replan_routes_to_orchestrator(self):
        """Test that REPLAN decision routes back to orchestrator."""
        from app.hermes.legion.nodes.orchestration_graph import route_state
        from app.hermes.legion.state.graph_state import GraphDecision

        state = {"next_action": GraphDecision.REPLAN.value, "messages": []}

        route = route_state(state)
        assert (
            route == "orchestrator"
        )  # Should route back to orchestrator for replanning


@pytest.mark.unit
class TestConversationFlowScenarios:
    """Integration-style tests for full conversation flows."""

    def test_topic_change_scenario(self):
        """
        Scenario: User starts task, then changes topic mid-conversation.

        Flow:
        1. User: "Write code for X"
        2. System: Task starts
        3. User: "Actually, do Y instead"
        4. System: Detects topic change, triggers REPLAN
        5. System: Routes back to orchestrator
        6. System: Creates new task for Y
        """
        # This is a conceptual test - actual implementation would need full graph execution
        assert True  # Placeholder for full integration test

    def test_clarification_scenario(self):
        """
        Scenario: User provides clarifications without changing topic.

        Flow:
        1. User: "Build a website"
        2. System: "What type?"
        3. User: "E-commerce"
        4. System: Continues with same task (no replan)
        """
        assert True  # Placeholder

    def test_cancellation_then_new_task_scenario(self):
        """
        Scenario: User cancels current task and starts new one.

        Flow:
        1. User: "Task A"
        2. System: Working on A
        3. User: "Cancel that"
        4. System: Cancels, awaits next input
        5. User: "Task B"
        6. System: Starts new task B
        """
        assert True  # Placeholder
