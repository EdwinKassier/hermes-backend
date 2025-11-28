"""Unit tests for RoutingIntelligence service."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock GeminiService before imports that depend on it
mock_gemini_module = MagicMock()
sys.modules["app.shared.services.GeminiService"] = mock_gemini_module
mock_gemini_module.GeminiService = MagicMock()

from app.hermes.legion.intelligence.routing_intelligence import (  # noqa: E402
    ConversationPhase,
    RiskLevel,
    RoutingAction,
    RoutingDecision,
)
from app.hermes.legion.intelligence.routing_service import (  # noqa: E402
    RoutingIntelligence,
)


@pytest.fixture
def mock_async_llm_service():
    """Mock the async LLM service."""
    service = MagicMock()
    service.generate_async = AsyncMock()
    with patch(
        "app.hermes.legion.intelligence.routing_service.get_async_llm_service",
        return_value=service,
    ):
        yield service


@pytest.mark.asyncio
@pytest.mark.unit
class TestRoutingIntelligence:
    """Test suite for routing intelligence."""

    @pytest.mark.asyncio
    async def test_simple_greeting_routing(self, mock_async_llm_service):
        """Test that greetings route to simple response."""
        mock_async_llm_service.generate_async.return_value = """
        {
            "action": "SIMPLE_RESPONSE",
            "reasoning": "Simple greeting, no action requested",
            "confidence": 0.95,
            "requires_agents": false,
            "conversation_type": "casual greeting",
            "complexity_estimate": 0.05,
            "user_goal": "say hello",
            "topic_change_detected": false,
            "should_seek_approval": false,
            "risk_level": "LOW",
            "conversation_phase": "INITIATING",
            "requires_followup": false,
            "awaiting_user_input": false,
            "builds_on_previous": false,
            "references_conversation_context": false
        }
        """

        intel = RoutingIntelligence()
        decision = await intel.analyze("hi there!")

        assert decision.action == RoutingAction.SIMPLE_RESPONSE
        assert decision.requires_agents == False
        assert decision.complexity_estimate < 0.2
        assert decision.conversation_phase == ConversationPhase.INITIATING

    @pytest.mark.asyncio
    async def test_complex_research_routing(self, mock_async_llm_service):
        """Test that complex research routes to orchestration."""
        mock_async_llm_service.generate_async.return_value = """
        {
            "action": "ORCHESTRATE",
            "reasoning": "Complex multi-domain research requiring factual investigation",
            "confidence": 0.92,
            "requires_agents": true,
            "conversation_type": "complex research request spanning multiple domains",
            "complexity_estimate": 0.85,
            "user_goal": "comprehensive analysis of quantum computing developments",
            "topic_change_detected": false,
            "should_seek_approval": true,
            "approval_reason": "High complexity research requiring multiple sources",
            "risk_level": "MEDIUM",
            "estimated_duration": "5-10 minutes",
            "conversation_phase": "INITIATING",
            "requires_followup": false,
            "awaiting_user_input": false,
            "builds_on_previous": false,
            "references_conversation_context": false
        }
        """

        intel = RoutingIntelligence()
        decision = await intel.analyze(
            "Research quantum computing developments in 2024"
        )

        assert decision.action == RoutingAction.ORCHESTRATE
        assert decision.requires_agents == True
        assert decision.complexity_estimate > 0.7
        assert decision.should_seek_approval == True
        assert decision.risk_level == RiskLevel.MEDIUM

    @pytest.mark.asyncio
    async def test_topic_change_detection(self, mock_async_llm_service):
        """Test topic change detection with active context."""
        mock_async_llm_service.generate_async.return_value = """
        {
            "action": "ORCHESTRATE",
            "reasoning": "User explicitly changed topic from quantum to climate",
            "confidence": 0.93,
            "requires_agents": true,
            "conversation_type": "topic shift to new research domain",
            "complexity_estimate": 0.75,
            "user_goal": "pivot to climate change research",
            "topic_change_detected": true,
            "topic_change_confidence": 0.95,
            "previous_topic_description": "quantum computing research",
            "new_topic_description": "climate change information",
            "should_abandon_current_task": true,
            "should_seek_approval": false,
            "risk_level": "LOW",
            "conversation_phase": "TOPIC_SHIFTING",
            "requires_followup": false,
            "awaiting_user_input": false,
            "builds_on_previous": false,
            "references_conversation_context": true
        }
        """

        intel = RoutingIntelligence()

        history = [
            {"role": "user", "content": "Research quantum computing"},
            {
                "role": "assistant",
                "content": "I'll research quantum computing for you...",
            },
        ]

        context = {"active_task": "quantum_research_task"}

        decision = await intel.analyze(
            "Actually, tell me about climate change instead",
            conversation_history=history,
            current_agent_context=context,
        )

        assert decision.topic_change_detected == True
        assert decision.topic_change_confidence > 0.8
        assert decision.should_abandon_current_task == True
        assert decision.conversation_phase == ConversationPhase.TOPIC_SHIFTING

    @pytest.mark.asyncio
    async def test_information_gathering_flow(self, mock_async_llm_service):
        """Test multi-turn information gathering."""
        mock_async_llm_service.generate_async.return_value = """
        {
            "action": "GATHER_INFO",
            "reasoning": "User provided website type, but need product details for e-commerce",
            "confidence": 0.88,
            "requires_agents": false,
            "conversation_type": "clarification - providing requested website type",
            "complexity_estimate": 0.15,
            "user_goal": "complete website specification",
            "topic_change_detected": false,
            "should_seek_approval": false,
            "risk_level": "LOW",
            "conversation_phase": "GATHERING_INFO",
            "requires_followup": true,
            "awaiting_user_input": true,
            "suggested_next_question": "What products will you be selling?",
            "builds_on_previous": true,
            "references_conversation_context": true,
            "context_summary": "Building e-commerce website, need product category"
        }
        """

        intel = RoutingIntelligence()

        history = [
            {"role": "user", "content": "Build me a website"},
            {"role": "assistant", "content": "What type of website?"},
        ]

        decision = await intel.analyze("E-commerce", conversation_history=history)

        assert decision.action == RoutingAction.GATHER_INFO
        assert decision.conversation_phase == ConversationPhase.GATHERING_INFO
        assert decision.requires_followup == True
        assert decision.suggested_next_question is not None
        assert decision.builds_on_previous == True

    @pytest.mark.asyncio
    async def test_refining_phase(self, mock_async_llm_service):
        """Test user refinement during execution."""
        mock_async_llm_service.generate_async.return_value = """
        {
            "action": "ORCHESTRATE",
            "reasoning": "User refining scope of ongoing research to focus on specific aspect",
            "confidence": 0.87,
            "requires_agents": true,
            "conversation_type": "refinement request - narrowing scope",
            "complexity_estimate": 0.60,
            "user_goal": "focus quantum research on cryptography applications",
            "topic_change_detected": false,
            "should_abandon_current_task": false,
            "should_seek_approval": false,
            "risk_level": "LOW",
            "conversation_phase": "REFINING",
            "requires_followup": false,
            "awaiting_user_input": false,
            "builds_on_previous": true,
            "references_conversation_context": true,
            "context_summary": "Refining quantum computing research to focus on cryptographic applications"
        }
        """

        intel = RoutingIntelligence()

        history = [
            {"role": "user", "content": "Research quantum computing"},
            {
                "role": "assistant",
                "content": "I found info on basics, applications, and hardware...",
            },
        ]

        context = {"active_task": "quantum_research", "conversation_phase": "executing"}

        decision = await intel.analyze(
            "Focus on cryptography applications",
            conversation_history=history,
            current_agent_context=context,
        )

        assert decision.conversation_phase == ConversationPhase.REFINING
        assert decision.topic_change_detected == False
        assert decision.should_abandon_current_task == False
        assert decision.builds_on_previous == True

    @pytest.mark.asyncio
    async def test_high_risk_approval_needed(self, mock_async_llm_service):
        """Test high-risk operations require approval."""
        mock_async_llm_service.generate_async.return_value = """
        {
            "action": "ORCHESTRATE",
            "reasoning": "High-volume data processing requiring email access",
            "confidence": 0.88,
            "requires_agents": true,
            "conversation_type": "sensitive data analysis with privacy implications",
            "complexity_estimate": 0.75,
            "user_goal": "email topic summarization",
            "topic_change_detected": false,
            "should_seek_approval": true,
            "approval_reason": "Requires email access and processing potentially sensitive information",
            "risk_level": "HIGH",
            "estimated_cost": 0.50,
            "estimated_duration": "5-10 minutes",
            "conversation_phase": "INITIATING",
            "requires_followup": false,
            "awaiting_user_input": false,
            "builds_on_previous": false,
            "references_conversation_context": false
        }
        """

        intel = RoutingIntelligence()
        decision = await intel.analyze(
            "Analyze all my recent emails and summarize by topic"
        )

        assert decision.should_seek_approval == True
        assert decision.risk_level == RiskLevel.HIGH
        assert decision.approval_reason is not None
        assert decision.estimated_cost is not None

    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(self, mock_async_llm_service):
        """Test graceful fallback when LLM fails."""
        mock_async_llm_service.generate_async.side_effect = Exception("LLM timeout")

        intel = RoutingIntelligence()
        decision = await intel.analyze("anything")

        # Should default to ORCHESTRATE (most conservative)
        assert decision.action == RoutingAction.ORCHESTRATE
        assert decision.confidence == 0.5
        assert "failed" in decision.reasoning.lower()

    @pytest.mark.asyncio
    async def test_batch_analysis(self, mock_async_llm_service):
        """Test parallel analysis of multiple messages."""
        mock_async_llm_service.generate_async.side_effect = [
            '{"action": "SIMPLE_RESPONSE", "reasoning": "greeting", "confidence": 0.9, "requires_agents": false, "conversation_type": "greeting", "complexity_estimate": 0.05, "user_goal": "greet", "topic_change_detected": false, "should_seek_approval": false, "risk_level": "LOW", "conversation_phase": "INITIATING", "requires_followup": false, "awaiting_user_input": false, "builds_on_previous": false, "references_conversation_context": false}',
            '{"action": "ORCHESTRATE", "reasoning": "research", "confidence": 0.85, "requires_agents": true, "conversation_type": "research", "complexity_estimate": 0.75, "user_goal": "research", "topic_change_detected": false, "should_seek_approval": false, "risk_level": "LOW", "conversation_phase": "INITIATING", "requires_followup": false, "awaiting_user_input": false, "builds_on_previous": false, "references_conversation_context": false}',
        ]

        intel = RoutingIntelligence()
        messages = ["hi", "research AI"]
        decisions = await intel.analyze_batch(messages)

        assert len(decisions) == 2
        assert decisions[0].action == RoutingAction.SIMPLE_RESPONSE
        assert decisions[1].action == RoutingAction.ORCHESTRATE
