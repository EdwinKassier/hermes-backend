"""
RoutingIntelligence service implementation.

LLM-powered routing analysis with no predefined categories.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from app.shared.utils.service_loader import get_async_llm_service

from ..utils.llm_utils import extract_json_from_llm_response
from .routing_intelligence import (
    ConversationPhase,
    RiskLevel,
    RoutingAction,
    RoutingDecision,
)

logger = logging.getLogger(__name__)


class RoutingIntelligence:
    """
    LLM-powered routing intelligence with no predefined categories.

    Analyzes messages and conversation context to determine the best
    routing action, providing rich reasoning about decisions.
    """

    def __init__(self):
        self.llm_service = get_async_llm_service()

    async def analyze(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        current_agent_context: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """
        Analyze message and determine routing action.

        Args:
            message: User's current message
            conversation_history: Recent conversation messages
            current_agent_context: Context about active tasks/agents

        Returns:
            RoutingDecision with rich context and reasoning
        """
        # Build rich context for LLM
        context_description = self._build_context(
            conversation_history, current_agent_context
        )

        # Comprehensive routing analysis prompt
        prompt = f"""You are a routing intelligence system for a conversational AI. Analyze this message and determine the best routing action.

{context_description}

**User's Current Message:**
"{message}"

**Your Task:**
Analyze what the user wants and decide how to route this message. Consider:

1. **User's Goal**: What are they trying to accomplish?
2. **Complexity**: Is this simple (direct answer) or complex (needs research/agents)?
3. **Context**: Does this continue the conversation or start something new?
4. **Topic Changes**: Did the user shift topics from what we were discussing?
5. **Risk Assessment**: Does this require sensitive data access or expensive operations?
6. **Multi-Turn Flow**: Are we in the middle of gathering information or refining a task?

**CRITICAL: Intelligent Inference Philosophy**
- **Strongly prefer inference over asking questions**
- Use conversation history, domain knowledge, and reasonable assumptions to fill in missing details
- Only use GATHER_INFO when information is **genuinely ambiguous, contradictory, or missing critical details** that cannot be reasonably inferred
- When in doubt, make an intelligent inference and proceed

**Routing Actions Available:**
- **SIMPLE_RESPONSE**: Can be answered directly without agents
  (greetings, thanks, factual questions, meta questions about the system)

- **GATHER_INFO**: Need more information from user before proceeding
  **USE SPARINGLY** - Only when:
  • Request is genuinely ambiguous with no context (e.g., "research it", "do that thing")
  • Requirements are contradictory or conflicting
  • Sensitive operations requiring explicit confirmation (data deletion, access to private info)
  • Cannot reasonably infer critical parameters from context

  **DO NOT use for**:
  • Missing time periods (infer from context: "trends" = recent, "history" = comprehensive)
  • Missing programming languages (infer from context or use widely-applicable defaults)
  • Missing depth/scope (infer from query complexity and user expertise level)
  • Missing specific details that can be reasonably assumed

- **ORCHESTRATE**: Complex task requiring agents/research/analysis
  (research requests, coding tasks, data analysis, multi-step work)

- **ERROR**: Cannot process (unclear, nonsensical, or too vague)

**Conversation Phases:**
- INITIATING: Starting new conversation/topic
- GATHERING_INFO: Collecting details from user
- EXECUTING: Actively working on task
- REFINING: User providing feedback/changes to ongoing work
- COMPLETING: Wrapping up conversation
- TOPIC_SHIFTING: User changing topics

**Risk Levels:**
- LOW: Safe to auto-execute
- MEDIUM: Consider approval based on settings
- HIGH: Should seek approval
- CRITICAL: Must seek approval

**Response Format (JSON only):**
{{
  "action": "SIMPLE_RESPONSE|GATHER_INFO|ORCHESTRATE|ERROR",
  "reasoning": "<your thought process for this routing decision>",
  "confidence": <0.0-1.0>,
  "requires_agents": <true|false>,
  "conversation_type": "<free-form description>",
  "complexity_estimate": <0.0-1.0>,
  "user_goal": "<what you understand the user wants>",

  "topic_change_detected": <true|false>,
  "topic_change_confidence": <0.0-1.0 if topic changed>,
  "previous_topic_description": "<what we were discussing, if topic changed>",
  "new_topic_description": "<new topic, if changed>",
  "should_abandon_current_task": <true|false, if topic changed>,

  "should_seek_approval": <true|false>,
  "approval_reason": "<why approval needed, if applicable>",
  "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
  "estimated_cost": <cost in dollars, if estimable>,
  "estimated_duration": "<duration string like '2 minutes', if estimable>",

  "conversation_phase": "INITIATING|GATHERING_INFO|EXECUTING|REFINING|COMPLETING|TOPIC_SHIFTING",
  "requires_followup": <true|false>,
  "awaiting_user_input": <true|false>,
  "suggested_next_question": "<what to ask next, if applicable>",

  "builds_on_previous": <true|false>,
  "references_conversation_context": <true|false>,
  "context_summary": "<summary of relevant context, if applicable>"
}}

**Examples of conversation_type (be creative, these are just examples):**
- "casual greeting"
- "gratitude expression"
- "complex multi-domain research request"
- "follow-up question on previous topic"
- "meta-question about system capabilities"
- "clarification providing requested information"
- "topic shift to new area while task is active"

**Inference Examples:**
✅ "Research AI trends" → Infer: recent/current trends (last 6 months), moderate depth
✅ "Write a sorting function" → Infer: Python (widely applicable), standard implementation
✅ "Analyze the data" → Infer: comprehensive analysis, all relevant metrics
❌ "Research it" → Cannot infer what "it" refers to → GATHER_INFO
❌ "Delete all my data" → Sensitive operation → Seek explicit confirmation

Be thoughtful and context-aware. The same words can mean different things in different contexts. **Prioritize intelligent inference to create fluid, natural conversations.**"""

        try:
            response = await self.llm_service.generate_async(prompt, persona="hermes")
            data = extract_json_from_llm_response(response)

            # Parse and validate the decision
            decision = RoutingDecision(
                action=RoutingAction(data["action"].lower()),
                reasoning=data["reasoning"],
                confidence=float(data["confidence"]),
                requires_agents=bool(data["requires_agents"]),
                conversation_type=data["conversation_type"],
                complexity_estimate=float(data["complexity_estimate"]),
                user_goal=data["user_goal"],
                topic_change_detected=bool(data.get("topic_change_detected", False)),
                topic_change_confidence=float(data.get("topic_change_confidence", 0.0)),
                previous_topic_description=data.get("previous_topic_description"),
                new_topic_description=data.get("new_topic_description"),
                should_abandon_current_task=bool(
                    data.get("should_abandon_current_task", False)
                ),
                should_seek_approval=bool(data.get("should_seek_approval", False)),
                approval_reason=data.get("approval_reason"),
                risk_level=RiskLevel(data.get("risk_level", "low").lower()),
                estimated_cost=data.get("estimated_cost"),
                estimated_duration=data.get("estimated_duration"),
                conversation_phase=ConversationPhase(
                    data["conversation_phase"].lower()
                ),
                requires_followup=bool(data.get("requires_followup", False)),
                awaiting_user_input=bool(data.get("awaiting_user_input", False)),
                suggested_next_question=data.get("suggested_next_question"),
                builds_on_previous=bool(data.get("builds_on_previous", False)),
                references_conversation_context=bool(
                    data.get("references_conversation_context", False)
                ),
                context_summary=data.get("context_summary"),
            )

            logger.info(
                f"Routing decision: {decision.action} "
                f"(type: '{decision.conversation_type}', "
                f"confidence: {decision.confidence:.2f})"
            )

            if decision.topic_change_detected:
                logger.info(
                    f"Topic change detected ({decision.topic_change_confidence:.2f}): "
                    f"{decision.previous_topic_description} → {decision.new_topic_description}"
                )

            return decision

        except Exception as e:
            logger.error(f"Routing analysis failed: {e}", exc_info=True)
            # Safe default - orchestrate to be conservative
            return RoutingDecision(
                action=RoutingAction.ORCHESTRATE,
                reasoning=f"Analysis failed, defaulting to orchestration for safety: {str(e)}",
                confidence=0.5,
                requires_agents=True,
                conversation_type="unknown - analysis failure",
                complexity_estimate=0.7,
                user_goal="unknown",
                conversation_phase=ConversationPhase.INITIATING,
            )

    def _build_context(
        self,
        history: Optional[List[Dict[str, str]]],
        agent_context: Optional[Dict[str, Any]],
    ) -> str:
        """
        Build rich context description for LLM.

        Args:
            history: Conversation history
            agent_context: Active agent/task context

        Returns:
            Formatted context string
        """
        parts = []

        # Add conversation history
        if history and len(history) > 0:
            recent = history[-5:]  # Last 5 messages
            history_text = "\n".join(
                [
                    f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                    for msg in recent
                ]
            )
            parts.append(f"**Recent Conversation:**\n{history_text}")
        else:
            parts.append("**New Conversation** (no prior context)")

        # Add agent context
        if agent_context:
            context_parts = []

            if agent_context.get("active_task"):
                context_parts.append(f"Active Task: {agent_context['active_task']}")

            if agent_context.get("waiting_for_info"):
                info_needed = agent_context.get("required_info", {})
                if info_needed:
                    fields = ", ".join(info_needed.keys())
                    context_parts.append(f"Waiting for information: {fields}")

            if agent_context.get("collected_info"):
                collected = agent_context["collected_info"]
                if collected:
                    context_parts.append(
                        f"Information collected: {', '.join(collected.keys())}"
                    )

            if agent_context.get("conversation_phase"):
                context_parts.append(
                    f"Current phase: {agent_context['conversation_phase']}"
                )

            if context_parts:
                parts.append("**Current Context:**\n" + "\n".join(context_parts))

        return "\n\n".join(parts)

    async def analyze_batch(
        self,
        messages: List[str],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> List[RoutingDecision]:
        """
        Analyze multiple messages in parallel.

        Useful for testing or analyzing conversation patterns.

        Args:
            messages: List of messages to analyze
            conversation_history: Shared conversation history

        Returns:
            List of routing decisions
        """
        tasks = [self.analyze(msg, conversation_history) for msg in messages]
        return await asyncio.gather(*tasks)
