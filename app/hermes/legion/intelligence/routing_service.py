"""
RoutingIntelligence service implementation.

LLM-powered routing analysis with no predefined categories.

Now with structured output support for guaranteed schema compliance.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

# LangChain imports for structured output
try:
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import HumanMessage

    STRUCTURED_OUTPUT_AVAILABLE = True
except ImportError:
    STRUCTURED_OUTPUT_AVAILABLE = False

from app.shared.utils.service_loader import get_async_llm_service

from ..utils.llm_utils import extract_json_from_llm_response
from ..utils.persona_context import get_current_legion_persona
from .routing_intelligence import (
    ConversationPhase,
    RiskLevel,
    RoutingAction,
    RoutingDecision,
)

logger = logging.getLogger(__name__)

# Configuration for structured output (can be overridden via environment)
USE_STRUCTURED_OUTPUT = os.getenv("USE_STRUCTURED_OUTPUT", "true").lower() == "true"


class RoutingIntelligence:
    """
    LLM-powered routing intelligence with no predefined categories.

    Analyzes messages and conversation context to determine the best
    routing action, providing rich reasoning about decisions.

    Features:
    - Structured output: Uses with_structured_output() for guaranteed schema compliance
    - Graceful fallback: Falls back to JSON parsing if structured output fails
    - Rich context: Provides detailed reasoning for routing decisions
    """

    def __init__(self, use_structured_output: bool = USE_STRUCTURED_OUTPUT):
        self.llm_service = get_async_llm_service()
        self.use_structured_output = (
            use_structured_output and STRUCTURED_OUTPUT_AVAILABLE
        )
        self._structured_model = None  # Cached structured output model

        if self.use_structured_output:
            logger.info(
                "RoutingIntelligence initialized with structured output support"
            )
        else:
            logger.info("RoutingIntelligence using JSON extraction fallback")

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
  (greetings, thanks, factual questions, meta questions about the system,
   **simple code snippets** like "hello world", basic functions, one-liner examples,
   definitions, explanations, yes/no questions, simple calculations)

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
  (multi-step research, complex analysis with multiple perspectives, tasks requiring
   external data retrieval, multi-file code projects, comprehensive reports)

  **DO NOT use for**:
  • Simple code snippets (hello world, single functions, basic examples)
  • Factual questions with well-known answers
  • Basic explanations or definitions

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
  "context_summary": "<summary of relevant context, if applicable>",

  "is_answer_refinement": <true|false>,
  "refinement_type": "repeat|clarify|update|expand|null",
  "prior_answer_reference": "<what prior answer they're asking about, if applicable>"
}}

**Examples of conversation_type (be creative, these are just examples):**
- "casual greeting"
- "gratitude expression"
- "complex multi-domain research request"
- "follow-up question on previous topic"
- "meta-question about system capabilities"
- "clarification providing requested information"
- "topic shift to new area while task is active"
- "request to repeat/clarify previous answer"
- "refinement of prior response"

**Answer Refinement Detection:**
Detect when user is asking to REPEAT, CLARIFY, UPDATE, or EXPAND a previous answer:
- "Tell me again" / "What was that?" / "Can you repeat?" → refinement_type: "repeat"
- "What do you mean by X?" / "Explain that part" → refinement_type: "clarify"
- "Update that with..." / "Change X to Y" → refinement_type: "update"
- "Tell me more about..." / "Expand on..." → refinement_type: "expand"

When is_answer_refinement is true:
- Use SIMPLE_RESPONSE (for repeating/clarifying) or ORCHESTRATE (for updating with new work)
- Set prior_answer_reference to identify what they're asking about
- DO NOT ask for clarification - use the conversation history to determine what they're referencing

**Inference Examples:**
✅ "Research AI trends" → Infer: recent/current trends (last 6 months), moderate depth
✅ "Write a sorting function" → Infer: Python (widely applicable), standard implementation
✅ "Analyze the data" → Infer: comprehensive analysis, all relevant metrics
❌ "Research it" → Cannot infer what "it" refers to → GATHER_INFO
❌ "Delete all my data" → Sensitive operation → Seek explicit confirmation

Be thoughtful and context-aware. The same words can mean different things in different contexts. **Prioritize intelligent inference to create fluid, natural conversations.**"""

        try:
            # Try structured output first if available (guaranteed schema compliance)
            if self.use_structured_output:
                try:
                    decision = await self._analyze_with_structured_output(prompt)
                    logger.info(
                        f"Routing decision (structured): {decision.action} "
                        f"(type: '{decision.conversation_type}', "
                        f"confidence: {decision.confidence:.2f})"
                    )
                    return decision
                except Exception as structured_error:
                    logger.warning(
                        f"Structured output failed, falling back to JSON: {structured_error}"
                    )

            # Fallback: Use AsyncLLMService with JSON extraction
            response = await self.llm_service.generate_async(
                prompt, persona=get_current_legion_persona()
            )
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
                # Answer refinement fields
                is_answer_refinement=bool(data.get("is_answer_refinement", False)),
                refinement_type=data.get("refinement_type"),
                prior_answer_reference=data.get("prior_answer_reference"),
            )

            logger.info(
                f"Routing decision (JSON): {decision.action} "
                f"(type: '{decision.conversation_type}', "
                f"confidence: {decision.confidence:.2f})"
            )

            if decision.topic_change_detected:
                logger.info(
                    f"Topic change detected ({decision.topic_change_confidence:.2f}): "
                    f"{decision.previous_topic_description} → {decision.new_topic_description}"
                )

            if decision.is_answer_refinement:
                logger.info(
                    f"Answer refinement detected: type={decision.refinement_type}, "
                    f"ref={decision.prior_answer_reference}"
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

    async def _analyze_with_structured_output(self, prompt: str) -> RoutingDecision:
        """
        Analyze using LangChain's structured output for guaranteed schema compliance.

        Uses with_structured_output() to ensure the LLM response matches the
        RoutingDecision Pydantic model exactly. This eliminates JSON parsing errors
        and provides type-safe routing decisions.

        Args:
            prompt: The analysis prompt to send to the LLM

        Returns:
            RoutingDecision validated by Pydantic

        Raises:
            Exception if structured output fails (caller should fall back to JSON)
        """
        import os

        # Get or create the structured model (cached for performance)
        if self._structured_model is None:
            model_name = os.getenv("LLM_MODEL", "gemini-2.5-flash")

            # Use ChatGoogleGenerativeAI for Gemini models (Google AI Studio)
            if model_name.startswith("gemini"):
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI

                    google_api_key = os.getenv("GOOGLE_API_KEY")
                    if google_api_key:
                        base_model = ChatGoogleGenerativeAI(
                            model=model_name,
                            temperature=0.3,
                            max_retries=2,
                            google_api_key=google_api_key,
                        )
                        logger.info(
                            f"Using ChatGoogleGenerativeAI for structured output ({model_name})"
                        )
                    else:
                        base_model = init_chat_model(
                            model_name,
                            temperature=0.3,
                            max_retries=2,
                        )
                except ImportError:
                    base_model = init_chat_model(
                        model_name,
                        temperature=0.3,
                        max_retries=2,
                    )
            else:
                base_model = init_chat_model(
                    model_name,
                    temperature=0.3,
                    max_retries=2,
                )

            self._structured_model = base_model.with_structured_output(RoutingDecision)
            logger.info(
                f"Created structured output model for RoutingDecision ({model_name})"
            )

        # Use ainvoke for async structured output
        import asyncio

        decision = await asyncio.to_thread(
            self._structured_model.invoke, [HumanMessage(content=prompt)]
        )

        return decision

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

        # Add conversation history - use extended window for better context
        if history and len(history) > 0:
            # Use last 10 messages for better conversation continuity
            recent = history[-10:]
            history_text = "\n".join(
                [
                    f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                    for msg in recent
                ]
            )
            parts.append(f"**Recent Conversation:**\n{history_text}")

            # Extract key topics from older history for deeper context
            if len(history) > 10:
                key_topics = self._extract_key_topics(history[:-10])
                if key_topics:
                    parts.append(f"**Earlier Topics Discussed:**\n{key_topics}")
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

    def _extract_key_topics(self, history: List[Dict[str, str]]) -> str:
        """
        Extract key topics from older conversation history.

        Provides a concise summary of earlier topics to maintain
        continuity without overwhelming the context window.

        Args:
            history: Older conversation messages (before recent window)

        Returns:
            String summarizing key topics discussed
        """
        if not history:
            return ""

        # Extract key nouns/topics from user messages
        topics = set()
        for msg in history:
            if msg.get("role") == "user":
                content = msg.get("content", "").lower()
                # Simple keyword extraction for common task indicators
                keywords = [
                    "research",
                    "code",
                    "analyze",
                    "write",
                    "create",
                    "explain",
                    "help",
                    "find",
                    "search",
                    "build",
                ]
                for keyword in keywords:
                    if keyword in content:
                        # Get context around keyword (rough topic extraction)
                        idx = content.find(keyword)
                        snippet = content[max(0, idx - 5) : min(len(content), idx + 30)]
                        topics.add(snippet.strip())

        if not topics:
            return ""

        # Limit to top 3 topics
        topic_list = list(topics)[:3]
        return "- " + "\n- ".join(topic_list)

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
