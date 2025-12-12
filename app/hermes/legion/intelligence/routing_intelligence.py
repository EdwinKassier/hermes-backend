"""
Routing intelligence for dynamic conversation handling.

This module provides LLM-powered routing decisions without predefined
categories, enabling flexible conversation flows.
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RoutingAction(str, Enum):
    """Actions the orchestrator can take."""

    SIMPLE_RESPONSE = "simple_response"  # Direct response, no agents
    GATHER_INFO = "gather_info"  # Need more information from user
    ORCHESTRATE = "orchestrate"  # Complex task needing agents
    ERROR = "error"  # Cannot process


class ConversationPhase(str, Enum):
    """Where we are in the conversation."""

    INITIATING = "initiating"  # Starting new topic/task
    GATHERING_INFO = "gathering_info"  # Collecting details from user
    EXECUTING = "executing"  # Actively working on task
    REFINING = "refining"  # User providing feedback/refinement
    COMPLETING = "completing"  # Wrapping up
    TOPIC_SHIFTING = "topic_shifting"  # User changing topics


class RiskLevel(str, Enum):
    """Risk assessment for automated actions."""

    LOW = "low"  # Safe to auto-execute
    MEDIUM = "medium"  # Consider approval based on settings
    HIGH = "high"  # Should seek approval
    CRITICAL = "critical"  # Must seek approval


class RoutingDecision(BaseModel):
    """
    Enhanced routing decision for complex conversations.

    Provides rich context about user intent and suggested actions
    without forcing conversations into predefined categories.
    """

    # Core routing
    action: RoutingAction = Field(..., description="What action to take")
    reasoning: str = Field(..., description="Why this routing decision was made")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in this decision"
    )

    # Rich understanding
    requires_agents: bool = Field(..., description="Whether agents are needed")
    conversation_type: str = Field(
        ..., description="Free-form description of conversation type"
    )
    complexity_estimate: float = Field(
        ..., ge=0.0, le=1.0, description="Task complexity (0=trivial, 1=very complex)"
    )
    user_goal: str = Field(..., description="What the user is trying to accomplish")

    # Topic change detection
    topic_change_detected: bool = Field(
        default=False, description="Whether user changed topics"
    )
    topic_change_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence in topic change detection"
    )
    previous_topic_description: Optional[str] = Field(
        default=None, description="What we were discussing before"
    )
    new_topic_description: Optional[str] = Field(
        default=None, description="What user now wants to discuss"
    )
    should_abandon_current_task: bool = Field(
        default=False, description="Whether to cancel active tasks"
    )

    # Human-in-the-loop signals
    should_seek_approval: bool = Field(
        default=False, description="Whether to interrupt for human approval"
    )
    approval_reason: Optional[str] = Field(
        default=None, description="Why approval is needed"
    )
    risk_level: RiskLevel = Field(
        default=RiskLevel.LOW, description="Risk level of this action"
    )
    estimated_cost: Optional[float] = Field(
        default=None, description="Estimated cost in dollars"
    )
    estimated_duration: Optional[str] = Field(
        default=None, description="Estimated duration (e.g., '2 minutes')"
    )

    # Multi-turn conversation
    conversation_phase: ConversationPhase = Field(
        default=ConversationPhase.INITIATING, description="Current conversation phase"
    )
    requires_followup: bool = Field(
        default=False, description="Whether we expect user to respond"
    )
    awaiting_user_input: bool = Field(
        default=False, description="Whether currently waiting for user"
    )
    suggested_next_question: Optional[str] = Field(
        default=None, description="What to ask user next"
    )

    # Context awareness
    builds_on_previous: bool = Field(
        default=False, description="Whether this builds on previous messages"
    )
    references_conversation_context: bool = Field(
        default=False, description="Whether this references prior context"
    )
    context_summary: Optional[str] = Field(
        default=None, description="Summary of relevant context"
    )

    # Answer refinement/repeat detection (Issue 1: response flexibility)
    is_answer_refinement: bool = Field(
        default=False,
        description="Whether user is asking to repeat, clarify, or update a previous answer",
    )
    refinement_type: Optional[str] = Field(
        default=None,
        description="Type of refinement: 'repeat', 'clarify', 'update', 'expand'",
    )
    prior_answer_reference: Optional[str] = Field(
        default=None,
        description="Reference to which prior answer should be refined or repeated",
    )
