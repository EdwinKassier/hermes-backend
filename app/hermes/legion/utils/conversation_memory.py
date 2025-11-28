"""
Conversation Memory Management for Long Conversations.

This module provides utilities for managing conversation history to prevent
context window overflow while maintaining relevant context.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.shared.utils.service_loader import get_async_llm_service

from ..utils.llm_utils import extract_json_from_llm_response

logger = logging.getLogger(__name__)


# Default thresholds
DEFAULT_MAX_MESSAGES = 50  # Before summarization triggers
DEFAULT_SUMMARY_THRESHOLD = 20  # Messages to summarize at once
DEFAULT_KEEP_RECENT = 10  # Recent messages to always keep
DEFAULT_MAX_CONTEXT_TOKENS = 8000  # Approximate token budget for context


@dataclass
class ConversationSummary:
    """A summary of a portion of conversation history."""

    summary_text: str
    message_count: int  # Number of messages summarized
    start_timestamp: str
    end_timestamp: str
    key_topics: List[str] = field(default_factory=list)
    key_decisions: List[str] = field(default_factory=list)
    pending_tasks: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ManagedConversation:
    """
    Conversation with managed memory.

    Maintains both raw recent messages and summarized older context.
    """

    summaries: List[ConversationSummary] = field(default_factory=list)
    recent_messages: List[Dict[str, Any]] = field(default_factory=list)
    total_message_count: int = 0

    def get_context_window(self, max_recent: int = 10) -> Dict[str, Any]:
        """
        Get the context window for LLM calls.

        Returns:
            Dictionary with 'context_summary' and 'recent_messages'
        """
        # Combine summaries into context
        if self.summaries:
            combined_summary = "\n\n".join([s.summary_text for s in self.summaries])
            pending = []
            for s in self.summaries:
                pending.extend(s.pending_tasks)
        else:
            combined_summary = None
            pending = []

        return {
            "context_summary": combined_summary,
            "recent_messages": self.recent_messages[-max_recent:],
            "total_messages": self.total_message_count,
            "summaries_count": len(self.summaries),
            "pending_tasks": list(set(pending)),  # Deduplicate
        }


class ConversationSummarizer:
    """
    Summarizes conversation history to manage context window.

    Uses LLM to create intelligent summaries that preserve:
    - Key information and decisions
    - Task context and status
    - User preferences and requirements
    """

    def __init__(
        self,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        summary_threshold: int = DEFAULT_SUMMARY_THRESHOLD,
        keep_recent: int = DEFAULT_KEEP_RECENT,
    ):
        self.max_messages = max_messages
        self.summary_threshold = summary_threshold
        self.keep_recent = keep_recent
        self.llm_service = get_async_llm_service()

    async def summarize_messages(
        self, messages: List[Dict[str, Any]], context: Optional[str] = None
    ) -> ConversationSummary:
        """
        Create a summary of a batch of messages.

        Args:
            messages: List of message dictionaries
            context: Optional existing context to incorporate

        Returns:
            ConversationSummary object
        """
        if not messages:
            return ConversationSummary(
                summary_text="No messages to summarize.",
                message_count=0,
                start_timestamp="",
                end_timestamp="",
            )

        # Build conversation text
        conversation_text = self._format_messages_for_summary(messages)

        # Get timestamps
        start_ts = messages[0].get("timestamp", "")
        end_ts = messages[-1].get("timestamp", "")

        prompt = f"""Summarize this conversation segment for use as context in future interactions.

{f"EXISTING CONTEXT (incorporate if relevant):{chr(10)}{context}{chr(10)}{chr(10)}" if context else ""}CONVERSATION TO SUMMARIZE:
{conversation_text}

Create a summary that:
1. Captures key information exchanged
2. Notes any decisions or conclusions reached
3. Identifies pending tasks or follow-ups
4. Preserves user preferences and requirements
5. Is concise but complete enough for context

Return JSON in this format:
{{
    "summary": "Comprehensive summary text...",
    "key_topics": ["topic1", "topic2"],
    "key_decisions": ["decision1", "decision2"],
    "pending_tasks": ["task1", "task2"],
    "user_preferences": ["preference1"]
}}
"""

        try:
            response = await self.llm_service.generate_async(prompt, persona="hermes")
            data = extract_json_from_llm_response(response)

            return ConversationSummary(
                summary_text=data.get("summary", "Summary generation failed."),
                message_count=len(messages),
                start_timestamp=start_ts,
                end_timestamp=end_ts,
                key_topics=data.get("key_topics", []),
                key_decisions=data.get("key_decisions", []),
                pending_tasks=data.get("pending_tasks", []),
            )

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Fallback: simple concatenation
            return ConversationSummary(
                summary_text=self._fallback_summary(messages),
                message_count=len(messages),
                start_timestamp=start_ts,
                end_timestamp=end_ts,
            )

    def _format_messages_for_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages as readable text for summarization."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Truncate very long messages
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"{role.upper()}: {content}")
        return "\n".join(lines)

    def _fallback_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Create a simple fallback summary without LLM."""
        user_messages = [m for m in messages if m.get("role") == "user"]
        assistant_messages = [m for m in messages if m.get("role") == "assistant"]

        summary_parts = [
            f"Conversation with {len(messages)} messages.",
            f"User sent {len(user_messages)} messages.",
            f"Assistant responded {len(assistant_messages)} times.",
        ]

        # Extract first and last user message as context
        if user_messages:
            first_user = user_messages[0].get("content", "")[:100]
            summary_parts.append(f"Started with: {first_user}...")
            if len(user_messages) > 1:
                last_user = user_messages[-1].get("content", "")[:100]
                summary_parts.append(f"Most recent: {last_user}...")

        return " ".join(summary_parts)

    async def manage_conversation_memory(
        self,
        messages: List[Dict[str, Any]],
        existing_summaries: List[ConversationSummary] = None,
    ) -> ManagedConversation:
        """
        Manage conversation memory by summarizing older messages.

        Args:
            messages: Full list of conversation messages
            existing_summaries: Previously created summaries

        Returns:
            ManagedConversation with summaries and recent messages
        """
        existing_summaries = existing_summaries or []

        if len(messages) <= self.keep_recent:
            # No summarization needed
            return ManagedConversation(
                summaries=existing_summaries,
                recent_messages=messages,
                total_message_count=len(messages),
            )

        # Check if we need to summarize
        if len(messages) > self.max_messages:
            # Summarize older messages
            messages_to_summarize = messages[: -self.keep_recent]
            recent_messages = messages[-self.keep_recent :]

            # Batch summarization if needed
            new_summaries = []
            for i in range(0, len(messages_to_summarize), self.summary_threshold):
                batch = messages_to_summarize[i : i + self.summary_threshold]
                # Include previous summary as context
                prev_context = (
                    existing_summaries[-1].summary_text if existing_summaries else None
                )
                summary = await self.summarize_messages(batch, context=prev_context)
                new_summaries.append(summary)

            return ManagedConversation(
                summaries=existing_summaries + new_summaries,
                recent_messages=recent_messages,
                total_message_count=len(messages),
            )

        # Under threshold, keep all recent and existing summaries
        return ManagedConversation(
            summaries=existing_summaries,
            recent_messages=messages[-self.keep_recent :],
            total_message_count=len(messages),
        )


class ConversationContextBuilder:
    """
    Builds optimized context for LLM calls from conversation history.

    Balances between providing enough context and staying within token limits.
    """

    def __init__(self, max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS):
        self.max_context_tokens = max_context_tokens
        self.summarizer = ConversationSummarizer()

    async def build_context(
        self,
        messages: List[Dict[str, Any]],
        current_query: str,
        task_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build optimized context for an LLM call.

        Args:
            messages: Full conversation history
            current_query: The current user query
            task_context: Optional active task context

        Returns:
            Dictionary with context components ready for LLM
        """
        # Manage memory if conversation is long
        managed = await self.summarizer.manage_conversation_memory(messages)

        # Build context window
        context_window = managed.get_context_window()

        # Estimate tokens (rough: ~4 chars per token)
        estimated_tokens = self._estimate_tokens(context_window)

        # If over budget, reduce recent messages
        while (
            estimated_tokens > self.max_context_tokens
            and len(context_window["recent_messages"]) > 3
        ):
            context_window["recent_messages"] = context_window["recent_messages"][1:]
            estimated_tokens = self._estimate_tokens(context_window)

        # Build the final context
        context = {
            "conversation_summary": context_window["context_summary"],
            "recent_messages": context_window["recent_messages"],
            "current_query": current_query,
            "pending_tasks": context_window["pending_tasks"],
            "total_conversation_length": managed.total_message_count,
            "estimated_tokens": estimated_tokens,
        }

        if task_context:
            context["active_task"] = task_context

        return context

    def _estimate_tokens(self, context_window: Dict[str, Any]) -> int:
        """Rough token estimation."""
        total_chars = 0

        if context_window.get("context_summary"):
            total_chars += len(context_window["context_summary"])

        for msg in context_window.get("recent_messages", []):
            total_chars += len(str(msg.get("content", "")))

        # Rough estimate: 4 characters per token
        return total_chars // 4

    def format_context_for_prompt(self, context: Dict[str, Any]) -> str:
        """
        Format context into a string suitable for inclusion in prompts.

        Args:
            context: Context dictionary from build_context()

        Returns:
            Formatted context string
        """
        parts = []

        if context.get("conversation_summary"):
            parts.append("## Previous Conversation Summary")
            parts.append(context["conversation_summary"])
            parts.append("")

        if context.get("pending_tasks"):
            parts.append("## Pending Tasks")
            for task in context["pending_tasks"]:
                parts.append(f"- {task}")
            parts.append("")

        if context.get("recent_messages"):
            parts.append("## Recent Messages")
            for msg in context["recent_messages"]:
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")
            parts.append("")

        if context.get("active_task"):
            parts.append("## Active Task")
            task = context["active_task"]
            parts.append(f"Task: {task.get('description', 'Unknown')}")
            parts.append(f"Status: {task.get('status', 'Unknown')}")
            parts.append("")

        return "\n".join(parts)
