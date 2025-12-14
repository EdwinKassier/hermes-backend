"""Human-readable decision rationale formatter."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DecisionRationaleFormatter:
    """
    Formats decision rationale into human-readable explanations.

    Transforms technical decision data into clear, understandable summaries.
    """

    @staticmethod
    def format_for_user(decision_rationale: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a user-friendly summary of decision rationale.

        Args:
            decision_rationale: Raw decision rationale from orchestrator

        Returns:
            Human-readable summary with explanations
        """
        if not decision_rationale:
            return {"summary": "No decisions made yet", "decisions": []}

        # Get the most recent decision
        latest = decision_rationale[-1]

        # Create simple summary
        summary = DecisionRationaleFormatter._create_summary(latest)

        # Create step-by-step explanation
        steps = DecisionRationaleFormatter._create_step_breakdown(decision_rationale)

        # Create quick facts
        quick_facts = DecisionRationaleFormatter._create_quick_facts(latest)

        return {
            "summary": summary,
            "quick_facts": quick_facts,
            "step_by_step": steps,
            "technical_details": decision_rationale,  # Keep original for debugging
        }

    @staticmethod
    def _create_summary(decision: Dict[str, Any]) -> str:
        """Create one-sentence summary of decision."""
        analysis = decision.get("analysis", {})
        decisions = decision.get("decisions", {})
        reasoning = decision.get("reasoning", {})

        # Check for multi-agent
        if analysis.get("multi_agent_task_detected"):
            return "Detected a complex task requiring multiple specialized agents working together"

        # Check if agent needed
        agent_needed = decisions.get("agent_needed", False)

        if not agent_needed:
            return "Orchestrator agent determined simple question and is providing the answer directly using knowledge base"

        # Agent needed
        agent_type = decisions.get("agent_type") or decisions.get("selected_task_type")
        action = decisions.get("action", "unknown")

        if action == "gather_info":
            return f"Need more details to complete this {agent_type} task"
        elif action == "execute_agent":
            return f"Activating {agent_type} specialist to handle this request"
        elif action == "parallel_orchestrate":
            return "Breaking down into subtasks for parallel processing"
        else:
            return f"Processing {agent_type} request"

    @staticmethod
    def _create_quick_facts(decision: Dict[str, Any]) -> Dict[str, Any]:
        """Create quick facts about the decision."""
        analysis = decision.get("analysis", {})
        decisions = decision.get("decisions", {})

        facts = {}

        # What was asked
        user_msg = analysis.get("user_message", "")
        if user_msg:
            facts["Request"] = user_msg[:100] + ("..." if len(user_msg) > 100 else "")

        # Task type
        task_type = analysis.get("identified_task_type")
        if task_type:
            # Dynamic agent system uses custom agent types - display as-is
            facts["Task Type"] = task_type.replace("_", " ").title()

        # Agent info
        if decisions.get("agent_needed"):
            agent_type = decisions.get("agent_type") or decisions.get(
                "selected_task_type"
            )
            agent_id = decisions.get("agent_id", "")
            if agent_type:
                facts["Specialist"] = f"{agent_type.title()} Agent" + (
                    f" ({agent_id})" if agent_id else ""
                )

        # Tools
        tool_alloc = analysis.get("tool_allocation", {})
        tools = tool_alloc.get("tools_allocated", [])
        if tools:
            facts["Tools"] = ", ".join(tools[:3]) + (
                f" (+{len(tools)-3} more)" if len(tools) > 3 else ""
            )

        # Multi-agent
        if analysis.get("multi_agent_task_detected"):
            subtasks = analysis.get("subtasks_count", "multiple")
            facts["Mode"] = f"Parallel Execution ({subtasks} agents)"

        # Action
        action = decisions.get("action", "")
        action_map = {
            "execute_agent": "Ready to Execute",
            "gather_info": "Needs More Information",
            "complete": "Direct Response",
            "parallel_orchestrate": "Multi-Agent Coordination",
        }
        if action:
            facts["Status"] = action_map.get(
                action, f"{action.replace('_', ' ').title()}"
            )

        return facts

    @staticmethod
    def _create_step_breakdown(decisions: List[Dict[str, Any]]) -> List[str]:
        """Create step-by-step explanation."""
        steps = []

        for i, decision in enumerate(decisions, 1):
            analysis = decision.get("analysis", {})
            decs = decision.get("decisions", {})
            reasoning = decision.get("reasoning", {})

            step = f"Step {i}: "

            # Determine what happened
            if analysis.get("multi_agent_task_detected"):
                step += "Orchestrator agent detected multi-part request - planning parallel execution"
            elif decs.get("agent_needed") == False:
                step += "Orchestrator agent determined simple question and is providing the answer directly from knowledge base"
            elif decs.get("action") == "gather_info":
                agent = decs.get("agent_type", "specialist")
                step += (
                    f"{agent.title()} agent needs clarification - asking for details"
                )
            elif decs.get("action") == "execute_agent":
                agent = decs.get("agent_type", "specialist")
                step += f"{agent.title()} agent activated - processing request"
            elif decs.get("action") == "parallel_execute":
                step += "Running multiple agents simultaneously - gathering results"
            elif decs.get("action") == "synthesize":
                step += "Combining results from all agents - creating final response"
            else:
                step += f"{decs.get('action', 'processing').replace('_', ' ').title()}"

            steps.append(step)

        return steps

    @staticmethod
    def format_as_markdown(formatted_rationale: Dict[str, Any]) -> str:
        """
        Convert formatted rationale to markdown string.

        Args:
            formatted_rationale: Output from format_for_user()

        Returns:
            Markdown-formatted explanation
        """
        md = []

        # Summary
        md.append("# 🧠 Decision Explanation\n")
        md.append(f"**{formatted_rationale['summary']}**\n")

        # Quick Facts
        if formatted_rationale.get("quick_facts"):
            md.append("\n## 📋 Quick Facts\n")
            for key, value in formatted_rationale["quick_facts"].items():
                md.append(f"- **{key}**: {value}")
            md.append("")

        # Step by step
        if formatted_rationale.get("step_by_step"):
            md.append("\n## 🔍 What Happened\n")
            for step in formatted_rationale["step_by_step"]:
                md.append(step)
                md.append("")

        return "\n".join(md)

    @staticmethod
    def format_as_text(formatted_rationale: Dict[str, Any]) -> str:
        """
        Convert formatted rationale to plain text.

        Args:
            formatted_rationale: Output from format_for_user()

        Returns:
            Plain text explanation
        """
        lines = []

        # Summary
        lines.append("DECISION EXPLANATION")
        lines.append("=" * 50)
        lines.append(formatted_rationale["summary"])
        lines.append("")

        # Quick Facts
        if formatted_rationale.get("quick_facts"):
            lines.append("QUICK FACTS")
            lines.append("-" * 50)
            for key, value in formatted_rationale["quick_facts"].items():
                # Remove emojis for plain text
                clean_value = value
                for emoji in ["🔍", "💻", "📊", "📁", "📋", "⚡", "✅", "❓", "✓", "→"]:
                    clean_value = clean_value.replace(emoji, "")
                lines.append(f"{key}: {clean_value.strip()}")
            lines.append("")

        # Steps
        if formatted_rationale.get("step_by_step"):
            lines.append("WHAT HAPPENED")
            lines.append("-" * 50)
            for step in formatted_rationale["step_by_step"]:
                # Remove markdown and emojis
                clean_step = step.replace("**", "").replace("→", "->")
                for emoji in ["🤝", "💬", "❓", "🤖", "⚡", "🔄"]:
                    clean_step = clean_step.replace(emoji, "")
                lines.append(clean_step.strip())
            lines.append("")

        return "\n".join(lines)
