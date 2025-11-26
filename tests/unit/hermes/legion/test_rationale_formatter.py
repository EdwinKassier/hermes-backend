"""Unit tests for DecisionRationaleFormatter."""

import pytest

from app.hermes.legion.utils.rationale_formatter import DecisionRationaleFormatter


class TestDecisionRationaleFormatter:
    """Test suite for decision rationale formatting."""

    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = DecisionRationaleFormatter()

    def test_format_simple_question_decision(self):
        """Test formatting of simple question decision."""
        decision_rationale = [
            {
                "timestamp": "2025-11-18T10:00:00",
                "node": "orchestrator",
                "analysis": {
                    "user_message": "What is AI?",
                    "identified_task_type": None,
                },
                "decisions": {"agent_needed": False, "action": "complete"},
                "reasoning": {},
            }
        ]

        result = self.formatter.format_for_user(decision_rationale)

        assert isinstance(result, dict)
        assert "summary" in result
        assert "quick_facts" in result
        assert "step_by_step" in result
        assert "knowledge base" in result["summary"].lower()

    def test_format_agent_execution_decision(self):
        """Test formatting of agent execution decision."""
        decision_rationale = [
            {
                "timestamp": "2025-11-18T10:00:00",
                "node": "orchestrator",
                "analysis": {
                    "user_message": "Write code to sort list",
                    "identified_task_type": "code",
                },
                "decisions": {
                    "agent_needed": True,
                    "agent_type": "code",
                    "action": "execute_agent",
                },
                "reasoning": {},
            }
        ]

        result = self.formatter.format_for_user(decision_rationale)

        assert "code" in result["summary"].lower()
        assert "quick_facts" in result
        assert "Task Type" in result["quick_facts"]

    def test_format_multi_agent_decision(self):
        """Test formatting of multi-agent decision."""
        decision_rationale = [
            {
                "timestamp": "2025-11-18T10:00:00",
                "node": "orchestrator",
                "analysis": {
                    "user_message": "Research AI and analyze trends",
                    "multi_agent_task_detected": True,
                    "subtasks_count": 2,
                },
                "decisions": {"action": "parallel_orchestrate"},
                "reasoning": {},
            }
        ]

        result = self.formatter.format_for_user(decision_rationale)

        assert (
            "multiple" in result["summary"].lower()
            or "agents" in result["summary"].lower()
        )
        assert "Mode" in result["quick_facts"] or "Parallel" in str(
            result["quick_facts"]
        )

    def test_no_emojis_in_output(self):
        """Test that output contains no emojis."""
        decision_rationale = [
            {
                "analysis": {"user_message": "test"},
                "decisions": {"agent_needed": False, "action": "complete"},
                "reasoning": {},
            }
        ]

        result = self.formatter.format_for_user(decision_rationale)

        # Check for common emoji characters
        summary = result.get("summary", "")
        quick_facts = str(result.get("quick_facts", ""))
        steps = str(result.get("step_by_step", ""))

        emoji_chars = ["✅", "🤖", "💻", "📊", "🔍", "❓", "⚡", "💬", "🤝"]
        full_text = summary + quick_facts + steps

        for emoji in emoji_chars:
            assert emoji not in full_text, f"Found emoji {emoji} in output"

    def test_quick_facts_structure(self):
        """Test structure of quick facts."""
        decision_rationale = [
            {
                "analysis": {
                    "user_message": "Write Python code",
                    "identified_task_type": "code",
                },
                "decisions": {
                    "agent_needed": True,
                    "agent_type": "code",
                    "action": "execute_agent",
                },
                "reasoning": {},
            }
        ]

        result = self.formatter.format_for_user(decision_rationale)
        facts = result["quick_facts"]

        assert isinstance(facts, dict)
        # Should have some of these keys
        possible_keys = [
            "Request",
            "Task Type",
            "Specialist",
            "Tools",
            "Status",
            "Mode",
        ]
        assert any(key in facts for key in possible_keys)

    def test_step_by_step_breakdown(self):
        """Test step-by-step breakdown."""
        decision_rationale = [
            {
                "analysis": {"identified_task_type": "research"},
                "decisions": {
                    "agent_needed": True,
                    "action": "execute_agent",
                    "agent_type": "research",
                },
                "reasoning": {},
            }
        ]

        result = self.formatter.format_for_user(decision_rationale)
        steps = result["step_by_step"]

        assert isinstance(steps, list)
        assert len(steps) > 0
        assert steps[0].startswith("Step 1:")

    def test_empty_decision_rationale(self):
        """Test handling of empty decision rationale."""
        result = self.formatter.format_for_user([])

        assert isinstance(result, dict)
        # Should return some default structure
        assert "summary" in result or "error" in result

    def test_multiple_decisions(self):
        """Test formatting of multiple sequential decisions."""
        decision_rationale = [
            {"analysis": {}, "decisions": {"action": "gather_info"}, "reasoning": {}},
            {"analysis": {}, "decisions": {"action": "execute_agent"}, "reasoning": {}},
        ]

        result = self.formatter.format_for_user(decision_rationale)
        steps = result.get("step_by_step", [])

        # Should have steps for each decision
        assert len(steps) >= 2


class TestFormatterEdgeCases:
    """Test edge cases in formatter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = DecisionRationaleFormatter()

    def test_missing_fields_in_decision(self):
        """Test handling of missing fields."""
        decision_rationale = [{"analysis": {}, "decisions": {}, "reasoning": {}}]

        result = self.formatter.format_for_user(decision_rationale)

        # Should handle gracefully without crashing
        assert isinstance(result, dict)

    def test_none_values(self):
        """Test handling of None values."""
        decision_rationale = [
            {
                "analysis": {"user_message": None},
                "decisions": {"agent_type": None},
                "reasoning": None,
            }
        ]

        result = self.formatter.format_for_user(decision_rationale)

        # Should not crash
        assert isinstance(result, dict)
