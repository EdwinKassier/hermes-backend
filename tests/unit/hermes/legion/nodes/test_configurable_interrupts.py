"""Unit tests for configurable interrupt logic."""

from unittest.mock import MagicMock

import pytest

from app.hermes.legion.nodes.legion_orchestrator import _should_interrupt_for_approval


@pytest.mark.unit
class TestConfigurableInterrupts:
    """Test suite for interrupt threshold logic."""

    def test_auto_approve_enabled(self):
        """Test that auto_approve metadata skips interrupt."""
        state = {"metadata": {"auto_approve": True}}
        workers = [{"worker_id": "w1"}, {"worker_id": "w2"}]

        result = _should_interrupt_for_approval(state, workers)

        assert result == False  # Should not interrupt

    def test_complexity_threshold_met(self):
        """Test that >3 workers triggers interrupt."""
        state = {"metadata": {}}
        workers = [
            {"worker_id": "w1"},
            {"worker_id": "w2"},
            {"worker_id": "w3"},
            {"worker_id": "w4"},
        ]

        result = _should_interrupt_for_approval(state, workers)

        assert result == True  # Should interrupt

    def test_complexity_threshold_not_met(self):
        """Test that <= 3 workers does not trigger interrupt."""
        state = {"metadata": {}}
        workers = [{"worker_id": "w1"}, {"worker_id": "w2"}]

        result = _should_interrupt_for_approval(state, workers)

        assert result == False  # Should not interrupt

    def test_high_risk_level(self):
        """Test that HIGH risk level triggers interrupt."""
        state = {"metadata": {}}
        workers = [{"worker_id": "w1"}]
        routing_decision = {"risk_level": "HIGH"}

        result = _should_interrupt_for_approval(state, workers, routing_decision)

        assert result == True  # Should interrupt

    def test_critical_risk_level(self):
        """Test that CRITICAL risk level triggers interrupt."""
        state = {"metadata": {}}
        workers = [{"worker_id": "w1"}]
        routing_decision = {"risk_level": "CRITICAL"}

        result = _should_interrupt_for_approval(state, workers, routing_decision)

        assert result == True  # Should interrupt

    def test_low_risk_level(self):
        """Test that LOW risk level does not trigger interrupt."""
        state = {"metadata": {}}
        workers = [{"worker_id": "w1"}]
        routing_decision = {"risk_level": "LOW"}

        result = _should_interrupt_for_approval(state, workers, routing_decision)

        assert result == False  # Should not interrupt

    def test_cost_threshold_met(self):
        """Test that estimated cost > $1.00 triggers interrupt."""
        state = {"metadata": {}}
        workers = [{"worker_id": "w1"}]
        routing_decision = {"risk_level": "LOW", "estimated_cost": 1.50}

        result = _should_interrupt_for_approval(state, workers, routing_decision)

        assert result == True  # Should interrupt

    def test_cost_threshold_not_met(self):
        """Test that estimated cost <= $1.00 does not trigger interrupt."""
        state = {"metadata": {}}
        workers = [{"worker_id": "w1"}]
        routing_decision = {"risk_level": "LOW", "estimated_cost": 0.50}

        result = _should_interrupt_for_approval(state, workers, routing_decision)

        assert result == False  # Should not interrupt

    def test_routing_recommends_approval(self):
        """Test that routing intelligence recommendation triggers interrupt."""
        state = {"metadata": {}}
        workers = [{"worker_id": "w1"}]
        routing_decision = {
            "risk_level": "LOW",
            "should_seek_approval": True,
            "approval_reason": "Sensitive data access required",
        }

        result = _should_interrupt_for_approval(state, workers, routing_decision)

        assert result == True  # Should interrupt

    def test_auto_approve_overrides_all(self):
        """Test that auto_approve overrides all other thresholds."""
        state = {"metadata": {"auto_approve": True}}
        workers = [{"worker_id": f"w{i}"} for i in range(5)]  # Many workers
        routing_decision = {
            "risk_level": "CRITICAL",
            "estimated_cost": 10.0,
            "should_seek_approval": True,
        }

        result = _should_interrupt_for_approval(state, workers, routing_decision)

        assert result == False  # Should not interrupt (auto-approve wins)

    def test_multiple_thresholds(self):
        """Test that any threshold can trigger interrupt."""
        # Test with only cost threshold
        state1 = {"metadata": {}}
        workers1 = [{"worker_id": "w1"}]
        routing1 = {"risk_level": "LOW", "estimated_cost": 2.0}

        assert _should_interrupt_for_approval(state1, workers1, routing1) == True

        # Test with only complexity threshold
        state2 = {"metadata": {}}
        workers2 = [{"worker_id": f"w{i}"} for i in range(4)]

        assert _should_interrupt_for_approval(state2, workers2, None) == True

        # Test with only risk threshold
        state3 = {"metadata": {}}
        workers3 = [{"worker_id": "w1"}]
        routing3 = {"risk_level": "HIGH"}

        assert _should_interrupt_for_approval(state3, workers3, routing3) == True
