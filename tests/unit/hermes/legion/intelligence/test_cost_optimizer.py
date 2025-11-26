"""Unit tests for CostOptimizer service."""

import pytest

from app.hermes.legion.intelligence.cost_optimizer import CostOptimizer


def test_cost_tracking():
    optimizer = CostOptimizer(budget_per_query=0.10)
    optimizer.reset()

    optimizer.record_llm_call("test", input_tokens=1000, output_tokens=500)

    # Cost should be: (1000/1000 * 0.0015) + (500/1000 * 0.002) = 0.0015 + 0.001 = 0.0025
    assert optimizer.current_cost == pytest.approx(0.0025, rel=0.01)
    assert optimizer.llm_calls == 1


def test_budget_check():
    optimizer = CostOptimizer(budget_per_query=0.05)
    optimizer.reset()

    # Add some cost
    optimizer.record_llm_call("test", input_tokens=10000, output_tokens=5000)

    # Cost should be: (10 * 0.0015) + (5 * 0.002) = 0.015 + 0.01 = 0.025
    assert optimizer.current_cost == pytest.approx(0.025, rel=0.01)

    # Should still be within budget
    assert optimizer.is_within_budget()

    # Add more cost to exceed budget
    optimizer.record_llm_call("test", input_tokens=10000, output_tokens=10000)

    # Should exceed budget now
    assert not optimizer.is_within_budget()


def test_worker_count_reduction():
    optimizer = CostOptimizer(budget_per_query=0.10)
    optimizer.reset()

    # Use most of budget
    optimizer.record_llm_call("test", input_tokens=30000, output_tokens=20000)

    # Should reduce worker count
    adjusted = optimizer.should_reduce_workers(5)
    assert adjusted < 5


def test_cheaper_model_recommendation():
    optimizer = CostOptimizer(budget_per_query=0.10)
    optimizer.reset()

    # Use less than 75% of budget
    optimizer.record_llm_call("test", input_tokens=10000, output_tokens=10000)
    assert not optimizer.should_use_cheaper_model()

    # Use more than 75% of budget
    optimizer.record_llm_call("test", input_tokens=30000, output_tokens=20000)
    assert optimizer.should_use_cheaper_model()


def test_get_cost_summary():
    optimizer = CostOptimizer(budget_per_query=0.10)
    optimizer.reset()

    optimizer.record_llm_call("test1", input_tokens=1000, output_tokens=500)
    optimizer.record_llm_call("test2", input_tokens=1000, output_tokens=500)

    summary = optimizer.get_cost_summary()

    assert summary["llm_calls"] == 2
    assert summary["budget"] == 0.10
    assert summary["within_budget"]
    assert "cost_per_call" in summary


def test_budget_setting():
    optimizer = CostOptimizer()
    optimizer.set_budget(0.50)

    assert optimizer.budget_per_query == 0.50

    # Even after high cost, should be within new budget
    optimizer.record_llm_call("test", input_tokens=50000, output_tokens=30000)
    assert optimizer.is_within_budget()
