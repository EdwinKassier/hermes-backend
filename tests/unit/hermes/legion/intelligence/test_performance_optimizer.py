"""Unit tests for PerformanceOptimizer service."""

import time

import pytest

from app.hermes.legion.intelligence.performance_optimizer import PerformanceOptimizer


def test_execution_timing():
    optimizer = PerformanceOptimizer()

    optimizer.start_execution()
    time.sleep(0.1)

    elapsed = optimizer.get_elapsed_time()
    assert elapsed >= 0.1
    assert elapsed < 0.2


def test_worker_timing():
    optimizer = PerformanceOptimizer()

    optimizer.start_worker("worker_1")
    time.sleep(0.05)
    duration = optimizer.end_worker("worker_1")

    assert duration >= 0.05
    assert duration < 0.1


def test_timeout_detection():
    optimizer = PerformanceOptimizer()
    optimizer.set_timeout_threshold(0.1)

    optimizer.start_execution()

    # Should not timeout initially
    assert not optimizer.should_timeout()

    # Wait for timeout
    time.sleep(0.15)

    # Should timeout now
    assert optimizer.should_timeout()


def test_adjust_worker_count_increase():
    optimizer = PerformanceOptimizer()
    optimizer.start_execution()

    # Simulate slow execution with high complexity
    adjusted = optimizer.adjust_worker_count(
        current_count=3, complexity_score=0.8, elapsed_time=70.0
    )

    # Should increase workers
    assert adjusted == 4


def test_adjust_worker_count_no_change():
    optimizer = PerformanceOptimizer()
    optimizer.start_execution()

    # Normal execution
    adjusted = optimizer.adjust_worker_count(
        current_count=3, complexity_score=0.5, elapsed_time=30.0
    )

    # Should keep same
    assert adjusted == 3


def test_get_performance_metrics():
    optimizer = PerformanceOptimizer()
    optimizer.start_execution()
    optimizer.start_worker("worker_1")

    metrics = optimizer.get_performance_metrics()

    assert "elapsed_time" in metrics
    assert metrics["active_workers"] == 1
    assert "timeout_threshold" in metrics
