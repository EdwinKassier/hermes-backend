"""Unit tests for FeedbackLearner service."""

import pytest

from app.hermes.legion.intelligence.feedback_learner import (
    ExecutionFeedback,
    FeedbackLearner,
)


def test_record_execution():
    learner = FeedbackLearner(max_history=10)

    learner.record_execution(
        query="test query",
        strategy="intelligent",
        worker_count=2,
        complexity_score=0.6,
        execution_time=30.0,
        quality_metrics={"completeness": 0.9, "coherence": 0.8},
        success=True,
    )

    assert len(learner.execution_history) == 1
    assert learner.execution_history[0].query == "test query"
    assert learner.execution_history[0].worker_count == 2


def test_max_history_limit():
    learner = FeedbackLearner(max_history=3)

    # Add 5 executions
    for i in range(5):
        learner.record_execution(
            query=f"query {i}",
            strategy="intelligent",
            worker_count=2,
            complexity_score=0.5,
            execution_time=10.0,
            quality_metrics={"completeness": 0.8},
            success=True,
        )

    # Should only keep last 3
    assert len(learner.execution_history) == 3
    assert learner.execution_history[0].query == "query 2"


def test_get_optimal_worker_count_no_history():
    learner = FeedbackLearner()

    # Should return default heuristic
    assert learner.get_optimal_worker_count(0.2) == 1
    assert learner.get_optimal_worker_count(0.5) == 2
    assert learner.get_optimal_worker_count(0.8) == 3


def test_get_optimal_worker_count_with_history():
    learner = FeedbackLearner()

    # Add history showing 3 workers work best for complexity 0.7
    for _ in range(5):
        learner.record_execution(
            query="test",
            strategy="intelligent",
            worker_count=3,
            complexity_score=0.7,
            execution_time=20.0,
            quality_metrics={"completeness": 0.9},
            success=True,
        )

    # Add some with 2 workers performing worse
    for _ in range(3):
        learner.record_execution(
            query="test",
            strategy="intelligent",
            worker_count=2,
            complexity_score=0.7,
            execution_time=40.0,
            quality_metrics={"completeness": 0.6},
            success=True,
        )

    optimal = learner.get_optimal_worker_count(0.7)
    assert optimal == 3


def test_get_best_strategy():
    learner = FeedbackLearner()

    # Add history showing "council" performs better
    for _ in range(5):
        learner.record_execution(
            query="test",
            strategy="council",
            worker_count=3,
            complexity_score=0.5,
            execution_time=20.0,
            quality_metrics={"completeness": 0.9},
            success=True,
        )

    for _ in range(3):
        learner.record_execution(
            query="test",
            strategy="parallel",
            worker_count=2,
            complexity_score=0.5,
            execution_time=20.0,
            quality_metrics={"completeness": 0.6},
            success=True,
        )

    best = learner.get_best_strategy(0.5)
    assert best == "council"


def test_get_execution_stats():
    learner = FeedbackLearner()

    # Add mixed success/failure executions
    for i in range(7):
        learner.record_execution(
            query="test",
            strategy="intelligent",
            worker_count=2,
            complexity_score=0.5,
            execution_time=10.0,
            quality_metrics={"completeness": 0.8},
            success=True,
        )

    for i in range(3):
        learner.record_execution(
            query="test",
            strategy="intelligent",
            worker_count=2,
            complexity_score=0.5,
            execution_time=10.0,
            quality_metrics={"completeness": 0.2},
            success=False,
        )

    stats = learner.get_execution_stats()
    assert stats["total_executions"] == 10
    assert stats["success_rate"] == 0.7
    assert stats["avg_quality"] == 0.62  # (7*0.8 + 3*0.2) / 10
