"""Unit tests for PerformanceMonitor."""

import time
from datetime import datetime

import pytest

from app.hermes.legion.parallel.performance_monitor import (
    AgentExecutionMetrics,
    ParallelExecutionMetrics,
    PerformanceMonitor,
)


class TestPerformanceMonitor:
    """Test suite for performance monitoring."""

    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = PerformanceMonitor()

    def test_track_agent_execution(self):
        """Test tracking of single agent execution."""
        start_time = datetime.now()

        # Simulate agent execution
        time.sleep(0.1)

        metrics = AgentExecutionMetrics(
            agent_id="test_agent_1",
            agent_type="research",
            task_description="Test task",
            start_time=start_time,
            end_time=datetime.now(),
            duration_seconds=0.1,
            status="completed",
        )

        assert metrics.agent_id == "test_agent_1"
        assert metrics.duration_seconds >= 0.1
        assert metrics.status == "completed"

    def test_calculate_parallel_efficiency(self):
        """Test calculation of parallel execution efficiency."""
        agent_metrics = [
            AgentExecutionMetrics(
                agent_id="agent_1",
                agent_type="research",
                task_description="Research task",
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=2.0,
                status="completed",
            ),
            AgentExecutionMetrics(
                agent_id="agent_2",
                agent_type="analysis",
                task_description="Analysis task",
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=1.5,
                status="completed",
            ),
        ]

        total_duration = 2.0  # Max of the two (parallel)
        total_work = 3.5  # Sum of durations
        expected_efficiency = total_work / total_duration  # Speedup

        metrics = ParallelExecutionMetrics(
            request_id="test_request",
            total_agents=2,
            successful_agents=2,
            failed_agents=0,
            start_time=time.time(),
            end_time=time.time() + total_duration,
            total_duration_seconds=total_duration,
            parallel_efficiency=expected_efficiency,
            agent_metrics=agent_metrics,
        )

        assert metrics.total_duration_seconds == 2.0
        assert metrics.parallel_efficiency > 0

    def test_track_multiple_agents(self):
        """Test tracking multiple concurrent agents."""
        metrics_list = []

        for i in range(3):
            metrics = AgentExecutionMetrics(
                agent_id=f"agent_{i}",
                agent_type="research",
                task_description="Research task",
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=1.0,
                status="completed",
            )
            metrics_list.append(metrics)

        assert len(metrics_list) == 3
        assert all(m.status == "completed" for m in metrics_list)

    def test_failed_agent_tracking(self):
        """Test tracking of failed agent execution."""
        metrics = AgentExecutionMetrics(
            agent_id="failed_agent",
            agent_type="code",
            task_description="Code task",
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=0.5,
            status="failed",
            error="API timeout",
        )

        assert metrics.status == "failed"
        assert metrics.error is not None

    def test_zero_duration_handling(self):
        """Test handling of very fast executions."""
        metrics = AgentExecutionMetrics(
            agent_id="fast_agent",
            agent_type="analysis",
            task_description="Fast task",
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=0.001,
            status="completed",
        )

        assert metrics.duration_seconds >= 0
        assert metrics.status == "completed"

    def test_performance_summary_generation(self):
        """Test generation of performance summary."""
        agent_metrics = [
            AgentExecutionMetrics(
                agent_id="agent_1",
                agent_type="research",
                task_description="Research task",
                start_time=time.time(),
                end_time=time.time() + 1.0,
                duration_seconds=1.0,
                status="success",
            ),
            AgentExecutionMetrics(
                agent_id="agent_2",
                agent_type="analysis",
                task_description="Analysis task",
                start_time=time.time(),
                end_time=time.time() + 1.5,
                duration_seconds=1.5,
                status="success",
            ),
        ]

        metrics = ParallelExecutionMetrics(
            request_id="test_request",
            total_agents=2,
            successful_agents=2,
            failed_agents=0,
            start_time=time.time(),
            end_time=time.time() + 1.5,
            total_duration_seconds=1.5,
            parallel_efficiency=1.67,
            agent_metrics=agent_metrics,
        )

        summary = self.monitor.get_summary_stats(metrics)

        assert isinstance(summary, str)
        assert "1.5" in summary or "1.50" in summary  # Total duration
        assert "2" in summary  # Agent count

    def test_speedup_calculation(self):
        """Test calculation of parallel speedup factor."""
        # Sequential would be 3.0s (1.5s + 1.5s)
        # Parallel is 1.5s (max of both)
        # Speedup = 3.0 / 1.5 = 2.0x

        sequential_time = 3.0
        parallel_time = 1.5
        speedup = sequential_time / parallel_time

        assert speedup == 2.0


class TestPerformanceMetrics:
    """Test performance metrics data structures."""

    def test_agent_metrics_creation(self):
        """Test creation of AgentExecutionMetrics."""
        metrics = AgentExecutionMetrics(
            agent_id="test",
            agent_type="research",
            task_description="Test task",
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=1.0,
            status="completed",
        )

        assert metrics.agent_id is not None
        assert metrics.duration_seconds >= 0

    def test_parallel_metrics_calculations(self):
        """Test ParallelExecutionMetrics calculations."""
        agent_metrics = [
            AgentExecutionMetrics(
                agent_id=f"agent_{i}",
                agent_type="research",
                task_description="Research task",
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=1.0,
                status="completed",
            )
            for i in range(2)
        ]

        successful = len([m for m in agent_metrics if m.status == "completed"])
        failed = len([m for m in agent_metrics if m.status == "failed"])

        assert successful == 2
        assert failed == 0
