"""Performance monitoring for multi-agent orchestration."""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentExecutionMetrics:
    """Metrics for a single agent execution."""

    agent_id: str
    agent_type: str
    task_description: str
    start_time: float
    end_time: float
    duration_seconds: float
    status: str  # success, failed, timeout
    error: Optional[str] = None


@dataclass
class ParallelExecutionMetrics:
    """Metrics for parallel multi-agent execution."""

    request_id: str
    total_agents: int
    successful_agents: int
    failed_agents: int
    start_time: float
    end_time: float
    total_duration_seconds: float
    parallel_efficiency: float  # speedup ratio
    agent_metrics: List[AgentExecutionMetrics]

    def to_dict(self) -> Dict:
        """Convert to dictionary for metadata."""
        return {
            "request_id": self.request_id,
            "total_agents": self.total_agents,
            "successful_agents": self.successful_agents,
            "failed_agents": self.failed_agents,
            "total_duration_seconds": round(self.total_duration_seconds, 3),
            "parallel_efficiency": round(self.parallel_efficiency, 2),
            "agent_performance": [
                {
                    "agent_id": m.agent_id,
                    "agent_type": m.agent_type,
                    "duration_seconds": round(m.duration_seconds, 3),
                    "status": m.status,
                }
                for m in self.agent_metrics
            ],
        }


class PerformanceMonitor:
    """Monitor and track multi-agent execution performance."""

    def __init__(self):
        """Initialize performance monitor."""
        self.active_executions: Dict[str, float] = {}
        self.agent_timings: Dict[str, Dict[str, float]] = {}

    def start_parallel_execution(self, request_id: str) -> None:
        """
        Mark start of parallel execution.

        Args:
            request_id: Unique request identifier
        """
        self.active_executions[request_id] = time.time()
        self.agent_timings[request_id] = {}
        logger.info(f"Started monitoring parallel execution: {request_id}")

    def start_agent(self, request_id: str, agent_id: str) -> None:
        """
        Mark start of individual agent.

        Args:
            request_id: Request identifier
            agent_id: Agent identifier
        """
        if request_id not in self.agent_timings:
            self.agent_timings[request_id] = {}

        self.agent_timings[request_id][agent_id] = {"start": time.time(), "end": None}

    def end_agent(
        self, request_id: str, agent_id: str, status: str = "success"
    ) -> None:
        """
        Mark end of individual agent.

        Args:
            request_id: Request identifier
            agent_id: Agent identifier
            status: Execution status
        """
        if (
            request_id in self.agent_timings
            and agent_id in self.agent_timings[request_id]
        ):
            self.agent_timings[request_id][agent_id]["end"] = time.time()
            self.agent_timings[request_id][agent_id]["status"] = status

    def end_parallel_execution(
        self, request_id: str, agent_results: Dict[str, Dict]
    ) -> ParallelExecutionMetrics:
        """
        Mark end of parallel execution and calculate metrics.

        Args:
            request_id: Request identifier
            agent_results: Results from all agents

        Returns:
            Performance metrics
        """
        end_time = time.time()
        start_time = self.active_executions.get(request_id, end_time)
        total_duration = end_time - start_time

        # Build agent metrics
        agent_metrics = []
        total_agent_time = 0.0
        successful = 0
        failed = 0

        for agent_id, result_data in agent_results.items():
            agent_type = result_data.get("agent_type", "unknown")
            task_desc = result_data.get("task_description", "")
            status = result_data.get("status", "unknown")

            # Get timing
            timing = self.agent_timings.get(request_id, {}).get(agent_id, {})
            agent_start = timing.get("start", start_time)
            agent_end = timing.get("end", end_time)
            duration = agent_end - agent_start

            total_agent_time += duration

            if status == "success":
                successful += 1
            else:
                failed += 1

            agent_metrics.append(
                AgentExecutionMetrics(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    task_description=task_desc,
                    start_time=agent_start,
                    end_time=agent_end,
                    duration_seconds=duration,
                    status=status,
                    error=result_data.get("error"),
                )
            )

        # Calculate parallel efficiency (speedup)
        # Efficiency = Sequential Time / Parallel Time
        # Sequential time = sum of all agent durations
        # Parallel time = actual wall clock time
        parallel_efficiency = (
            total_agent_time / total_duration if total_duration > 0 else 1.0
        )

        metrics = ParallelExecutionMetrics(
            request_id=request_id,
            total_agents=len(agent_results),
            successful_agents=successful,
            failed_agents=failed,
            start_time=start_time,
            end_time=end_time,
            total_duration_seconds=total_duration,
            parallel_efficiency=parallel_efficiency,
            agent_metrics=agent_metrics,
        )

        logger.info(
            f"Parallel execution complete: {request_id} - "
            f"{successful}/{len(agent_results)} agents succeeded, "
            f"duration: {total_duration:.2f}s, "
            f"efficiency: {parallel_efficiency:.2f}x"
        )

        # Cleanup
        self.active_executions.pop(request_id, None)
        self.agent_timings.pop(request_id, None)

        return metrics

    def get_summary_stats(self, metrics: ParallelExecutionMetrics) -> str:
        """
        Generate human-readable summary.

        Args:
            metrics: Execution metrics

        Returns:
            Summary string
        """
        return (
            f"⚡ Parallel Execution Summary:\n"
            f"  • Agents: {metrics.successful_agents}/{metrics.total_agents} successful\n"
            f"  • Duration: {metrics.total_duration_seconds:.2f}s\n"
            f"  • Speedup: {metrics.parallel_efficiency:.2f}x faster than sequential\n"
            f"  • Efficiency: {(metrics.parallel_efficiency / metrics.total_agents * 100):.1f}%"
        )
