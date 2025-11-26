"""
Performance Optimizer Service.

Monitors and optimizes execution performance in real-time.
"""

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """
    Optimizes performance by monitoring execution metrics and
    making runtime adjustments.
    """

    def __init__(self):
        self.execution_start_time: Optional[float] = None
        self.worker_start_times: Dict[str, float] = {}
        self.timeout_threshold = 120.0  # 2 minutes default timeout

    def start_execution(self) -> None:
        """Mark the start of an execution."""
        self.execution_start_time = time.time()
        self.worker_start_times = {}
        logger.info("Performance monitoring started")

    def start_worker(self, worker_id: str) -> None:
        """Mark the start of a worker execution."""
        self.worker_start_times[worker_id] = time.time()

    def end_worker(self, worker_id: str) -> float:
        """Mark the end of worker execution and return duration."""
        if worker_id not in self.worker_start_times:
            logger.warning(f"Worker {worker_id} ended without start time")
            return 0.0

        duration = time.time() - self.worker_start_times[worker_id]
        logger.info(f"Worker {worker_id} completed in {duration:.2f}s")
        return duration

    def get_elapsed_time(self) -> float:
        """Get total elapsed time since execution start."""
        if self.execution_start_time is None:
            return 0.0
        return time.time() - self.execution_start_time

    def should_timeout(self) -> bool:
        """Check if execution should timeout."""
        elapsed = self.get_elapsed_time()
        if elapsed > self.timeout_threshold:
            logger.warning(f"Execution timeout after {elapsed:.2f}s")
            return True
        return False

    def adjust_worker_count(
        self, current_count: int, complexity_score: float, elapsed_time: float
    ) -> int:
        """
        Dynamically adjust worker count based on performance.

        This is called mid-execution if we detect slow progress.
        """
        # If we're taking too long and complexity is high, add workers
        if elapsed_time > 60 and complexity_score > 0.7 and current_count < 5:
            logger.info(
                f"Performance optimization: increasing workers from {current_count} to {current_count + 1}"
            )
            return current_count + 1

        # If we're very fast and complexity is low, we could reduce
        # (though this is less common mid-execution)
        if elapsed_time < 10 and complexity_score < 0.3 and current_count > 1:
            logger.info(
                f"Performance optimization: reducing workers from {current_count} to {current_count - 1}"
            )
            return current_count - 1

        return current_count

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            "elapsed_time": self.get_elapsed_time(),
            "active_workers": len(self.worker_start_times),
            "timeout_threshold": self.timeout_threshold,
            "should_timeout": self.should_timeout(),
        }

    def set_timeout_threshold(self, seconds: float) -> None:
        """Set the timeout threshold."""
        self.timeout_threshold = seconds
        logger.info(f"Timeout threshold set to {seconds}s")
