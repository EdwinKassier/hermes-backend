"""
Feedback Learning Service.

Tracks execution history and learns from past performance to improve
future orchestration decisions.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExecutionFeedback:
    """Stores feedback from a single execution."""

    def __init__(
        self,
        query: str,
        strategy: str,
        worker_count: int,
        complexity_score: float,
        execution_time: float,
        quality_metrics: Dict[str, float],
        success: bool,
        timestamp: Optional[datetime] = None,
    ):
        self.query = query
        self.strategy = strategy
        self.worker_count = worker_count
        self.complexity_score = complexity_score
        self.execution_time = execution_time
        self.quality_metrics = quality_metrics
        self.success = success
        self.timestamp = timestamp or datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "strategy": self.strategy,
            "worker_count": self.worker_count,
            "complexity_score": self.complexity_score,
            "execution_time": self.execution_time,
            "quality_metrics": self.quality_metrics,
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
        }


class FeedbackLearner:
    """
    Learns from execution history to improve future decisions.

    Tracks:
    - Which strategies work best for different query types
    - Optimal worker counts for different complexity levels
    - Tool effectiveness
    - Common failure patterns
    """

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.execution_history: List[ExecutionFeedback] = []

    def record_execution(
        self,
        query: str,
        strategy: str,
        worker_count: int,
        complexity_score: float,
        execution_time: float,
        quality_metrics: Dict[str, float],
        success: bool = True,
    ) -> None:
        """Record feedback from an execution."""
        feedback = ExecutionFeedback(
            query=query,
            strategy=strategy,
            worker_count=worker_count,
            complexity_score=complexity_score,
            execution_time=execution_time,
            quality_metrics=quality_metrics,
            success=success,
        )

        self.execution_history.append(feedback)

        # Keep history bounded
        if len(self.execution_history) > self.max_history:
            self.execution_history = self.execution_history[-self.max_history :]

        logger.info(
            f"Recorded execution feedback: strategy={strategy}, quality={quality_metrics.get('completeness', 0):.2f}"
        )

    def get_optimal_worker_count(self, complexity_score: float) -> int:
        """
        Suggest optimal worker count based on historical performance.
        """
        if not self.execution_history:
            # Default heuristic
            if complexity_score < 0.3:
                return 1
            elif complexity_score < 0.7:
                return 2
            else:
                return 3

        # Find similar complexity executions
        similar_executions = [
            e
            for e in self.execution_history
            if abs(e.complexity_score - complexity_score) < 0.2 and e.success
        ]

        if not similar_executions:
            # Fallback to default
            if complexity_score < 0.3:
                return 1
            elif complexity_score < 0.7:
                return 2
            else:
                return 3

        # Find worker count with best quality/time ratio
        worker_performance = {}
        for exec in similar_executions:
            count = exec.worker_count
            quality = exec.quality_metrics.get("completeness", 0.5)
            time = exec.execution_time

            # Score = quality / (time in minutes)
            score = quality / max(time / 60, 0.1)

            if count not in worker_performance:
                worker_performance[count] = []
            worker_performance[count].append(score)

        # Average scores and pick best
        avg_scores = {
            count: sum(scores) / len(scores)
            for count, scores in worker_performance.items()
        }

        optimal_count = max(avg_scores.keys(), key=lambda k: avg_scores[k])
        logger.info(
            f"Learned optimal worker count for complexity {complexity_score:.2f}: {optimal_count}"
        )
        return optimal_count

    def get_best_strategy(self, complexity_score: float) -> str:
        """
        Recommend best strategy based on historical performance.
        """
        if not self.execution_history:
            return "intelligent"

        # Find similar complexity executions
        similar_executions = [
            e
            for e in self.execution_history
            if abs(e.complexity_score - complexity_score) < 0.2 and e.success
        ]

        if not similar_executions:
            return "intelligent"

        # Calculate average quality per strategy
        strategy_performance = {}
        for exec in similar_executions:
            strategy = exec.strategy
            quality = exec.quality_metrics.get("completeness", 0.5)

            if strategy not in strategy_performance:
                strategy_performance[strategy] = []
            strategy_performance[strategy].append(quality)

        avg_quality = {
            strategy: sum(scores) / len(scores)
            for strategy, scores in strategy_performance.items()
        }

        best_strategy = max(avg_quality.keys(), key=lambda k: avg_quality[k])
        logger.info(
            f"Learned best strategy for complexity {complexity_score:.2f}: {best_strategy}"
        )
        return best_strategy

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get statistics about execution history."""
        if not self.execution_history:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "avg_execution_time": 0.0,
                "avg_quality": 0.0,
            }

        total = len(self.execution_history)
        successes = sum(1 for e in self.execution_history if e.success)
        avg_time = sum(e.execution_time for e in self.execution_history) / total
        avg_quality = (
            sum(
                e.quality_metrics.get("completeness", 0) for e in self.execution_history
            )
            / total
        )

        return {
            "total_executions": total,
            "success_rate": successes / total,
            "avg_execution_time": avg_time,
            "avg_quality": avg_quality,
            "strategy_distribution": self._get_strategy_distribution(),
        }

    def _get_strategy_distribution(self) -> Dict[str, int]:
        """Count executions per strategy."""
        distribution = {}
        for exec in self.execution_history:
            distribution[exec.strategy] = distribution.get(exec.strategy, 0) + 1
        return distribution
