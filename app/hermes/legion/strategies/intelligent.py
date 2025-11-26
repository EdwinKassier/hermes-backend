"""
Intelligent Strategy Implementation with Advanced Features.

This strategy uses all intelligence services including Phase 4 optimizations
to create an adaptive, self-optimizing orchestration approach.
"""

import logging
from typing import Any, Dict, List

from app.shared.utils.toolhub import get_all_tools

from ..intelligence.adaptive_synthesizer import AdaptiveSynthesizer
from ..intelligence.cost_optimizer import CostOptimizer
from ..intelligence.feedback_learner import FeedbackLearner
from ..intelligence.performance_optimizer import PerformanceOptimizer
from ..intelligence.query_analyzer import QueryAnalyzer
from ..intelligence.tool_intelligence import ToolIntelligence
from ..intelligence.worker_planner import IntelligentWorkerPlanner
from .base import LegionStrategy

logger = logging.getLogger(__name__)

# Global instances for learning across executions
_global_feedback_learner = FeedbackLearner(max_history=1000)


class IntelligentStrategy(LegionStrategy):
    """
    Intelligent strategy: Uses AI-powered analysis to determine
    optimal worker composition, tool allocation, and synthesis approach.

    Phase 4 Enhancements:
    - Learns from execution history
    - Monitors and optimizes performance in real-time
    - Manages costs to stay within budget
    """

    def __init__(self):
        # Core intelligence services (Phase 2)
        self.query_analyzer = QueryAnalyzer()
        self.worker_planner = IntelligentWorkerPlanner()
        self.tool_intelligence = ToolIntelligence()
        self.adaptive_synthesizer = AdaptiveSynthesizer()

        # Advanced optimization services (Phase 4)
        self.feedback_learner = _global_feedback_learner  # Shared across instances
        self.performance_optimizer = PerformanceOptimizer()
        self.cost_optimizer = CostOptimizer()

    async def generate_workers(
        self, query: str, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate workers using intelligent planning with Phase 4 optimizations."""
        try:
            # Start performance monitoring
            self.performance_optimizer.start_execution()
            self.cost_optimizer.reset()

            # 1. Analyze query complexity
            complexity = await self.query_analyzer.analyze_complexity(query)
            logger.info(
                f"Query complexity: {complexity.score}, suggested workers: {complexity.suggested_workers}"
            )

            # Record LLM call cost
            self.cost_optimizer.record_llm_call(
                "complexity_analysis", input_tokens=200, output_tokens=100
            )

            # 2. Learn from history - get optimal worker count
            learned_worker_count = self.feedback_learner.get_optimal_worker_count(
                complexity.score
            )

            # Use learned value if available, otherwise use AI suggestion
            suggested_count = (
                learned_worker_count
                if len(self.feedback_learner.execution_history) > 10
                else complexity.suggested_workers
            )

            # 3. Apply cost optimization
            optimized_count = self.cost_optimizer.should_reduce_workers(suggested_count)
            logger.info(
                f"Worker count: suggested={suggested_count}, optimized={optimized_count}"
            )

            # 4. Plan workers (with optimized count)
            # Override complexity.suggested_workers for the planner
            complexity.suggested_workers = optimized_count

            worker_plans = await self.worker_planner.plan_workers(
                query=query,
                complexity=complexity,
                constraints=context.get("constraints"),
            )

            self.cost_optimizer.record_llm_call(
                "worker_planning", input_tokens=300, output_tokens=200
            )

            # 5. Allocate tools intelligently
            available_tools = get_all_tools()

            workers = []
            for plan in worker_plans:
                # Get intelligent tool recommendations
                recommended_tools = await self.tool_intelligence.recommend_tools(
                    role=plan.role,
                    task=plan.task_description,
                    available_tools=available_tools,
                )

                self.cost_optimizer.record_llm_call(
                    "tool_recommendation", input_tokens=150, output_tokens=50
                )

                workers.append(
                    {
                        "worker_id": plan.worker_id,
                        "role": plan.role,
                        "task_description": plan.task_description,
                        "tools": recommended_tools,
                        "metadata": {
                            "specialization": plan.specialization,
                            "priority": plan.priority,
                            "estimated_duration": plan.estimated_duration,
                            "complexity_score": complexity.score,
                            # Phase 4: Add optimization metadata
                            "learned_from_history": len(
                                self.feedback_learner.execution_history
                            )
                            > 0,
                            "cost_optimized": optimized_count < suggested_count,
                            "performance_monitoring": True,
                        },
                    }
                )

            logger.info(
                f"Generated {len(workers)} intelligent workers with Phase 4 optimizations"
            )
            logger.info(f"Cost summary: {self.cost_optimizer.get_cost_summary()}")

            return workers

        except Exception as e:
            logger.error(f"Error in intelligent worker generation: {e}")
            # Fallback to single worker
            return [
                {
                    "worker_id": "fallback_worker",
                    "role": "general",
                    "task_description": query,
                    "tools": [],
                }
            ]

    async def synthesize_results(
        self, original_query: str, results: Dict[str, Any], persona: str
    ) -> str:
        """Synthesize results using adaptive synthesis with Phase 4 tracking."""
        try:
            # 1. Assess quality
            quality = await self.adaptive_synthesizer.assess_result_quality(results)
            logger.info(
                f"Result quality - completeness: {quality.completeness}, confidence: {quality.confidence}"
            )

            self.cost_optimizer.record_llm_call(
                "quality_assessment", input_tokens=400, output_tokens=100
            )

            # 2. Synthesize adaptively
            final_response = await self.adaptive_synthesizer.synthesize_adaptively(
                original_query=original_query,
                results=results,
                quality=quality,
                strategy="intelligent",
                persona=persona,
            )

            self.cost_optimizer.record_llm_call(
                "synthesis", input_tokens=500, output_tokens=300
            )

            # 3. Record feedback for learning
            execution_time = self.performance_optimizer.get_elapsed_time()
            worker_count = len(results)

            # Extract complexity score from results metadata if available
            complexity_score = 0.5
            for worker_data in results.values():
                if isinstance(worker_data, dict) and "metadata" in worker_data:
                    complexity_score = worker_data.get("metadata", {}).get(
                        "complexity_score", 0.5
                    )
                    break

            self.feedback_learner.record_execution(
                query=original_query,
                strategy="intelligent",
                worker_count=worker_count,
                complexity_score=complexity_score,
                execution_time=execution_time,
                quality_metrics={
                    "completeness": quality.completeness,
                    "coherence": quality.coherence,
                    "relevance": quality.relevance,
                    "confidence": quality.confidence,
                },
                success=True,
            )

            # Log learning stats
            stats = self.feedback_learner.get_execution_stats()
            logger.info(
                f"Learning stats: {stats['total_executions']} executions, "
                f"{stats['success_rate']:.2%} success rate, "
                f"avg quality: {stats['avg_quality']:.2f}"
            )

            logger.info(f"Final cost: ${self.cost_optimizer.current_cost:.4f}")

            return final_response

        except Exception as e:
            logger.error(f"Error in adaptive synthesis: {e}")

            # Record failure
            self.feedback_learner.record_execution(
                query=original_query,
                strategy="intelligent",
                worker_count=len(results),
                complexity_score=0.5,
                execution_time=self.performance_optimizer.get_elapsed_time(),
                quality_metrics={
                    "completeness": 0.0,
                    "coherence": 0.0,
                    "relevance": 0.0,
                    "confidence": 0.0,
                },
                success=False,
            )

            return "I apologize, but I encountered an error synthesizing the results."
