"""
Intelligent Strategy Implementation with Advanced Features.

This strategy uses all intelligence services including Phase 4 optimizations
to create an adaptive, self-optimizing orchestration approach.
"""

import logging
from typing import Any, Dict, List

from app.shared.utils.toolhub import get_all_tools

from ..agents.task_agent_planner import TaskAgentPlanner
from ..intelligence.adaptive_synthesizer import AdaptiveSynthesizer
from ..intelligence.cost_optimizer import CostOptimizer
from ..intelligence.feedback_learner import FeedbackLearner
from ..intelligence.performance_optimizer import PerformanceOptimizer
from ..intelligence.tool_intelligence import ToolIntelligence
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
        # Core dynamic agent intelligence
        self.tool_intelligence = ToolIntelligence()
        self.adaptive_synthesizer = AdaptiveSynthesizer()

        # Advanced optimization services
        self.feedback_learner = _global_feedback_learner  # Shared across instances
        self.performance_optimizer = PerformanceOptimizer()
        self.cost_optimizer = CostOptimizer()

        # Dynamic agents handle their own personas - no persona generator needed

    async def generate_workers(
        self, query: str, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate workers using TaskAgentPlanner with intelligent optimizations."""
        try:
            # Start performance monitoring
            self.performance_optimizer.start_execution()
            self.cost_optimizer.reset()

            # Use TaskAgentPlanner for intelligent agent creation
            planner = TaskAgentPlanner()
            analysis = planner.analyze_task_and_plan_agents(
                task_description=query,
                user_context=context.get("user_context"),
                complexity_estimate=self._get_complexity_estimate(context),
            )

            if not analysis.get("agent_plan"):
                logger.warning("TaskAgentPlanner failed, using intelligent fallback")
                return await self._create_intelligent_fallback_worker(query)

            # Apply intelligent optimizations to the agent plan
            optimized_plan = await self._apply_intelligent_optimizations(
                analysis, query, context
            )

            # Convert to worker plan with dynamic agents
            worker_plan = planner.create_worker_plan_from_analysis(
                optimized_plan, query
            )

            # Apply intelligent tool allocation
            worker_plan = await self._apply_intelligent_tool_allocation(worker_plan)

            # End performance monitoring
            execution_time = self.performance_optimizer.get_elapsed_time()
            logger.info(
                f"Intelligent strategy generated {len(worker_plan)} dynamic agents in {execution_time:.2f}s"
            )
            logger.info(f"Cost summary: {self.cost_optimizer.get_cost_summary()}")

            return worker_plan

        except Exception as e:
            logger.error(f"Intelligent strategy failed: {e}")
            return await self._create_intelligent_fallback_worker(query)

    async def _apply_intelligent_optimizations(
        self, analysis: Dict[str, Any], query: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply intelligent optimizations to the agent plan."""
        try:
            # Use complexity estimate from context or default to moderate
            complexity_estimate = context.get("complexity_estimate", "moderate")
            complexity_score = {"simple": 3, "moderate": 6, "complex": 9}.get(
                complexity_estimate, 6
            )

            # Learn from history for worker count optimization
            agent_plan = analysis.get("agent_plan", [])
            if len(agent_plan) > 1:
                learned_count = self.feedback_learner.get_optimal_worker_count(
                    complexity_score
                )
                if len(self.feedback_learner.execution_history) > 10 and learned_count:
                    # Apply learned optimization
                    optimized_count = min(len(agent_plan), learned_count)
                    if optimized_count < len(agent_plan):
                        logger.info(
                            f"Reducing agent count from {len(agent_plan)} to {optimized_count} based on learning"
                        )
                        agent_plan = agent_plan[:optimized_count]
                        analysis["agent_plan"] = agent_plan
                        analysis["task_analysis"]["estimated_steps"] = optimized_count

            # Apply cost optimization
            total_agents = len(agent_plan)
            optimized_count = self.cost_optimizer.should_reduce_workers(total_agents)
            if optimized_count < total_agents:
                logger.info(
                    f"Cost optimization: reducing agents from {total_agents} to {optimized_count}"
                )
                agent_plan = agent_plan[:optimized_count]
                analysis["agent_plan"] = agent_plan

            # Record costs
            self.cost_optimizer.record_llm_call(
                "task_analysis", input_tokens=400, output_tokens=300
            )
            self.cost_optimizer.record_llm_call(
                "agent_optimization", input_tokens=200, output_tokens=100
            )

            return analysis

        except Exception as e:
            logger.warning(f"Intelligent optimization failed: {e}, using original plan")
            return analysis

    async def _apply_intelligent_tool_allocation(
        self, worker_plan: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply intelligent tool allocation to workers."""
        try:
            available_tools = get_all_tools()

            for worker in worker_plan:
                agent_config = worker.get("dynamic_agent_config", {})
                task_description = worker.get("task_description", "")

                # Use intelligent tool allocation based on agent capabilities
                capabilities = agent_config.get("capabilities", {})
                tools_needed = capabilities.get("tools_needed", [])

                # Get intelligent tool recommendations
                recommended_tools = await self.tool_intelligence.recommend_tools(
                    role=agent_config.get("agent_type", "general"),
                    task=task_description,
                    available_tools=available_tools,
                )

                # Update worker with intelligently allocated tools
                worker["tools"] = [t.name for t in recommended_tools]

                # Add metadata about intelligent allocation
                if "metadata" not in worker:
                    worker["metadata"] = {}
                worker["metadata"].update(
                    {
                        "intelligent_tool_allocation": True,
                        "tool_count": len(recommended_tools),
                        "performance_monitoring": True,
                        "cost_optimized": True,
                    }
                )

                self.cost_optimizer.record_llm_call(
                    "tool_recommendation", input_tokens=150, output_tokens=50
                )

            return worker_plan

        except Exception as e:
            logger.warning(
                f"Intelligent tool allocation failed: {e}, using basic allocation"
            )
            return worker_plan

    def _get_complexity_estimate(self, context: Dict[str, Any]) -> str:
        """Extract complexity estimate from context."""
        complexity = context.get("complexity_estimate", "moderate")
        # Map to expected values
        if isinstance(complexity, (int, float)):
            if complexity > 7:
                return "complex"
            elif complexity > 4:
                return "moderate"
            else:
                return "simple"
        return str(complexity).lower()

    async def _create_intelligent_fallback_worker(
        self, query: str
    ) -> List[Dict[str, Any]]:
        """Create an intelligent fallback worker when planning fails."""
        fallback_config = {
            "agent_id": "intelligent_fallback",
            "agent_type": "intelligent_fallback_specialist",
            "task_types": ["general", "analysis", "problem_solving"],
            "capabilities": {
                "primary_focus": "handling complex tasks when intelligent planning fails",
                "tools_needed": ["reasoning", "analysis", "problem_solving"],
                "expertise_level": "intermediate",
                "specializations": ["fallback_handling", "general_assistance"],
                "knowledge_domains": ["general_knowledge", "task_management"],
            },
            "prompts": {
                "identify_required_info": """Analyze this task to determine what information is needed.

Task: "{task}"
User Message: "{user_message}"

Determine what information is needed to complete this task effectively.

Response format (JSON):
{{
  "needs_info": true|false,
  "inferred_values": {{}},
  "required_fields": [],
  "reasoning": "why you need this information"
}}""",
                "execute_task": """Complete this task using your intelligent analysis capabilities.

Task: {task}
{judge_feedback}

Your capabilities: {capabilities}
Available tools: {tool_context}

Provide a comprehensive solution to the task, applying intelligent analysis and problem-solving.""",
            },
            "persona": "intelligent_assistant",
            "task_portion": query,
            "dependencies": [],
        }

        return [
            {
                "worker_id": "intelligent_fallback",
                "role": "intelligent_fallback",
                "task_description": query,
                "tools": [],
                "execution_level": 0,
                "dependencies": [],
                "dynamic_agent_config": fallback_config,
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
