"""
Cost Optimizer Service.

Manages LLM usage costs and optimizes for cost-effectiveness.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CostOptimizer:
    """
    Optimizes costs by tracking LLM usage and making cost-aware decisions.

    Note: Actual token counting would require integration with the LLM provider's
    API. This implementation provides the structure for cost tracking.
    """

    def __init__(self, budget_per_query: float = 0.10):
        """
        Initialize cost optimizer.

        Args:
            budget_per_query: Maximum cost per query in USD (default $0.10)
        """
        self.budget_per_query = budget_per_query
        self.current_cost = 0.0
        self.llm_calls = 0

        # Rough cost estimates (USD per 1K tokens)
        # These would be updated based on actual provider pricing
        self.cost_per_1k_input = 0.0015  # Example for Gemini Pro
        self.cost_per_1k_output = 0.002

    def reset(self) -> None:
        """Reset cost tracking for a new query."""
        self.current_cost = 0.0
        self.llm_calls = 0

    def record_llm_call(
        self, purpose: str, input_tokens: int = 0, output_tokens: int = 0
    ) -> None:
        """
        Record an LLM call and its estimated cost.

        Args:
            purpose: What the LLM call was for (e.g., "complexity_analysis")
            input_tokens: Approximate input token count
            output_tokens: Approximate output token count
        """
        input_cost = (input_tokens / 1000) * self.cost_per_1k_input
        output_cost = (output_tokens / 1000) * self.cost_per_1k_output
        call_cost = input_cost + output_cost

        self.current_cost += call_cost
        self.llm_calls += 1

        logger.info(
            f"LLM call for {purpose}: ~${call_cost:.4f} (total: ${self.current_cost:.4f})"
        )

    def is_within_budget(self) -> bool:
        """Check if we're still within budget."""
        return self.current_cost < self.budget_per_query

    def get_remaining_budget(self) -> float:
        """Get remaining budget for this query."""
        return max(0.0, self.budget_per_query - self.current_cost)

    def should_reduce_workers(self, planned_worker_count: int) -> int:
        """
        Suggest worker count reduction if approaching budget limit.

        Returns:
            Adjusted worker count
        """
        remaining_budget = self.get_remaining_budget()

        # Estimate cost per worker (rough approximation)
        # Each worker typically makes 2-3 LLM calls
        estimated_cost_per_worker = 0.02

        affordable_workers = int(remaining_budget / estimated_cost_per_worker)

        if affordable_workers < planned_worker_count:
            logger.warning(
                f"Cost optimization: reducing workers from {planned_worker_count} "
                f"to {affordable_workers} to stay within budget"
            )
            return max(1, affordable_workers)  # At least 1 worker

        return planned_worker_count

    def should_use_cheaper_model(self) -> bool:
        """
        Determine if we should switch to a cheaper model.

        Returns:
            True if we should use a less expensive model
        """
        # If we've used >75% of budget, consider cheaper models
        if self.current_cost > (self.budget_per_query * 0.75):
            logger.info(
                "Cost optimization: recommending cheaper model for remaining operations"
            )
            return True
        return False

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost tracking summary."""
        return {
            "current_cost": self.current_cost,
            "budget": self.budget_per_query,
            "remaining_budget": self.get_remaining_budget(),
            "llm_calls": self.llm_calls,
            "within_budget": self.is_within_budget(),
            "cost_per_call": self.current_cost / max(self.llm_calls, 1),
        }

    def set_budget(self, budget: float) -> None:
        """Set budget for the current query."""
        self.budget_per_query = budget
        logger.info(f"Budget set to ${budget:.2f}")
