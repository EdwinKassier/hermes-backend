"""
Worker Planner Service.

Intelligently plans the composition of the Legion swarm based on query analysis.
Determines the optimal number of workers, their roles, and specific tasks.
"""

import logging
from typing import Any, Dict, List, Optional

from app.shared.utils.service_loader import get_async_llm_service

from ..models import Domain, QueryComplexity, WorkerPlan
from ..utils.llm_utils import extract_json_from_llm_response
from ..utils.persona_context import get_current_legion_persona

logger = logging.getLogger(__name__)


class IntelligentWorkerPlanner:
    """Plans worker teams for Legion orchestration."""

    def __init__(self):
        self.llm_service = get_async_llm_service()

    async def plan_workers(
        self,
        query: str,
        complexity: QueryComplexity,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> List[WorkerPlan]:
        """
        Create a detailed plan for the worker swarm.
        """
        try:
            # 1. Determine optimal worker count
            num_workers = self._optimize_worker_count(complexity)

            # 2. Generate worker definitions
            prompt = f"""
            Plan a team of {num_workers} AI agents to solve this task.

            Task: "{query}"

            Complexity Analysis:
            - Score: {complexity.score}
            - Dimensions: {complexity.dimensions}

            Create a plan where each worker has a specific role, specialization, and subtask.
            Ensure the subtasks cover the entire original task.

            Return ONLY valid JSON:
            {{
                "workers": [
                    {{
                        "role": "researcher",
                        "specialization": "historical_facts",
                        "task_description": "Research the history of...",
                        "priority": 1,
                        "estimated_duration": 30.0
                    }}
                ]
            }}
            """

            response = await self.llm_service.generate_async(
                prompt, persona=get_current_legion_persona()
            )
            data = extract_json_from_llm_response(response)

            worker_dicts = data.get("workers", [])
            plans = []

            for i, w in enumerate(worker_dicts):
                # Ensure unique IDs
                worker_id = f"worker_{i}_{w['role']}"

                plan = WorkerPlan(
                    worker_id=worker_id,
                    role=w["role"],
                    specialization=w.get("specialization", "general"),
                    task_description=w["task_description"],
                    tools=[],  # Tools will be allocated by ToolIntelligence
                    priority=w.get("priority", 1),
                    estimated_duration=w.get("estimated_duration", 30.0),
                )
                plans.append(plan)

            return plans

        except Exception as e:
            logger.error(f"Error planning workers: {e}")
            # Fallback to single worker
            return [
                WorkerPlan(
                    worker_id="fallback_worker",
                    role="general",
                    specialization="general",
                    task_description=query,
                    tools=[],
                    priority=1,
                )
            ]

    def _optimize_worker_count(self, complexity: QueryComplexity) -> int:
        """
        Calculate optimal number of workers based on complexity.
        """
        # Base logic:
        # Low complexity (<0.3) -> 1 worker
        # Medium complexity (0.3-0.7) -> 2-3 workers
        # High complexity (>0.7) -> 3-5 workers

        score = complexity.score
        suggested = complexity.suggested_workers

        # Trust the LLM's suggestion but bound it
        if score < 0.3:
            return max(1, min(suggested, 2))
        elif score < 0.7:
            return max(2, min(suggested, 4))
        else:
            return max(3, min(suggested, 6))
