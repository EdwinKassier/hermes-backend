"""Parallel strategy implementation."""

import logging
from typing import Any, Dict, List

from ..parallel.result_synthesizer import ResultSynthesizer
from ..parallel.task_decomposer import ParallelTaskDecomposer
from ..utils import ToolAllocator
from .base import LegionStrategy

logger = logging.getLogger(__name__)


class ParallelStrategy(LegionStrategy):
    """
    Parallel strategy: Decomposes a complex task into subtasks
    and executes them in parallel.
    """

    async def generate_workers(
        self, query: str, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Decompose task into subtasks."""
        decomposer = ParallelTaskDecomposer()
        # Note: decompose_task is sync currently
        subtasks = decomposer.decompose_task(query, skip_check=True)

        if not subtasks:
            logger.warning(
                "Task decomposition failed or returned empty, falling back to single worker"
            )
            return [
                {
                    "worker_id": "parallel_default",
                    "role": "general",
                    "task_description": query,
                    "tools": [],
                }
            ]

        workers = []
        tool_allocator = ToolAllocator()

        for i, subtask in enumerate(subtasks):
            # Allocate tools
            tools = tool_allocator.allocate_tools_for_task(
                task_type=subtask["agent_type"], task_description=subtask["description"]
            )
            tool_names = [t.name for t in tools]

            workers.append(
                {
                    "worker_id": f"parallel_{i}_{subtask['agent_type']}",
                    "role": subtask["agent_type"],
                    "task_description": subtask["description"],
                    "tools": tool_names,
                }
            )

        return workers

    async def synthesize_results(
        self, original_query: str, results: Dict[str, Any], persona: str
    ) -> str:
        """Synthesize subtask results."""
        # Format results for synthesizer
        formatted_results = {}
        for worker_id, data in results.items():
            formatted_results[worker_id] = {
                "agent_id": worker_id,
                "result": data["result"],
                "status": data["status"],
                "agent_type": data["role"],
            }

        synthesizer = ResultSynthesizer()
        return synthesizer.synthesize_results(
            original_query=original_query,
            agent_results=formatted_results,
            persona=persona,
        )
