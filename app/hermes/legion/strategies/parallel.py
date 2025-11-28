"""Parallel strategy implementation with dependency-aware execution."""

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
    and executes them with dependency awareness.

    Key Features:
    - AI-powered task decomposition
    - Dependency detection between subtasks
    - Level-based execution (tasks at same level run in parallel)
    - Sequential execution between levels
    """

    async def generate_workers(
        self, query: str, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Decompose task into subtasks with dependency analysis.

        Returns workers annotated with execution level information.
        """
        decomposer = ParallelTaskDecomposer()
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
                    "execution_level": 0,
                    "dependencies": [],
                }
            ]

        # Analyze dependencies between tasks
        try:
            dependency_info = await decomposer.analyze_task_dependencies(subtasks)
            execution_levels = dependency_info.get("execution_levels", [])
            tasks_by_id = dependency_info.get("tasks", {})
            is_sequential = dependency_info.get("is_sequential", False)

            if is_sequential:
                logger.info(
                    "Detected sequential workflow with %d levels", len(execution_levels)
                )
        except (ValueError, RuntimeError) as e:
            logger.error("Dependency analysis failed: %s, treating all as parallel", e)
            execution_levels = [[f"task_{i}" for i in range(len(subtasks))]]
            tasks_by_id = {}
            is_sequential = False

        workers = []
        tool_allocator = ToolAllocator()

        # Build worker list with execution level info
        for level_idx, level_tasks in enumerate(execution_levels):
            for task_id in level_tasks:
                # Find the original subtask data
                if task_id in tasks_by_id:
                    task_info = tasks_by_id[task_id]
                    subtask = {
                        "description": task_info.description,
                        "agent_type": task_info.agent_type,
                        "keywords": task_info.keywords,
                    }
                    dependencies = task_info.dependencies
                else:
                    # Fallback to original subtask by index
                    # Try to extract index from task_id like "task_0", "task_1"
                    idx = 0
                    if "_" in task_id:
                        try:
                            # Get the last part after underscore and try to parse as int
                            last_part = task_id.split("_")[-1]
                            idx = int(last_part)
                        except ValueError:
                            # Not a number, use position in execution_levels
                            idx = level_tasks.index(task_id)

                    if idx < len(subtasks):
                        subtask = subtasks[idx]
                    else:
                        # Skip this task if we can't find data
                        logger.warning("Could not find subtask data for %s", task_id)
                        continue
                    dependencies = []

                # Allocate tools
                tools = tool_allocator.allocate_tools_for_task(
                    task_type=subtask.get("agent_type", "general"),
                    task_description=subtask.get("description", ""),
                )
                tool_names = [t.name for t in tools]

                workers.append(
                    {
                        "worker_id": task_id,
                        "role": subtask.get("agent_type", "general"),
                        "task_description": subtask.get("description", ""),
                        "tools": tool_names,
                        "execution_level": level_idx,
                        "dependencies": dependencies,
                        "metadata": {
                            "is_sequential": is_sequential,
                            "total_levels": len(execution_levels),
                            "keywords": subtask.get("keywords", []),
                        },
                    }
                )

        # Sort by execution level to ensure proper ordering
        workers.sort(key=lambda w: w["execution_level"])

        logger.info(
            "Generated %d workers across %d execution levels",
            len(workers),
            len(execution_levels),
        )
        return workers

    async def synthesize_results(
        self, original_query: str, results: Dict[str, Any], persona: str
    ) -> str:
        """
        Synthesize subtask results with level-aware ordering.

        Results from earlier execution levels are presented first to
        maintain logical flow in the synthesized response.
        """
        # Format results for synthesizer, preserving level information
        formatted_results = {}
        for worker_id, data in results.items():
            formatted_results[worker_id] = {
                "agent_id": worker_id,
                "result": data["result"],
                "status": data["status"],
                "agent_type": data["role"],
                # Include metadata for synthesis
                "execution_level": data.get("execution_level", 0),
            }

        # Sort by execution level for ordered synthesis
        sorted_results = dict(
            sorted(
                formatted_results.items(),
                key=lambda x: x[1].get("execution_level", 0),
            )
        )

        synthesizer = ResultSynthesizer()
        return synthesizer.synthesize_results(
            original_query=original_query,
            agent_results=sorted_results,
            persona=persona,
        )
