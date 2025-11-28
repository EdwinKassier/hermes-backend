"""
Task Dependency Management for Sequential and Parallel Execution.

This module provides utilities for detecting, representing, and managing
dependencies between tasks in multi-agent workflows.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Set

from app.shared.utils.service_loader import get_async_llm_service

from ..utils.llm_utils import extract_json_from_llm_response

logger = logging.getLogger(__name__)


class DependencyType(str, Enum):
    """Type of dependency between tasks."""

    # Task B needs output from Task A
    DATA_DEPENDENCY = "data_dependency"
    # Task B must run after Task A (ordering constraint)
    SEQUENCE_DEPENDENCY = "sequence_dependency"


@dataclass
class TaskDependency:
    """Represents a dependency between two tasks."""

    source_task_id: str  # The task that must complete first
    target_task_id: str  # The task that depends on source
    dependency_type: DependencyType
    description: str = ""  # Human-readable explanation


@dataclass
class DependencyAwareTask:
    """
    A task with dependency information.

    This extends the basic subtask structure with dependency tracking
    to support sequential execution where needed.
    """

    task_id: str
    description: str
    agent_type: str
    keywords: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # Task IDs this depends on
    is_parallel: bool = True  # Can run in parallel with other tasks at same level
    priority: int = 0  # Lower = higher priority (executed first among peers)
    estimated_duration_seconds: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskDependencyGraph:
    """
    Manages a directed acyclic graph (DAG) of task dependencies.

    Provides methods for:
    - Adding tasks and dependencies
    - Topological sorting for execution order
    - Detecting cycles
    - Finding parallel execution groups
    """

    def __init__(self):
        self.tasks: Dict[str, DependencyAwareTask] = {}
        self.dependencies: List[TaskDependency] = []
        self._adjacency: Dict[str, Set[str]] = {}  # task_id -> dependent task_ids
        self._reverse_adjacency: Dict[str, Set[str]] = (
            {}
        )  # task_id -> prerequisite task_ids

    def add_task(self, task: DependencyAwareTask) -> None:
        """Add a task to the graph."""
        self.tasks[task.task_id] = task
        if task.task_id not in self._adjacency:
            self._adjacency[task.task_id] = set()
        if task.task_id not in self._reverse_adjacency:
            self._reverse_adjacency[task.task_id] = set()

        # Add edges from declared dependencies
        for dep_id in task.dependencies:
            self.add_dependency(
                TaskDependency(
                    source_task_id=dep_id,
                    target_task_id=task.task_id,
                    dependency_type=DependencyType.DATA_DEPENDENCY,
                    description=f"Task {task.task_id} depends on output from {dep_id}",
                )
            )

    def add_dependency(self, dependency: TaskDependency) -> None:
        """Add a dependency edge to the graph."""
        self.dependencies.append(dependency)

        # Update adjacency lists
        if dependency.source_task_id not in self._adjacency:
            self._adjacency[dependency.source_task_id] = set()
        self._adjacency[dependency.source_task_id].add(dependency.target_task_id)

        if dependency.target_task_id not in self._reverse_adjacency:
            self._reverse_adjacency[dependency.target_task_id] = set()
        self._reverse_adjacency[dependency.target_task_id].add(
            dependency.source_task_id
        )

    def has_cycle(self) -> bool:
        """Check if the dependency graph has a cycle."""
        visited = set()
        rec_stack = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in self._adjacency.get(node, set()):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for task_id in self.tasks:
            if task_id not in visited:
                if dfs(task_id):
                    return True

        return False

    def topological_sort(self) -> List[str]:
        """
        Return tasks in topological order (dependencies before dependents).

        Raises:
            ValueError: If the graph contains a cycle
        """
        if self.has_cycle():
            raise ValueError("Cannot sort: dependency graph contains a cycle")

        # Kahn's algorithm
        in_degree = {task_id: 0 for task_id in self.tasks}
        for task_id in self.tasks:
            for dep in self._reverse_adjacency.get(task_id, set()):
                if dep in in_degree:  # Only count known tasks
                    in_degree[task_id] += 1

        # Start with tasks that have no dependencies
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            # Sort by priority within the queue (lower priority number = higher priority)
            queue.sort(key=lambda t: self.tasks[t].priority if t in self.tasks else 0)
            current = queue.pop(0)
            result.append(current)

            for neighbor in self._adjacency.get(current, set()):
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        return result

    def get_execution_levels(self) -> List[List[str]]:
        """
        Group tasks into execution levels.

        Tasks at the same level can be executed in parallel.
        Tasks at level N+1 depend on tasks at level N or earlier.

        Returns:
            List of task ID lists, where each inner list is an execution level
        """
        if self.has_cycle():
            raise ValueError("Cannot determine levels: graph contains a cycle")

        levels: List[List[str]] = []
        remaining = set(self.tasks.keys())
        completed = set()

        while remaining:
            # Find tasks whose dependencies are all completed
            current_level = []
            for task_id in remaining:
                deps = self._reverse_adjacency.get(task_id, set())
                # Only consider dependencies that are actual tasks in our graph
                relevant_deps = deps & set(self.tasks.keys())
                if relevant_deps <= completed:
                    current_level.append(task_id)

            if not current_level:
                # No progress possible - shouldn't happen if no cycle
                logger.error(f"No progress possible with remaining tasks: {remaining}")
                break

            # Sort current level by priority
            current_level.sort(
                key=lambda t: self.tasks[t].priority if t in self.tasks else 0
            )
            levels.append(current_level)
            completed.update(current_level)
            remaining -= set(current_level)

        return levels

    def get_ready_tasks(self, completed_tasks: Set[str]) -> List[str]:
        """
        Get tasks that are ready to execute given completed tasks.

        Args:
            completed_tasks: Set of task IDs that have completed

        Returns:
            List of task IDs ready to execute
        """
        ready = []
        for task_id, task in self.tasks.items():
            if task_id in completed_tasks:
                continue
            deps = self._reverse_adjacency.get(task_id, set())
            relevant_deps = deps & set(self.tasks.keys())
            if relevant_deps <= completed_tasks:
                ready.append(task_id)

        # Sort by priority
        ready.sort(key=lambda t: self.tasks[t].priority if t in self.tasks else 0)
        return ready


class DependencyAnalyzer:
    """
    AI-powered analyzer for detecting dependencies between tasks.

    Uses LLM to analyze task descriptions and identify:
    - Data dependencies (output from one feeds into another)
    - Sequence dependencies (logical ordering requirements)
    - Tasks that can be parallelized
    """

    def __init__(self):
        self.llm_service = get_async_llm_service()

    async def analyze_dependencies(
        self, subtasks: List[Dict[str, Any]]
    ) -> TaskDependencyGraph:
        """
        Analyze a list of subtasks and build a dependency graph.

        Args:
            subtasks: List of subtask dictionaries with 'description' and 'agent_type'

        Returns:
            TaskDependencyGraph with tasks and detected dependencies
        """
        if not subtasks:
            return TaskDependencyGraph()

        # Generate task IDs if not present
        tasks_with_ids = []
        for i, task in enumerate(subtasks):
            task_id = task.get("task_id", f"task_{i}")
            tasks_with_ids.append({**task, "task_id": task_id})

        # Use LLM to detect dependencies
        try:
            dependencies = await self._detect_dependencies_with_llm(tasks_with_ids)
        except Exception as e:
            logger.error(f"LLM dependency detection failed: {e}")
            dependencies = []

        # Build the graph
        graph = TaskDependencyGraph()

        for task_data in tasks_with_ids:
            # Find dependencies for this task
            task_deps = [
                d["source"]
                for d in dependencies
                if d.get("target") == task_data["task_id"]
            ]

            task = DependencyAwareTask(
                task_id=task_data["task_id"],
                description=task_data.get("description", ""),
                agent_type=task_data.get("agent_type", "general"),
                keywords=task_data.get("keywords", []),
                dependencies=task_deps,
                is_parallel=len(task_deps) == 0,  # Parallel if no dependencies
                priority=task_data.get("priority", 0),
            )
            graph.add_task(task)

        return graph

    async def _detect_dependencies_with_llm(
        self, tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Use LLM to detect dependencies between tasks."""
        # Build task descriptions
        task_list = "\n".join(
            [
                f"- {t['task_id']}: {t.get('description', 'No description')}"
                for t in tasks
            ]
        )

        prompt = f"""Analyze these tasks and identify dependencies between them.

TASKS:
{task_list}

DEPENDENCY RULES:
1. A task has a DATA DEPENDENCY if it needs the output/result from another task
2. A task has a SEQUENCE DEPENDENCY if it must logically come after another task
3. Tasks with NO dependencies can run IN PARALLEL

EXAMPLES:
- "Analyze the research results" depends on "Research the topic" (data dependency)
- "Write the report" depends on "Gather all findings" (sequence dependency)
- "Research topic A" and "Research topic B" can run in parallel (no dependency)

Return ONLY valid JSON in this format:
{{
    "dependencies": [
        {{
            "source": "task_id_that_must_complete_first",
            "target": "task_id_that_depends_on_source",
            "type": "data_dependency|sequence_dependency",
            "reason": "brief explanation"
        }}
    ],
    "parallel_groups": [
        ["task_id_1", "task_id_2"]  // Tasks that can run together
    ]
}}

If all tasks can run in parallel, return empty dependencies list.
Be conservative - only add dependencies when there's a clear logical requirement.
"""

        response = await self.llm_service.generate_async(prompt, persona="hermes")
        data = extract_json_from_llm_response(response)

        dependencies = data.get("dependencies", [])
        logger.info(f"Detected {len(dependencies)} task dependencies")

        return dependencies


async def analyze_and_structure_tasks(subtasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convenience function to analyze tasks and return structured execution plan.

    Args:
        subtasks: Raw subtask list from task decomposition

    Returns:
        Dictionary with:
        - 'execution_levels': List of parallel execution groups
        - 'total_levels': Number of sequential levels
        - 'is_sequential': Whether any dependencies were found
        - 'tasks': Dictionary of task_id -> DependencyAwareTask
    """
    analyzer = DependencyAnalyzer()
    graph = await analyzer.analyze_dependencies(subtasks)

    try:
        levels = graph.get_execution_levels()
    except ValueError as e:
        logger.error(f"Failed to compute execution levels: {e}")
        # Fallback: all parallel
        levels = [list(graph.tasks.keys())]

    return {
        "execution_levels": levels,
        "total_levels": len(levels),
        "is_sequential": len(levels) > 1,
        "tasks": graph.tasks,
        "dependency_count": len(graph.dependencies),
    }
