"""Orchestrator utilities initialization."""

from .result_synthesizer import ResultSynthesizer
from .task_dependencies import (
    DependencyAnalyzer,
    DependencyAwareTask,
    DependencyType,
    TaskDependency,
    TaskDependencyGraph,
    analyze_and_structure_tasks,
)

# Re-export classes from the parent legion.orchestrator module for convenience
# Note: This avoids circular imports by not importing here
# Users should import from app.hermes.legion.orchestrator directly

__all__ = [
    # Task dependencies
    "DependencyType",
    "TaskDependency",
    "DependencyAwareTask",
    "TaskDependencyGraph",
    "DependencyAnalyzer",
    "analyze_and_structure_tasks",
    # Result synthesis
    "ResultSynthesizer",
]
