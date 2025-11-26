"""Orchestrator utilities initialization."""

from .task_decomposer import ParallelTaskDecomposer

# Re-export classes from the parent legion.orchestrator module for convenience
# Note: This avoids circular imports by not importing here
# Users should import from app.hermes.legion.orchestrator directly

__all__ = ["ParallelTaskDecomposer"]
