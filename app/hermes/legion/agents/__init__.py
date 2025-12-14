"""Sub-agent implementations for Legion."""

from .base import BaseSubAgent
from .dynamic_agent import DynamicAgent
from .dynamic_agent_utils import (
    create_mixed_agent_plan,
    create_worker_plan_with_dynamic_agents,
    example_dynamic_mixed_team,
)
from .factory import AgentFactory
from .task_agent_planner import TaskAgentPlanner

__all__ = [
    "BaseSubAgent",
    "DynamicAgent",
    "AgentFactory",
    "TaskAgentPlanner",
    # Dynamic agent utilities
    "create_worker_plan_with_dynamic_agents",
    "create_mixed_agent_plan",
    "example_dynamic_mixed_team",
]
