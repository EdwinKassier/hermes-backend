"""Agent factory for dynamic agent creation."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.shared.utils.service_loader import get_llm_service

from ..agents.base import BaseSubAgent
from ..agents.dynamic_agent import DynamicAgent
from ..models import RequiredInfoField, SubAgentState
from ..state import AgentConfig, AgentInfo

logger = logging.getLogger(__name__)


class AgentFactory:
    """
    Factory for creating dynamic agents from scratch.

    Only supports dynamic agent creation - no hardcoded agent types.
    """

    @classmethod
    def create_dynamic_agent(
        cls,
        agent_id: str,
        task_types: List[str],
        capabilities: Dict[str, Any],
        prompts: Dict[str, str],
        persona: str = "hermes",
        tools: Optional[List[Any]] = None,
        **config,
    ) -> BaseSubAgent:
        """
        Create a dynamic agent from configuration.

        Args:
            agent_id: Unique identifier for the agent
            task_types: List of task types the agent can handle
            capabilities: Dictionary defining agent capabilities
            prompts: Dictionary of prompt templates
            persona: AI persona to use
            tools: Optional list of tools
            **config: Additional configuration options

        Returns:
            DynamicAgent instance
        """
        try:
            agent = DynamicAgent(
                agent_id=agent_id,
                task_types=task_types,
                persona=persona,
                capabilities=capabilities,
                prompts=prompts,
                **config,
            )

            # Set tools if provided
            if tools and hasattr(agent, "set_tools"):
                agent.set_tools(tools)

            logger.info(
                f"Created dynamic agent '{agent_id}' with persona '{persona}' "
                f"and {len(tools) if tools else 0} tools"
            )

            return agent

        except Exception as e:
            logger.error(f"Failed to create dynamic agent '{agent_id}': {e}")
            raise
