"""Agent factory for dynamic agent creation."""

import logging
from typing import Any, Dict, List, Optional

from app.shared.utils.service_loader import get_gemini_service

from ..agents.analysis_agent import AnalysisAgent
from ..agents.base import BaseSubAgent
from ..agents.code_agent import CodeAgent
from ..agents.data_agent import DataAgent
from ..agents.research_agent import ResearchAgent
from ..models import RequiredInfoField, SubAgentState
from ..state import AgentConfig, AgentInfo

logger = logging.getLogger(__name__)


class AgentFactory:
    """
    Factory for creating agents dynamically based on task requirements.

    This enables the Magentic pattern's dynamic agent creation capability.
    """

    # Registry of agent types to agent classes
    _agent_classes: Dict[str, type[BaseSubAgent]] = {
        "research": ResearchAgent,
        "code": CodeAgent,
        "programming": CodeAgent,  # Alias
        "analysis": AnalysisAgent,
        "data": DataAgent,
        # Add more agent types here as they're implemented
    }

    @classmethod
    def register_agent_type(
        cls, agent_type: str, agent_class: type[BaseSubAgent]
    ) -> None:
        """
        Register a new agent type.

        Args:
            agent_type: Type identifier (e.g., 'research', 'code')
            agent_class: Agent class that implements BaseSubAgent
        """
        if not issubclass(agent_class, BaseSubAgent):
            raise ValueError(f"Agent class must inherit from BaseSubAgent")

        cls._agent_classes[agent_type] = agent_class
        logger.info(f"Registered agent type '{agent_type}': {agent_class.__name__}")

    @classmethod
    def create_agent(
        cls,
        config: AgentConfig,
        tools: Optional[List[Any]] = None,
        persona: str = "hermes",
    ) -> BaseSubAgent:
        """
        Create an agent instance from configuration.

        Args:
            config: Agent configuration
            tools: Optional list of tools to provide to the agent
            persona: AI persona to use for this agent

        Returns:
            Instantiated agent

        Raises:
            ValueError: If agent type is not registered
        """
        agent_class = cls._agent_classes.get(config.agent_type)
        if not agent_class:
            raise ValueError(
                f"Unknown agent type '{config.agent_type}'. "
                f"Available: {list(cls._agent_classes.keys())}"
            )

        try:
            # Instantiate the agent with persona
            agent = agent_class(persona=persona)

            # Set tools if the agent supports it
            if tools and hasattr(agent, "set_tools"):
                agent.set_tools(tools)

            logger.info(
                f"Created agent '{agent.agent_id}' of type '{config.agent_type}' "
                f"with persona '{persona}' and {len(tools) if tools else 0} tools"
            )

            return agent

        except Exception as e:
            logger.error(f"Failed to create agent of type '{config.agent_type}': {e}")
            raise

    @classmethod
    def create_agent_from_task(
        cls,
        task_description: str,
        task_type: str,
        tools: Optional[List[Any]] = None,
        persona: str = "hermes",
    ) -> tuple[BaseSubAgent, AgentInfo]:
        """
        Create an agent based on task analysis.

        Args:
            task_description: Description of the task
            task_type: Type of task (research, code, analysis)
            tools: Optional list of tools
            persona: AI persona to use for this agent

        Returns:
            Tuple of (agent instance, agent info)
        """
        # Create config based on task
        config = AgentConfig(
            agent_type=task_type,
            required_tools=[],  # Will be determined by ToolAllocator
            metadata={
                "task_description": task_description,
            },
        )

        # Create agent with persona
        agent = cls.create_agent(config, tools, persona=persona)

        # Create agent info for tracking
        agent_info = AgentInfo(
            agent_id=agent.agent_id,
            agent_type=config.agent_type,
            tools=[getattr(t, "name", str(t)) for t in tools] if tools else [],
            metadata=config.metadata,
        )

        return agent, agent_info

    @classmethod
    def get_available_agent_types(cls) -> List[str]:
        """Get list of available agent types."""
        return list(cls._agent_classes.keys())

    @classmethod
    def from_config_dict(cls, config_dict: Dict[str, Any]) -> BaseSubAgent:
        """
        Create agent from dictionary configuration.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Instantiated agent
        """
        config = AgentConfig(**config_dict)
        return cls.create_agent(config)
