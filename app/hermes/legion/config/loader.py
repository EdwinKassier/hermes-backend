"""Configuration loader for agent definitions."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


class AgentDefinition(BaseModel):
    """Agent definition from configuration."""

    id: str
    type: str
    description: str
    task_types: List[str]
    required_tools: List[str] = Field(default_factory=list)
    optional_tools: List[str] = Field(default_factory=list)
    max_iterations: int = 5
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GlobalConfig(BaseModel):
    """Global agent configuration."""

    default_timeout: int = 300
    max_concurrent_per_user: int = 5
    enable_persistence: bool = True
    log_level: str = "INFO"
    retry: Dict[str, Any] = Field(default_factory=dict)


class ToolAllocationConfig(BaseModel):
    """Tool allocation configuration."""

    strategy: str = "recommended"
    allow_tool_sharing: bool = True
    max_tools_per_agent: int = 10


class AgentConfiguration(BaseModel):
    """Complete agent configuration."""

    agents: List[AgentDefinition]
    global_config: GlobalConfig = Field(alias="global", default_factory=GlobalConfig)
    tool_allocation: ToolAllocationConfig = Field(default_factory=ToolAllocationConfig)


class ConfigLoader:
    """Loads and manages agent configurations from YAML files."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to agents.yaml. If None, uses default location.
        """
        if config_path is None:
            # Default to config/agents.yaml in legion module
            legion_dir = Path(__file__).parent.parent
            config_path = legion_dir / "config" / "agents.yaml"

        self.config_path = Path(config_path)
        self._config: Optional[AgentConfiguration] = None
        self._last_modified: Optional[float] = None

    def load_config(self, force_reload: bool = False) -> AgentConfiguration:
        """
        Load configuration from YAML file.

        Args:
            force_reload: Force reload even if already loaded

        Returns:
            AgentConfiguration object

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValidationError: If config is invalid
        """
        # Check if need to reload
        if not force_reload and self._config is not None:
            # Check if file modified
            if self.config_path.exists():
                current_mtime = self.config_path.stat().st_mtime
                if current_mtime == self._last_modified:
                    logger.debug("Using cached configuration")
                    return self._config

        # Load from file
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        logger.info(f"Loading agent configuration from {self.config_path}")

        try:
            with open(self.config_path, "r") as f:
                config_data = yaml.safe_load(f)

            # Parse and validate
            self._config = AgentConfiguration(**config_data)
            self._last_modified = self.config_path.stat().st_mtime

            logger.info(f"Loaded {len(self._config.agents)} agent definitions")
            return self._config

        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML config: {e}")
            raise
        except ValidationError as e:
            logger.error(f"Invalid configuration: {e}")
            raise

    def get_agent_definition(self, agent_type: str) -> Optional[AgentDefinition]:
        """
        Get agent definition by type.

        Args:
            agent_type: Type of agent to get

        Returns:
            AgentDefinition if found, None otherwise
        """
        config = self.load_config()

        for agent_def in config.agents:
            if agent_def.type == agent_type or agent_type in agent_def.task_types:
                return agent_def

        return None

    def get_all_agent_types(self) -> List[str]:
        """
        Get list of all available agent types.

        Returns:
            List of agent type strings
        """
        config = self.load_config()
        return [agent.type for agent in config.agents]

    def get_required_tools(self, agent_type: str) -> List[str]:
        """
        Get required tools for an agent type.

        Args:
            agent_type: Type of agent

        Returns:
            List of required tool names
        """
        agent_def = self.get_agent_definition(agent_type)
        return agent_def.required_tools if agent_def else []

    def get_optional_tools(self, agent_type: str) -> List[str]:
        """
        Get optional tools for an agent type.

        Args:
            agent_type: Type of agent

        Returns:
            List of optional tool names
        """
        agent_def = self.get_agent_definition(agent_type)
        return agent_def.optional_tools if agent_def else []

    def get_max_iterations(self, agent_type: str) -> int:
        """
        Get max iterations for an agent type.

        Args:
            agent_type: Type of agent

        Returns:
            Maximum iterations
        """
        agent_def = self.get_agent_definition(agent_type)
        return agent_def.max_iterations if agent_def else 5

    def hot_reload(self) -> bool:
        """
        Check if configuration has changed and reload if needed.

        Returns:
            True if configuration was reloaded, False otherwise
        """
        if not self.config_path.exists():
            return False

        current_mtime = self.config_path.stat().st_mtime

        if self._last_modified is None or current_mtime != self._last_modified:
            try:
                self.load_config(force_reload=True)
                logger.info("Configuration hot-reloaded successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to hot-reload configuration: {e}")
                return False

        return False

    @property
    def global_config(self) -> GlobalConfig:
        """Get global configuration."""
        config = self.load_config()
        return config.global_config

    @property
    def tool_allocation_config(self) -> ToolAllocationConfig:
        """Get tool allocation configuration."""
        config = self.load_config()
        return config.tool_allocation


# Singleton instance
_config_loader: Optional[ConfigLoader] = None


def get_config_loader(config_path: Optional[str] = None) -> ConfigLoader:
    """
    Get or create the configuration loader singleton.

    Args:
        config_path: Optional path to config file

    Returns:
        ConfigLoader instance
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)
    return _config_loader
