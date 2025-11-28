"""
Tool Registry for efficient tool instance management.

This module provides a singleton registry for tool instances, allowing
workers to look up tools by name without re-instantiating them each time.
This improves performance and ensures consistent tool behavior.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Singleton registry for tool instances.

    Provides efficient lookup of tool instances by name, avoiding the need
    to re-instantiate tools in each worker. This is particularly useful
    for Legion workers that receive tool names via the Send API.

    Usage:
        registry = ToolRegistry.get_instance()
        tools = registry.get_tools(["web_search", "database_query"])
    """

    _instance: Optional["ToolRegistry"] = None
    _initialized: bool = False

    def __init__(self):
        """Initialize the tool registry."""
        self._tools: Dict[str, Any] = {}
        self._gemini_service = None

    @classmethod
    def get_instance(cls) -> "ToolRegistry":
        """
        Get the singleton instance of ToolRegistry.

        Returns:
            ToolRegistry singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
        if not cls._initialized:
            cls._instance._initialize_tools()
            cls._initialized = True
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance.

        Useful for testing or when tool configuration changes.
        """
        cls._instance = None
        cls._initialized = False
        logger.info("ToolRegistry singleton reset")

    def _initialize_tools(self) -> None:
        """
        Initialize all tools from the Gemini service.

        This loads all available tools and indexes them by name.
        """
        try:
            from app.shared.utils.service_loader import get_gemini_service

            self._gemini_service = get_gemini_service()
            all_tools = getattr(self._gemini_service, "all_tools", [])

            for tool in all_tools:
                tool_name = self._get_tool_name(tool)
                self._tools[tool_name] = tool

            logger.info(
                f"ToolRegistry initialized with {len(self._tools)} tools: "
                f"{list(self._tools.keys())}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize ToolRegistry: {e}")
            # Continue with empty registry - tools will be allocated dynamically

    def _get_tool_name(self, tool: Any) -> str:
        """Extract tool name from tool object."""
        if hasattr(tool, "name"):
            return tool.name
        elif hasattr(tool, "__name__"):
            return tool.__name__
        else:
            return str(tool)

    def get_tool(self, tool_name: str) -> Optional[Any]:
        """
        Get a single tool by name.

        Args:
            tool_name: Name of the tool to retrieve

        Returns:
            Tool instance or None if not found
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            logger.warning(f"Tool '{tool_name}' not found in registry")
        return tool

    def get_tools(self, tool_names: List[str]) -> List[Any]:
        """
        Get multiple tools by name.

        Args:
            tool_names: List of tool names to retrieve

        Returns:
            List of tool instances (only includes found tools)
        """
        tools = []
        for name in tool_names:
            tool = self._tools.get(name)
            if tool is not None:
                tools.append(tool)
            else:
                logger.warning(f"Tool '{name}' not found in registry")
        return tools

    def get_all_tools(self) -> List[Any]:
        """
        Get all registered tools.

        Returns:
            List of all tool instances
        """
        return list(self._tools.values())

    def get_tool_names(self) -> List[str]:
        """
        Get names of all registered tools.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def has_tool(self, tool_name: str) -> bool:
        """
        Check if a tool is registered.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if tool is registered, False otherwise
        """
        return tool_name in self._tools

    def register_tool(self, tool: Any, name: Optional[str] = None) -> None:
        """
        Manually register a tool.

        Args:
            tool: Tool instance to register
            name: Optional name (extracted from tool if not provided)
        """
        tool_name = name or self._get_tool_name(tool)
        self._tools[tool_name] = tool
        logger.info(f"Manually registered tool: {tool_name}")

    def unregister_tool(self, tool_name: str) -> bool:
        """
        Unregister a tool.

        Args:
            tool_name: Name of the tool to unregister

        Returns:
            True if tool was unregistered, False if not found
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            logger.info(f"Unregistered tool: {tool_name}")
            return True
        return False


def get_tool_registry() -> ToolRegistry:
    """
    Convenience function to get the ToolRegistry singleton.

    Returns:
        ToolRegistry singleton instance
    """
    return ToolRegistry.get_instance()
