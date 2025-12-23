"""
Tool Registry for efficient tool instance management.

This module provides a singleton registry for tool instances, allowing
workers to look up tools by name without re-instantiating them each time.
This improves performance and ensures consistent tool behavior.
"""

import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Singleton registry for tool instances.

    Provides efficient lookup of tool instances by name, avoiding the need
    to re-instantiate tools in each worker. This is particularly useful
    for Legion workers that receive tool names via the Send API.

    Thread-safe: Uses a lock to prevent race conditions during initialization.

    Usage:
        registry = ToolRegistry.get_instance()
        tools = registry.get_tools(["web_search", "database_query"])
    """

    _instance: Optional["ToolRegistry"] = None
    _initialized: bool = False
    _lock: threading.Lock = threading.Lock()

    # Circuit breaker settings
    DEGRADATION_TIMEOUT_SECONDS = 300  # 5 minutes
    MAX_TRANSIENT_ERRORS = 3

    def __init__(self):
        """Initialize the tool registry."""
        self._tools: Dict[str, Any] = {}
        self._gemini_service = None

        # Tool health tracking
        self._degraded_tools: Dict[str, float] = (
            {}
        )  # name -> timestamp when it can recover
        self._transient_errors: Dict[str, int] = {}  # name -> error count
        self._health_lock: threading.Lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "ToolRegistry":
        """
        Get the singleton instance of ToolRegistry.

        Thread-safe: Uses double-checked locking pattern for efficiency.

        Returns:
            ToolRegistry singleton instance
        """
        # Fast path: already initialized
        if cls._initialized and cls._instance is not None:
            return cls._instance

        # Slow path: need initialization with lock
        with cls._lock:
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

        Useful for test isolation or when tool configuration changes.
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

    def is_tool_healthy(self, tool_name: str) -> bool:
        """
        Check if a tool is currently healthy (not degraded).

        Returns True if tool is healthy, False if degraded.
        Auto-recovers if timeout has passed.
        """
        with self._health_lock:
            # Check if degraded
            if tool_name in self._degraded_tools:
                recovery_time = self._degraded_tools[tool_name]
                import time

                if time.time() > recovery_time:
                    # Auto-recover
                    del self._degraded_tools[tool_name]
                    if tool_name in self._transient_errors:
                        del self._transient_errors[tool_name]
                    logger.info(
                        f"Tool '{tool_name}' has auto-recovered from degradation."
                    )
                    return True
                else:
                    return False
            return True

    def mark_tool_failed(self, tool_name: str, error_msg: str) -> None:
        """
        Record a failure for a tool. May trigger degradation.

        Args:
            tool_name: Name of the failed tool
            error_msg: Error message for logging
        """
        import time

        with self._health_lock:
            current_errors = self._transient_errors.get(tool_name, 0) + 1
            self._transient_errors[tool_name] = current_errors

            if current_errors >= self.MAX_TRANSIENT_ERRORS:
                # Trigger degradation
                recovery_time = time.time() + self.DEGRADATION_TIMEOUT_SECONDS
                self._degraded_tools[tool_name] = recovery_time
                logger.warning(
                    f"Tool '{tool_name}' marked as DEGRADED until {recovery_time}. "
                    f"Reason: {self.MAX_TRANSIENT_ERRORS} failures. Last error: {error_msg}"
                )
            else:
                logger.warning(
                    f"Tool '{tool_name}' failed ({current_errors}/{self.MAX_TRANSIENT_ERRORS}). "
                    f"Error: {error_msg}"
                )

    def mark_tool_success(self, tool_name: str) -> None:
        """
        Record a successful execution. Resets transient error count.
        """
        with self._health_lock:
            if tool_name in self._transient_errors:
                del self._transient_errors[tool_name]


def get_tool_registry() -> ToolRegistry:
    """
    Convenience function to get the ToolRegistry singleton.

    Returns:
        ToolRegistry singleton instance
    """
    return ToolRegistry.get_instance()
