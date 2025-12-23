"""
This module serves as the central hub for all LangChain tools in the system.
It provides a collection of custom tools that can be used by LangChain agents.

Tools are cached after first load to prevent redundant file I/O and module imports.
"""

import importlib
import inspect
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

# Optional LangChain dependency
try:
    from langchain.tools import BaseTool

    LANGCHAIN_AVAILABLE = True
except ImportError:
    BaseTool = None
    LANGCHAIN_AVAILABLE = False


# Module-level cache for tools
_tools_cache: Optional[List] = None


def get_all_tools():
    """
    Returns a list of all available tools by dynamically loading them from the tools directory.

    Tools are cached after first load to eliminate redundant file system operations
    and module imports (~10-50ms savings per call after first load).

    Returns:
        list[BaseTool] or list: List of all available tool instances, or empty list if LangChain not available
    """
    global _tools_cache

    # Return cached tools if available
    if _tools_cache is not None:
        return _tools_cache

    if not LANGCHAIN_AVAILABLE or BaseTool is None:
        print("LangChain not available. No tools loaded.")
        _tools_cache = []
        return _tools_cache

    tools = []
    tools_dir = Path(__file__).parent / "tools"

    # Skip __init__.py and __pycache__
    for file in tools_dir.glob("*.py"):
        if file.name.startswith("__"):
            continue

        # Convert file path to module path
        module_name = f"app.shared.utils.tools.{file.stem}"
        try:
            # Import the module
            module = importlib.import_module(module_name)

            # Find all classes in the module that inherit from BaseTool
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, BaseTool)
                    and obj != BaseTool
                ):
                    tools.append(obj())

        except Exception as e:
            print(f"Error loading tool from {file}: {e}")

    _tools_cache = tools
    return _tools_cache


def clear_tools_cache():
    """
    Clear the cached tools, forcing a reload on next get_all_tools() call.

    Useful for testing or when tools directory contents change.
    """
    global _tools_cache, _tools_name_index
    _tools_cache = None
    _tools_name_index = None
    logging.info("Tools cache cleared")


def get_tool_by_name(tool_name: str):
    """
    Get a specific tool by its name.

    Uses cached tool list for O(1) lookup (after first call builds index).

    Args:
        tool_name (str): The name of the tool to retrieve

    Returns:
        Optional[BaseTool]: The requested tool instance or None if not found
    """
    # Build name index on first call for O(1) lookups
    global _tools_name_index
    if "_tools_name_index" not in globals() or _tools_name_index is None:
        _tools_name_index = {tool.name: tool for tool in get_all_tools()}

    return _tools_name_index.get(tool_name)


# Module-level name index (built on first get_tool_by_name call)
_tools_name_index = None
