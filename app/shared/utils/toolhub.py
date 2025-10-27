"""
This module serves as the central hub for all LangChain tools in the system.
It provides a collection of custom tools that can be used by LangChain agents.
"""

import importlib
import inspect
import os
from pathlib import Path
from typing import List

# Optional LangChain dependency
try:
    from langchain.tools import BaseTool

    LANGCHAIN_AVAILABLE = True
except ImportError:
    BaseTool = None
    LANGCHAIN_AVAILABLE = False


def get_all_tools():
    """
    Returns a list of all available tools by dynamically loading them from the tools directory.

    Returns:
        list[BaseTool] or list: List of all available tool instances, or empty list if LangChain not available
    """
    if not LANGCHAIN_AVAILABLE or BaseTool is None:
        print("LangChain not available. No tools loaded.")
        return []

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

    return tools


def get_tool_by_name(tool_name: str):
    """
    Get a specific tool by its name.

    Args:
        tool_name (str): The name of the tool to retrieve

    Returns:
        Optional[BaseTool]: The requested tool instance or None if not found
    """
    for tool in get_all_tools():
        if tool.name == tool_name:
            return tool
    return None
