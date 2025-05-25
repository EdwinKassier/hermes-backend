"""
This module serves as the central hub for all LangChain tools in the system.
It provides a collection of custom tools that can be used by LangChain agents.
"""

import os
import importlib
import inspect
from typing import List
from langchain.tools import BaseTool
from pathlib import Path

def get_all_tools() -> List[BaseTool]:
    """
    Returns a list of all available tools by dynamically loading them from the tools directory.
    
    Returns:
        list[BaseTool]: List of all available tool instances
    """
    tools = []
    tools_dir = Path(__file__).parent / "tools"
    
    # Skip __init__.py and __pycache__
    for file in tools_dir.glob("*.py"):
        if file.name.startswith("__"):
            continue
            
        # Convert file path to module path
        module_name = f"app.utils.tools.{file.stem}"
        try:
            # Import the module
            module = importlib.import_module(module_name)
            
            # Find all classes in the module that inherit from BaseTool
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseTool) and 
                    obj != BaseTool):
                    tools.append(obj())
                    
        except Exception as e:
            print(f"Error loading tool from {file}: {e}")
            
    return tools

def get_tool_by_name(tool_name: str) -> BaseTool | None:
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