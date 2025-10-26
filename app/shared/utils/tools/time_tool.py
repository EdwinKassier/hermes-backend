"""
Tool for getting current time information in different timezones.
"""

from typing import Type
from datetime import datetime
from zoneinfo import ZoneInfo
from pydantic import BaseModel, Field

# Optional LangChain dependency
try:
    from langchain.tools import BaseTool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    BaseTool = object  # Fallback base class
    LANGCHAIN_AVAILABLE = False


class TimeInfoInput(BaseModel):
    """Input for the TimeInfo tool."""
    timezone: str = Field(
        default="UTC",
        description=(
            "The timezone to get the time for "
            "(e.g., 'UTC', 'US/Pacific', 'Europe/London')"
        )
    )


class TimeInfoTool(BaseTool):
    """Tool for getting current time information in different timezones."""
    name: str = "time_info"
    description: str = """
    Get current time information for a specific timezone.
    Useful for when you need to know the current time in any timezone.
    """
    args_schema: Type[BaseModel] = TimeInfoInput

    def _run(self, timezone: str = "UTC") -> str:
        try:
            current_time = datetime.now(ZoneInfo(timezone))
            formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S %Z")
            iso_time = current_time.isoformat()
            
            # Return formatted string for LangChain
            return (
                f"Current time in {timezone}: {formatted_time}\n"
                f"ISO format: {iso_time}\n"
                f"Unix timestamp: {current_time.timestamp()}"
            )
        except Exception as e:
            return f"Error getting time information: {str(e)}"

    def _arun(self, timezone: str = "UTC"):
        raise NotImplementedError(
            "TimeInfoTool does not support async execution"
        ) 