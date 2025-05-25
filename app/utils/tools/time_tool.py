"""
Tool for getting current time information in different timezones.
"""

from typing import Type
from datetime import datetime
from zoneinfo import ZoneInfo
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


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
            return {
                "current_time": (
                    current_time.strftime("%Y-%m-%d %H:%M:%S %Z")
                ),
                "timezone": timezone,
                "timestamp": current_time.timestamp(),
                "iso_format": current_time.isoformat()
            }
        except Exception as e:
            return f"Error getting time information: {str(e)}"

    def _arun(self, timezone: str = "UTC"):
        raise NotImplementedError(
            "TimeInfoTool does not support async execution"
        ) 