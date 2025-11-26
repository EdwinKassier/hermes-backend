"""
Tool for getting current time information with rich context for LLM parsing.
"""

from datetime import datetime
from typing import Type
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
            "(e.g., 'UTC', 'US/Pacific', 'Europe/London', 'Asia/Tokyo')"
        ),
    )


class TimeInfoTool(BaseTool):
    """Tool for getting current time information with rich context."""

    name: str = "time_info"
    description: str = """
    Get current time information with rich context for a specific timezone.

    Provides comprehensive time data including:
    - Current date and time
    - Day of week
    - ISO format timestamp
    - Unix timestamp
    - Relative time descriptions

    Useful for when you need to know the current time in any timezone.
    """
    args_schema: Type[BaseModel] = TimeInfoInput

    def _run(self, timezone: str = "UTC") -> str:
        try:
            current_time = datetime.now(ZoneInfo(timezone))

            # Get various time components
            day_of_week = current_time.strftime("%A")  # Full day name
            month_name = current_time.strftime("%B")  # Full month name
            day = current_time.day
            year = current_time.year
            hour = current_time.hour
            minute = current_time.minute
            second = current_time.second

            # Determine time of day
            if 5 <= hour < 12:
                time_of_day = "morning"
            elif 12 <= hour < 17:
                time_of_day = "afternoon"
            elif 17 <= hour < 21:
                time_of_day = "evening"
            else:
                time_of_day = "night"

            # Format 12-hour time
            hour_12 = current_time.strftime("%I:%M %p")

            # Build comprehensive response
            response_parts = [
                f"=== Current Time Information ===",
                f"Timezone: {timezone}",
                f"",
                f"Date: {day_of_week}, {month_name} {day}, {year}",
                f"Time: {hour:02d}:{minute:02d}:{second:02d} (24-hour)",
                f"Time: {hour_12} (12-hour)",
                f"",
                f"Context:",
                f"- Day of week: {day_of_week}",
                f"- Time of day: {time_of_day}",
                f"",
                f"Technical Details:",
                f"- ISO 8601: {current_time.isoformat()}",
                f"- Unix timestamp: {int(current_time.timestamp())}",
                f"- Timezone offset: {current_time.strftime('%z')}",
            ]

            return "\n".join(response_parts)

        except Exception as e:
            return f"Error getting time information: {str(e)}"

    def _arun(self, timezone: str = "UTC"):
        raise NotImplementedError("TimeInfoTool does not support async execution")
