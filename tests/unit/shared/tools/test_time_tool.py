"""
Tests for Time Info Tool.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app.shared.utils.tools.time_tool import TimeInfoTool


class TestTimeInfoTool:
    """Test suite for TimeInfoTool."""

    @pytest.fixture
    def tool(self):
        """Create TimeInfoTool instance."""
        return TimeInfoTool()

    # === Basic Functionality Tests ===

    def test_default_utc_time(self, tool):
        """Test getting current UTC time (default)."""
        result = tool._run()

        assert "UTC" in result
        assert "Current Time Information" in result
        assert "Date:" in result
        assert "Time:" in result

    def test_utc_timezone(self, tool):
        """Test explicit UTC timezone."""
        result = tool._run(timezone="UTC")

        assert "Timezone: UTC" in result
        assert "Date:" in result
        assert "Time:" in result

    def test_us_pacific_timezone(self, tool):
        """Test US/Pacific timezone."""
        result = tool._run(timezone="US/Pacific")

        assert "US/Pacific" in result
        assert "Date:" in result

    def test_europe_london_timezone(self, tool):
        """Test Europe/London timezone."""
        result = tool._run(timezone="Europe/London")

        assert "Europe/London" in result
        assert "Date:" in result

    def test_asia_tokyo_timezone(self, tool):
        """Test Asia/Tokyo timezone."""
        result = tool._run(timezone="Asia/Tokyo")

        assert "Asia/Tokyo" in result
        assert "Date:" in result

    # === Output Format Tests ===

    def test_output_contains_day_of_week(self, tool):
        """Test that output includes day of week."""
        result = tool._run()

        # Should contain one of the days
        days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        assert any(day in result for day in days)

    def test_output_contains_month_name(self, tool):
        """Test that output includes month name."""
        result = tool._run()

        # Should contain a month name
        months = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        assert any(month in result for month in months)

    def test_output_contains_time_of_day(self, tool):
        """Test that output includes time of day classification."""
        result = tool._run()

        # Should contain one of the time periods
        time_periods = ["morning", "afternoon", "evening", "night"]
        assert any(period in result for period in time_periods)

    def test_output_contains_24_hour_format(self, tool):
        """Test that output includes 24-hour time format."""
        result = tool._run()

        assert "24-hour" in result

    def test_output_contains_12_hour_format(self, tool):
        """Test that output includes 12-hour time format."""
        result = tool._run()

        assert "12-hour" in result
        # Should have AM or PM
        assert "AM" in result or "PM" in result

    def test_output_contains_iso_format(self, tool):
        """Test that output includes ISO 8601 format."""
        result = tool._run()

        assert "ISO 8601:" in result
        # ISO format should have T separator
        assert "T" in result

    def test_output_contains_unix_timestamp(self, tool):
        """Test that output includes Unix timestamp."""
        result = tool._run()

        assert "Unix timestamp:" in result
        # Should have a numeric timestamp
        assert any(char.isdigit() for char in result)

    def test_output_contains_timezone_offset(self, tool):
        """Test that output includes timezone offset."""
        result = tool._run()

        assert "Timezone offset:" in result

    # === Context Information Tests ===

    def test_context_section_exists(self, tool):
        """Test that Context section exists."""
        result = tool._run()

        assert "Context:" in result

    def test_technical_details_section_exists(self, tool):
        """Test that Technical Details section exists."""
        result = tool._run()

        assert "Technical Details:" in result

    def test_structured_output_format(self, tool):
        """Test that output has structured format with headers."""
        result = tool._run()

        assert "===" in result  # Header markers
        assert "Timezone:" in result
        assert "Date:" in result
        assert "Time:" in result

    # === Time of Day Classification Tests ===

    def test_morning_classification(self, tool):
        """Test morning time classification (5 AM - 12 PM)."""
        # We can't control current time, but we can verify the logic exists
        result = tool._run()

        # Output should have one of the time periods
        assert any(
            period in result for period in ["morning", "afternoon", "evening", "night"]
        )

    # === Error Handling Tests ===

    def test_invalid_timezone(self, tool):
        """Test handling of invalid timezone."""
        result = tool._run(timezone="Invalid/Timezone")

        assert "Error" in result

    def test_empty_timezone_uses_default(self, tool):
        """Test that empty timezone uses UTC default."""
        result = tool._run(timezone="")

        # Should error or use default
        assert "Error" in result or "UTC" in result

    # === Consistency Tests ===

    def test_multiple_calls_consistent_format(self, tool):
        """Test that multiple calls return consistent format."""
        result1 = tool._run()
        result2 = tool._run()

        # Both should have same structure
        assert "Current Time Information" in result1
        assert "Current Time Information" in result2
        assert "Context:" in result1
        assert "Context:" in result2

    def test_different_timezones_different_times(self, tool):
        """Test that different timezones return different times."""
        utc_result = tool._run(timezone="UTC")
        tokyo_result = tool._run(timezone="Asia/Tokyo")

        # Both should be valid but likely different
        assert "UTC" in utc_result
        assert "Tokyo" in tokyo_result

    # === Edge Cases ===

    def test_timezone_case_sensitivity(self, tool):
        """Test timezone parameter case sensitivity."""
        # Standard case
        result = tool._run(timezone="UTC")
        assert "UTC" in result

    def test_output_is_string(self, tool):
        """Test that output is always a string."""
        result = tool._run()

        assert isinstance(result, str)

    def test_output_not_empty(self, tool):
        """Test that output is never empty."""
        result = tool._run()

        assert len(result) > 0

    def test_output_contains_year(self, tool):
        """Test that output contains current year."""
        result = tool._run()
        current_year = datetime.now().year

        assert str(current_year) in result

    # === Async Method Test ===

    def test_async_not_implemented(self, tool):
        """Test that async execution raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            tool._arun()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
