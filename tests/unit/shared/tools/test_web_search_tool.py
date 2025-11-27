"""
Tests for Web Search Tool.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from app.shared.utils.tools.web_search_tool import WebSearchTool


class TestWebSearchTool:
    """Test suite for WebSearchTool."""

    @pytest.fixture
    def tool_with_api_key(self):
        """Create WebSearchTool with mocked API key."""
        with patch.dict("os.environ", {"FIRECRAWL_API_KEY": "test_api_key"}):
            return WebSearchTool()

    @pytest.fixture
    def tool_without_api_key(self):
        """Create WebSearchTool without API key."""
        with patch.dict("os.environ", {}, clear=True):
            return WebSearchTool()

    # === Initialization Tests ===

    def test_initialization_with_api_key(self, tool_with_api_key):
        """Test tool initializes with API key."""
        assert tool_with_api_key.api_key == "test_api_key"

    def test_initialization_without_api_key(self, tool_without_api_key):
        """Test tool initializes without API key (logs warning)."""
        assert tool_without_api_key.api_key is None

    # === API Key Validation Tests ===

    def test_no_api_key_returns_error(self, tool_without_api_key):
        """Test that missing API key returns error message."""
        result = tool_without_api_key._run("test query")

        assert "unavailable" in result
        assert "FIRECRAWL_API_KEY" in result

    # === Successful Search Tests ===

    def test_successful_search(self, tool_with_api_key):
        """Test successful web search."""
        # Mock the firecrawl import and FirecrawlApp
        with patch("firecrawl.FirecrawlApp") as mock_fc:
            # Create mock app instance
            mock_app = MagicMock()
            mock_fc.return_value = mock_app

            # Create mock search result with web attribute
            mock_result = MagicMock()
            mock_result.web = [
                MagicMock(
                    title="Test Result 1",
                    url="https://example.com/1",
                    description="Test description 1",
                ),
                MagicMock(
                    title="Test Result 2",
                    url="https://example.com/2",
                    description="Test description 2",
                ),
            ]
            mock_app.search.return_value = mock_result

            result = tool_with_api_key._run("test query", max_results=5)

            assert "Test Result 1" in result
            assert "Test Result 2" in result
            assert "https://example.com/1" in result
            assert "https://example.com/2" in result
            assert "Test description 1" in result

    def test_search_with_max_results(self, tool_with_api_key):
        """Test search respects max_results parameter."""
        with patch("firecrawl.FirecrawlApp") as mock_fc:
            mock_app = MagicMock()
            mock_fc.return_value = mock_app

            mock_result = MagicMock()
            mock_result.web = [
                MagicMock(
                    title=f"Result {i}",
                    url=f"https://example.com/{i}",
                    description=f"Desc {i}",
                )
                for i in range(10)
            ]
            mock_app.search.return_value = mock_result

            result = tool_with_api_key._run("test query", max_results=3)

            # Should only show first 3 results
            assert "Result 0" in result
            assert "Result 1" in result
            assert "Result 2" in result
            # Should not show result 3 or beyond
            assert "Result 3" not in result

    def test_search_caps_at_10_results(self, tool_with_api_key):
        """Test search caps max_results at 10."""
        with patch("firecrawl.FirecrawlApp") as mock_fc:
            mock_app = MagicMock()
            mock_fc.return_value = mock_app

            mock_result = MagicMock()
            mock_result.web = []
            mock_app.search.return_value = mock_result

            tool_with_api_key._run("test query", max_results=20)

            # Should call with limit=10 (capped)
            call_args = mock_app.search.call_args
            assert call_args[1]["limit"] == 10

    # === Result Formatting Tests ===

    def test_result_formatting(self, tool_with_api_key):
        """Test that results are formatted correctly."""
        with patch("firecrawl.FirecrawlApp") as mock_fc:
            mock_app = MagicMock()
            mock_fc.return_value = mock_app

            mock_result = MagicMock()
            mock_result.web = [
                MagicMock(
                    title="Test Title",
                    url="https://example.com",
                    description="Test Description",
                )
            ]
            mock_app.search.return_value = mock_result

            result = tool_with_api_key._run("test query")

            assert "Search results for: test query" in result
            assert "1. Test Title" in result
            assert "URL: https://example.com" in result
            assert "Test Description" in result

    def test_result_with_content_fallback(self, tool_with_api_key):
        """Test that long descriptions are truncated."""
        with patch("firecrawl.FirecrawlApp") as mock_fc:
            mock_app = MagicMock()
            mock_fc.return_value = mock_app

            long_desc = (
                "This is a long content that should be truncated to 200 characters maximum. "
                * 5
            )
            mock_result = MagicMock()
            mock_result.web = [
                MagicMock(
                    title="Test Title", url="https://example.com", description=long_desc
                )
            ]
            mock_app.search.return_value = mock_result

            result = tool_with_api_key._run("test query")

            # Should truncate to 200 chars
            assert "Test Title" in result
            assert "..." in result  # Truncation indicator

    def test_result_with_no_description_or_content(self, tool_with_api_key):
        """Test handling when neither description nor content available."""
        with patch("firecrawl.FirecrawlApp") as mock_fc:
            mock_app = MagicMock()
            mock_fc.return_value = mock_app

            mock_result = MagicMock()
            mock_result.web = [
                MagicMock(title="Test Title", url="https://example.com", description="")
            ]
            mock_app.search.return_value = mock_result

            result = tool_with_api_key._run("test query")

            assert "No description available" in result

    # === Error Handling Tests ===

    def test_no_results_found(self, tool_with_api_key):
        """Test handling when no results found."""
        with patch("firecrawl.FirecrawlApp") as mock_fc:
            mock_app = MagicMock()
            mock_fc.return_value = mock_app

            mock_result = MagicMock()
            mock_result.web = []
            mock_app.search.return_value = mock_result

            result = tool_with_api_key._run("test query")

            assert "No results found" in result

    def test_unsuccessful_search(self, tool_with_api_key):
        """Test handling when search returns None."""
        with patch("firecrawl.FirecrawlApp") as mock_fc:
            mock_app = MagicMock()
            mock_fc.return_value = mock_app
            mock_app.search.return_value = None

            result = tool_with_api_key._run("test query")

            assert "No results found" in result

    def test_search_exception(self, tool_with_api_key):
        """Test handling of search exceptions."""
        with patch("firecrawl.FirecrawlApp") as mock_fc:
            mock_app = MagicMock()
            mock_fc.return_value = mock_app
            mock_app.search.side_effect = Exception("API Error")

            result = tool_with_api_key._run("test query")

            assert "Error" in result
            assert "API Error" in result

    def test_firecrawl_not_installed(self, tool_with_api_key):
        """Test handling when firecrawl package not installed."""
        with patch.dict("sys.modules", {"firecrawl": None}):
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module named 'firecrawl'"),
            ):
                result = tool_with_api_key._run("test query")

                assert "unavailable" in result
                assert "firecrawl-py" in result

    # === Input Validation Tests ===

    def test_empty_query(self, tool_with_api_key):
        """Test search with empty query."""
        with patch("firecrawl.FirecrawlApp") as mock_fc:
            mock_app = MagicMock()
            mock_fc.return_value = mock_app

            mock_result = MagicMock()
            mock_result.web = []
            mock_app.search.return_value = mock_result

            result = tool_with_api_key._run("")

            # Should still attempt search (Firecrawl handles validation)
            assert isinstance(result, str)

    def test_special_characters_in_query(self, tool_with_api_key):
        """Test search with special characters."""
        with patch("firecrawl.FirecrawlApp") as mock_fc:
            mock_app = MagicMock()
            mock_fc.return_value = mock_app

            mock_result = MagicMock()
            mock_result.web = []
            mock_app.search.return_value = mock_result

            result = tool_with_api_key._run("test @#$% query")

            # Should handle special characters
            assert isinstance(result, str)

    # === Edge Cases ===

    def test_max_results_zero(self, tool_with_api_key):
        """Test search with max_results=0."""
        with patch("firecrawl.FirecrawlApp") as mock_fc:
            mock_app = MagicMock()
            mock_fc.return_value = mock_app

            mock_result = MagicMock()
            mock_result.web = []
            mock_app.search.return_value = mock_result

            result = tool_with_api_key._run("test query", max_results=0)

            # Should handle gracefully (min will make it 1)
            assert isinstance(result, str)
            # Verify it was called with limit=1 (min of 0 and 1)
            call_args = mock_app.search.call_args
            assert call_args[1]["limit"] == 1

    def test_max_results_negative(self, tool_with_api_key):
        """Test search with negative max_results."""
        with patch("firecrawl.FirecrawlApp") as mock_fc:
            mock_app = MagicMock()
            mock_fc.return_value = mock_app

            mock_result = MagicMock()
            mock_result.web = []
            mock_app.search.return_value = mock_result

            result = tool_with_api_key._run("test query", max_results=-1)

            # Should handle gracefully (min will make it 1)
            assert isinstance(result, str)
            call_args = mock_app.search.call_args
            assert call_args[1]["limit"] == 1

    # === Async Method Test ===

    def test_async_not_implemented(self, tool_with_api_key):
        """Test that async execution raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            tool_with_api_key._arun()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
