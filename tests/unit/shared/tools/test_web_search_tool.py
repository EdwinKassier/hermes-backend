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

    @patch("app.shared.utils.tools.web_search_tool.FirecrawlApp")
    def test_successful_search(self, mock_firecrawl_class, tool_with_api_key):
        """Test successful web search."""
        # Mock Firecrawl response
        mock_app = MagicMock()
        mock_firecrawl_class.return_value = mock_app
        mock_app.search.return_value = {
            "success": True,
            "data": [
                {
                    "title": "Test Result 1",
                    "url": "https://example.com/1",
                    "description": "Test description 1",
                },
                {
                    "title": "Test Result 2",
                    "url": "https://example.com/2",
                    "description": "Test description 2",
                },
            ],
        }

        result = tool_with_api_key._run("test query", max_results=5)

        assert "Test Result 1" in result
        assert "Test Result 2" in result
        assert "https://example.com/1" in result
        assert "https://example.com/2" in result
        assert "Test description 1" in result

    @patch("app.shared.utils.tools.web_search_tool.FirecrawlApp")
    def test_search_with_max_results(self, mock_firecrawl_class, tool_with_api_key):
        """Test search respects max_results parameter."""
        mock_app = MagicMock()
        mock_firecrawl_class.return_value = mock_app
        mock_app.search.return_value = {
            "success": True,
            "data": [
                {
                    "title": f"Result {i}",
                    "url": f"https://example.com/{i}",
                    "description": f"Desc {i}",
                }
                for i in range(10)
            ],
        }

        result = tool_with_api_key._run("test query", max_results=3)

        # Should only show first 3 results
        assert "Result 0" in result
        assert "Result 1" in result
        assert "Result 2" in result
        # Should not show result 3 or beyond
        assert "Result 3" not in result

    @patch("app.shared.utils.tools.web_search_tool.FirecrawlApp")
    def test_search_caps_at_10_results(self, mock_firecrawl_class, tool_with_api_key):
        """Test search caps max_results at 10."""
        mock_app = MagicMock()
        mock_firecrawl_class.return_value = mock_app

        tool_with_api_key._run("test query", max_results=20)

        # Should call with limit=10 (capped)
        call_args = mock_app.search.call_args
        assert call_args[1]["params"]["limit"] == 10

    # === Result Formatting Tests ===

    @patch("app.shared.utils.tools.web_search_tool.FirecrawlApp")
    def test_result_formatting(self, mock_firecrawl_class, tool_with_api_key):
        """Test that results are formatted correctly."""
        mock_app = MagicMock()
        mock_firecrawl_class.return_value = mock_app
        mock_app.search.return_value = {
            "success": True,
            "data": [
                {
                    "title": "Test Title",
                    "url": "https://example.com",
                    "description": "Test Description",
                }
            ],
        }

        result = tool_with_api_key._run("test query")

        assert "Search results for: test query" in result
        assert "1. Test Title" in result
        assert "URL: https://example.com" in result
        assert "Test Description" in result

    @patch("app.shared.utils.tools.web_search_tool.FirecrawlApp")
    def test_result_with_content_fallback(
        self, mock_firecrawl_class, tool_with_api_key
    ):
        """Test that content is used when description is missing."""
        mock_app = MagicMock()
        mock_firecrawl_class.return_value = mock_app
        mock_app.search.return_value = {
            "success": True,
            "data": [
                {
                    "title": "Test Title",
                    "url": "https://example.com",
                    "content": "This is a long content that should be truncated to 200 characters maximum. "
                    * 5,
                }
            ],
        }

        result = tool_with_api_key._run("test query")

        # Should use content and truncate to 200 chars
        assert "Test Title" in result
        assert "..." in result  # Truncation indicator

    @patch("app.shared.utils.tools.web_search_tool.FirecrawlApp")
    def test_result_with_no_description_or_content(
        self, mock_firecrawl_class, tool_with_api_key
    ):
        """Test handling when neither description nor content available."""
        mock_app = MagicMock()
        mock_firecrawl_class.return_value = mock_app
        mock_app.search.return_value = {
            "success": True,
            "data": [{"title": "Test Title", "url": "https://example.com"}],
        }

        result = tool_with_api_key._run("test query")

        assert "No description available" in result

    # === Error Handling Tests ===

    @patch("app.shared.utils.tools.web_search_tool.FirecrawlApp")
    def test_no_results_found(self, mock_firecrawl_class, tool_with_api_key):
        """Test handling when no results found."""
        mock_app = MagicMock()
        mock_firecrawl_class.return_value = mock_app
        mock_app.search.return_value = {"success": True, "data": []}

        result = tool_with_api_key._run("test query")

        assert "No results found" in result

    @patch("app.shared.utils.tools.web_search_tool.FirecrawlApp")
    def test_unsuccessful_search(self, mock_firecrawl_class, tool_with_api_key):
        """Test handling when search is unsuccessful."""
        mock_app = MagicMock()
        mock_firecrawl_class.return_value = mock_app
        mock_app.search.return_value = {"success": False}

        result = tool_with_api_key._run("test query")

        assert "No results found" in result

    @patch("app.shared.utils.tools.web_search_tool.FirecrawlApp")
    def test_search_exception(self, mock_firecrawl_class, tool_with_api_key):
        """Test handling of search exceptions."""
        mock_app = MagicMock()
        mock_firecrawl_class.return_value = mock_app
        mock_app.search.side_effect = Exception("API Error")

        result = tool_with_api_key._run("test query")

        assert "Error" in result
        assert "API Error" in result

    def test_firecrawl_not_installed(self, tool_with_api_key):
        """Test handling when firecrawl package not installed."""
        with patch(
            "app.shared.utils.tools.web_search_tool.FirecrawlApp",
            side_effect=ImportError,
        ):
            result = tool_with_api_key._run("test query")

            assert "unavailable" in result
            assert "firecrawl-py" in result

    # === Input Validation Tests ===

    @patch("app.shared.utils.tools.web_search_tool.FirecrawlApp")
    def test_empty_query(self, mock_firecrawl_class, tool_with_api_key):
        """Test search with empty query."""
        mock_app = MagicMock()
        mock_firecrawl_class.return_value = mock_app
        mock_app.search.return_value = {"success": True, "data": []}

        result = tool_with_api_key._run("")

        # Should still attempt search (Firecrawl handles validation)
        assert isinstance(result, str)

    @patch("app.shared.utils.tools.web_search_tool.FirecrawlApp")
    def test_special_characters_in_query(self, mock_firecrawl_class, tool_with_api_key):
        """Test search with special characters."""
        mock_app = MagicMock()
        mock_firecrawl_class.return_value = mock_app
        mock_app.search.return_value = {"success": True, "data": []}

        result = tool_with_api_key._run("test @#$% query")

        # Should handle special characters
        assert isinstance(result, str)

    # === Edge Cases ===

    @patch("app.shared.utils.tools.web_search_tool.FirecrawlApp")
    def test_max_results_zero(self, mock_firecrawl_class, tool_with_api_key):
        """Test search with max_results=0."""
        mock_app = MagicMock()
        mock_firecrawl_class.return_value = mock_app
        mock_app.search.return_value = {"success": True, "data": []}

        result = tool_with_api_key._run("test query", max_results=0)

        # Should handle gracefully
        assert isinstance(result, str)

    @patch("app.shared.utils.tools.web_search_tool.FirecrawlApp")
    def test_max_results_negative(self, mock_firecrawl_class, tool_with_api_key):
        """Test search with negative max_results."""
        mock_app = MagicMock()
        mock_firecrawl_class.return_value = mock_app
        mock_app.search.return_value = {"success": True, "data": []}

        result = tool_with_api_key._run("test query", max_results=-1)

        # Should handle gracefully (min will make it 0 or handle error)
        assert isinstance(result, str)

    # === Async Method Test ===

    def test_async_not_implemented(self, tool_with_api_key):
        """Test that async execution raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            tool_with_api_key._arun()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
