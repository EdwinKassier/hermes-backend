"""
Web Search Tool - Search the internet using Firecrawl API.
"""

import logging
import os
from typing import Optional, Type

try:
    from langchain.tools import BaseTool
    from pydantic import BaseModel, Field

    LANGCHAIN_AVAILABLE = True
except ImportError:

    class BaseTool:
        def __init__(self):
            self.name = "web_search"
            self.description = "Web search tool (LangChain not available)"
            self.args_schema = None

    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def Field(*args, **kwargs):
        return None

    LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)


class WebSearchInput(BaseModel):
    """Input schema for Web Search Tool."""

    def __init__(self, query: str = "", max_results: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.query = query
        self.max_results = max_results

    if LANGCHAIN_AVAILABLE:
        query: str = Field(
            ...,
            description=(
                "Search query to find information on the internet. "
                "Examples: 'latest AI developments', 'Python best practices', "
                "'quantum computing news'"
            ),
        )

        max_results: int = Field(
            default=5,
            description="Maximum number of search results to return (1-10)",
        )


class WebSearchTool(BaseTool):
    """
    Web Search Tool using Firecrawl API.

    Provides internet search capabilities for research and information gathering.
    """

    name: str = "web_search"
    description: str = """
    Search the internet for current information, news, and research.

    Use this tool when you need to:
    - Find current information or news
    - Research a topic
    - Verify facts
    - Get the latest developments on a subject

    Returns search results with titles, URLs, and content snippets.
    """
    args_schema: Type[BaseModel] = WebSearchInput
    api_key: Optional[str] = None  # Add as class attribute for Pydantic

    def __init__(self):
        """Initialize Web Search Tool."""
        super().__init__()

        # Check for Firecrawl API key
        self.api_key = os.environ.get("FIRECRAWL_API_KEY")
        if not self.api_key:
            logger.warning("FIRECRAWL_API_KEY not set - web search will be limited")

    def _run(self, query: str, max_results: int = 5) -> str:
        """
        Execute web search using Firecrawl.

        Args:
            query: Search query
            max_results: Maximum number of results to return (capped at 10)

        Returns:
            Formatted search results
        """
        if not self.api_key:
            return (
                "Web search unavailable: FIRECRAWL_API_KEY environment variable not set"
            )

        try:
            from firecrawl import FirecrawlApp
        except ImportError:
            return "Web search unavailable: firecrawl-py package not installed. Install with: pip install firecrawl-py"

        try:
            # Initialize Firecrawl client
            app = FirecrawlApp(api_key=self.api_key)

            # Cap max_results at 10 and ensure it's an integer
            # LLMs sometimes return floats like 5.0
            limit = min(max(int(max_results), 1), 10)

            # Perform search - returns SearchData object
            search_result = app.search(query, limit=limit)

            # Extract web results from SearchData object
            if not search_result or not hasattr(search_result, "web"):
                return f"No results found for: {query}"

            results = search_result.web if search_result.web else []

            if not results:
                return f"No results found for: {query}"

            # Format results
            formatted_results = [f"Search results for: {query}\n"]

            for i, result in enumerate(results[:limit], 1):
                # SearchResultWeb has url, title, description attributes
                title = getattr(result, "title", "No title")
                url = getattr(result, "url", "No URL")
                description = getattr(result, "description", "")

                # Truncate description if too long
                if description and len(description) > 200:
                    description = description[:200] + "..."

                if not description:
                    description = "No description available"

                formatted_results.append(f"\n{i}. {title}")
                formatted_results.append(f"   URL: {url}")
                formatted_results.append(f"   {description}")

            return "\n".join(formatted_results)

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"Error performing web search: {str(e)}"

    def _arun(self, **kwargs):
        """Async execution not supported."""
        raise NotImplementedError("WebSearchTool does not support async execution")
