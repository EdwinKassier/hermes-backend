"""
Firecrawl Tool - Search and Scrape using Firecrawl API.
"""

import logging
import os
from typing import Literal, Optional, Type

try:
    from langchain.tools import BaseTool
    from pydantic import BaseModel, Field

    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Fallback/Mock classes for environments without LangChain installed
    class BaseTool:
        def __init__(self):
            self.name = "firecrawl"
            self.description = "Search and Scrape tool"
            self.args_schema = None

    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def Field(*args, **kwargs):
        return None

    LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)


class FirecrawlInput(BaseModel):
    """Input schema for Firecrawl Tool."""

    def __init__(
        self,
        query: str = "",
        url: str = "",
        mode: str = "search",
        max_results: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.query = query
        self.url = url
        self.mode = mode
        self.max_results = max_results

    if LANGCHAIN_AVAILABLE:
        mode: Literal["search", "scrape"] = Field(
            default="search",
            description=(
                "The operation to perform: 'search' to find information, "
                "or 'scrape' to extract content from a specific URL."
            ),
        )

        query: Optional[str] = Field(
            default="",
            description=(
                "Search query. Required if mode is 'search'. "
                "Example: 'latest AI developments'"
            ),
        )

        url: Optional[str] = Field(
            default="",
            description=(
                "Target URL. Required if mode is 'scrape'. "
                "Example: 'https://example.com/article'"
            ),
        )

        max_results: int = Field(
            default=5,
            description="Maximum number of search results to return (1-10). Only used in 'search' mode.",
        )


class FirecrawlTool(BaseTool):
    """
    Firecrawl Tool for Web Search and Scraping.

    Allows agents to:
    1. Search the web for URLs and snippets.
    2. Scrape full content (markdown) from specific URLs.
    """

    name: str = "firecrawl"
    description: str = """
    Access the internet to find information or read web pages.

    MODES:
    - 'search': Use to find news, facts, or URLs. Returns a list of results.
      Input: query="topic"

    - 'scrape': Use to read the full text of a specific URL. Returns markdown.
      Input: url="https://...", mode="scrape"
    """
    args_schema: Type[BaseModel] = FirecrawlInput
    api_key: Optional[str] = None

    def __init__(self):
        """Initialize Firecrawl Tool."""
        super().__init__()

        self.api_key = os.environ.get("FIRECRAWL_API_KEY")
        if not self.api_key:
            logger.warning("FIRECRAWL_API_KEY not set - tool will be limited")

    def _run(
        self, mode: str = "search", query: str = "", url: str = "", max_results: int = 5
    ) -> str:
        """
        Execute Firecrawl API request.

        Args:
            mode: 'search' or 'scrape'
            query: Search string (for search mode)
            url: Specific URL (for scrape mode)
            max_results: Limit for search results
        """
        if not self.api_key:
            return "Error: FIRECRAWL_API_KEY environment variable not set"

        try:
            from firecrawl import FirecrawlApp
        except ImportError:
            return "Error: firecrawl-py package not installed. Install with: pip install firecrawl-py"

        try:
            # Initialize Firecrawl client
            app = FirecrawlApp(api_key=self.api_key)

            # --- SCRAPE MODE ---
            if mode == "scrape":
                if not url:
                    return "Error: 'url' parameter is required for scrape mode."

                # Scrape URL for Markdown
                # Docs: https://docs.firecrawl.dev/features/scrape
                logger.info(f"Scraping URL: {url}")
                scrape_result = app.scrape(url=url, formats=["markdown"])

                if not scrape_result:
                    return f"Failed to scrape content from {url}"

                # Extract data from Document object
                # The SDK returns a Document object with attributes, not a dict
                try:
                    markdown = getattr(scrape_result, "markdown", "")
                    metadata = getattr(scrape_result, "metadata", {})
                    title = (
                        metadata.get("title", "No Title")
                        if isinstance(metadata, dict)
                        else "No Title"
                    )
                except Exception as e:
                    logger.error(f"Error extracting data from scrape result: {e}")
                    return f"Error extracting content from {url}: {str(e)}"

                if not markdown:
                    return f"No markdown content found at {url}"

                return f"Title: {title}\nURL: {url}\n\n{markdown}"

            # --- SEARCH MODE ---
            else:
                if not query:
                    return "Error: 'query' parameter is required for search mode."

                # Cap max_results at 10
                limit = min(max(int(max_results), 1), 10)

                logger.info(f"Searching for: {query}")
                search_result = app.search(query, limit=limit)

                # Extract web results
                if not search_result or not hasattr(search_result, "web"):
                    return f"No results found for: {query}"

                results = search_result.web if search_result.web else []

                if not results:
                    return f"No results found for: {query}"

                # Format results
                formatted_results = [f"Search results for: {query}\n"]

                for i, result in enumerate(results[:limit], 1):
                    title = getattr(result, "title", "No title")
                    res_url = getattr(result, "url", "No URL")
                    description = getattr(result, "description", "")

                    if description and len(description) > 200:
                        description = description[:200] + "..."

                    if not description:
                        description = "No description available"

                    formatted_results.append(f"\n{i}. {title}")
                    formatted_results.append(f"   URL: {res_url}")
                    formatted_results.append(f"   {description}")

                return "\n".join(formatted_results)

        except Exception as e:
            logger.error(f"Firecrawl error: {e}")
            return f"Error performing Firecrawl request: {str(e)}"

    def _arun(self, **kwargs):
        """Async execution not supported."""
        raise NotImplementedError("FirecrawlTool does not support async execution")
