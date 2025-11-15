"""
Database Query Tool - LangChain tool wrapper for Supabase MCP Server.
"""

import asyncio
import json
import logging
from typing import Type

try:
    from langchain.tools import BaseTool
    from pydantic import BaseModel, Field

    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Create minimal fallback classes for when LangChain is not available
    class BaseTool:
        def __init__(self):
            self.name = "database_query"
            self.description = "Database query tool (LangChain not available)"
            self.args_schema = None

    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def Field(*args, **kwargs):  # pylint: disable=unused-argument
        return None

    LANGCHAIN_AVAILABLE = False

from app.shared.services.MCPDatabaseService import (
    MCPAuthenticationError,
    MCPConnectionError,
    MCPDatabaseServiceError,
    MCPTimeoutError,
    MCPValidationError,
    get_mcp_database_service,
)

logger = logging.getLogger(__name__)


class DatabaseQueryInput(BaseModel):
    """Input schema for Database Query Tool."""

    def __init__(
        self,
        query: str = "",
        include_analysis: bool = True,
        use_cache: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.query = query
        self.include_analysis = include_analysis
        self.use_cache = use_cache

    if LANGCHAIN_AVAILABLE:
        query: str = Field(
            ...,
            description=(
                "Natural language description of what data you want to retrieve. "
                "Examples: 'Show me all users created in the last week', "
                "'What are the top 10 most active sessions?', "
                "'How many conversations happened yesterday?', "
                "'List all tables in the database'"
            ),
        )

        include_analysis: bool = Field(
            default=True,
            description="Whether to include AI-powered analysis of the results",
        )

        use_cache: bool = Field(
            default=True,
            description="Whether to use cached results for identical queries",
        )


class DatabaseQueryTool(BaseTool):
    """
    Database Query Tool using Official Supabase MCP Server.

    Provides secure, performant database querying with AI-powered analysis.
    """

    name: str = "database_query"
    description: str = """
    Query the Supabase PostgreSQL database using natural language.

    This tool uses the official Supabase MCP Server to provide:
    - Secure SQL query generation from natural language
    - AI-powered data analysis and synthesis
    - High-performance caching and connection pooling
    - Full compliance with security best practices

    Use this when you need to retrieve, analyze, or understand data from the database.
    """
    args_schema: Type[BaseModel] = DatabaseQueryInput

    def __init__(self):
        """Initialize Database Query Tool."""
        super().__init__()

        try:
            # Store MCP service as a private attribute
            self._mcp_service = get_mcp_database_service()
            logger.info("DatabaseQueryTool initialized with MCP server")
        except Exception as e:
            logger.error(f"Failed to initialize DatabaseQueryTool: {e}")
            raise

    def _run(
        self, query: str, include_analysis: bool = True, use_cache: bool = True
    ) -> str:
        """
        Execute database query.

        Args:
            query: Natural language query description
            include_analysis: Whether to include AI analysis
            use_cache: Whether to use cached results

        Returns:
            Query results with optional AI analysis
        """
        try:
            # Run async method in sync context with proper event loop handling
            return self._run_async_safe(
                self._mcp_service.execute_query(
                    query=query, include_analysis=include_analysis, use_cache=use_cache
                )
            )
        except MCPValidationError as e:
            return f"Query validation error: {str(e)}"
        except MCPAuthenticationError as e:
            return f"Authentication error: {str(e)}"
        except MCPConnectionError as e:
            return f"Connection error: {str(e)}"
        except MCPTimeoutError as e:
            return f"Request timeout: {str(e)}"
        except MCPDatabaseServiceError as e:
            return f"Database query error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in DatabaseQueryTool: {e}")
            return f"Unexpected error: {str(e)}"

    def _arun(self, **kwargs):
        """Async execution not supported."""
        raise NotImplementedError("DatabaseQueryTool does not support async execution")

    def _run_async_safe(self, coro):
        """
        Safely run async coroutine in sync context.
        Handles existing event loops properly.
        """
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, we need to use a different approach
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, coro)
                    return future.result()
            else:
                # Loop exists but not running, we can use it
                return loop.run_until_complete(coro)
        except RuntimeError:
            # No event loop exists, create a new one
            return asyncio.run(coro)

    def list_tables(self) -> str:
        """
        List all available tables in the database.

        Returns:
            Comma-separated list of table names
        """
        try:
            tables = self._run_async_safe(self._mcp_service.list_tables())
            if not tables:
                return "No tables found in the database"

            return f"Available tables: {', '.join(tables)}"
        except Exception as e:
            logger.error(f"Failed to list tables: {e}")
            return f"Error listing tables: {str(e)}"

    def describe_table(self, table_name: str) -> str:
        """
        Get detailed information about a specific table.

        Args:
            table_name: Name of the table to describe

        Returns:
            Table schema information
        """
        try:
            schema_info = self._run_async_safe(
                self._mcp_service.describe_table(table_name)
            )
            return f"Table '{table_name}' schema:\n{json.dumps(schema_info, indent=2, default=str)}"
        except Exception as e:
            logger.error(f"Failed to describe table {table_name}: {e}")
            return f"Error describing table: {str(e)}"

    def get_schema_info(self) -> str:
        """
        Get comprehensive schema information for the entire database.

        Returns:
            Complete database schema information
        """
        try:
            schema_info = self._run_async_safe(self._mcp_service.get_schema_info())
            return f"Database schema:\n{json.dumps(schema_info, indent=2, default=str)}"
        except Exception as e:
            logger.error(f"Failed to get schema info: {e}")
            return f"Error getting schema info: {str(e)}"
