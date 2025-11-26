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

from app.shared.services.SupabaseDatabaseService import (
    SupabaseAuthenticationError,
    SupabaseConnectionError,
    SupabaseDatabaseService,
    SupabaseDatabaseServiceError,
    SupabaseTimeoutError,
    SupabaseValidationError,
    get_supabase_database_service,
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
    Tool for querying the database using natural language.

    This tool allows agents to query the database using natural language descriptions.
    It uses an LLM to convert the description into a safe, read-only PostgREST query.
    """

    name = "database_query"
    description = "Query the database using natural language. Provide a description of what data you need."

    def __init__(self):
        """Initialize the database query tool."""
        super().__init__()
        try:
            self.db_service = get_supabase_database_service()
            logger.info("DatabaseQueryTool initialized with Supabase service")
        except Exception as e:
            logger.error(f"Failed to initialize DatabaseQueryTool: {e}")
            self.db_service = None

    def _run(self, query: str) -> str:
        """
        Execute the database query.

        Args:
            query: Natural language description of the data needed

        Returns:
            Query results or error message
        """
        if not self.db_service:
            return (
                "Error: Database service is not available. Please check configuration."
            )

        try:
            # Run async query in sync context
            # We use a new event loop if needed, or run_until_complete
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if loop.is_running():
                # If we are already in an event loop (e.g. FastAPI), we can't block it.
                # Ideally this tool should be async, but LangChain tools are often sync.
                # For now, we assume this is called in a thread or we use nest_asyncio if installed.
                # Or better, we use the async implementation _arun if supported.
                # But BaseTool usually supports _run (sync) and _arun (async).
                # Let's try to return a coroutine if possible, but _run expects str.

                # Hack: Use a separate thread to run the async code if loop is running
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        lambda: asyncio.run(self.db_service.execute_query(query))
                    )
                    return future.result()
            else:
                return loop.run_until_complete(self.db_service.execute_query(query))

        except SupabaseValidationError as e:
            return f"Validation Error: {str(e)}"
        except SupabaseAuthenticationError:
            return "Authentication Error: Failed to authenticate with database."
        except SupabaseConnectionError:
            return "Connection Error: Failed to connect to database."
        except SupabaseTimeoutError:
            return "Timeout Error: Database query timed out."
        except SupabaseDatabaseServiceError as e:
            return f"Database Error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in DatabaseQueryTool: {e}")
            return f"Error: An unexpected error occurred: {str(e)}"

    async def _arun(self, query: str) -> str:
        """
        Execute the database query asynchronously.

        Args:
            query: Natural language description of the data needed

        Returns:
            Query results or error message
        """
        if not self.db_service:
            return (
                "Error: Database service is not available. Please check configuration."
            )

        try:
            return await self.db_service.execute_query(query)
        except SupabaseValidationError as e:
            return f"Validation Error: {str(e)}"
        except SupabaseAuthenticationError:
            return "Authentication Error: Failed to authenticate with database."
        except SupabaseConnectionError:
            return "Connection Error: Failed to connect to database."
        except SupabaseTimeoutError:
            return "Timeout Error: Database query timed out."
        except SupabaseDatabaseServiceError as e:
            return f"Database Error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in DatabaseQueryTool: {e}")
            return f"Error: An unexpected error occurred: {str(e)}"

    def _describe_table(self, table_name: str) -> str:
        """
        Get detailed information about a specific table.

        NOTE: Internal utility method, not exposed to LangChain.
        Call directly if needed: tool._describe_table("table_name")

        Args:
            table_name: Name of the table to describe

        Returns:
            Table schema information
        """
        try:
            schema_info = asyncio.run(self._mcp_service.describe_table(table_name))
            return f"Table '{table_name}' schema:\n{json.dumps(schema_info, indent=2, default=str)}"
        except Exception as e:
            logger.error(f"Failed to describe table {table_name}: {e}")
            return f"Error describing table: {str(e)}"

    def _get_schema_info(self) -> str:
        """
        Get comprehensive schema information for the entire database.

        NOTE: Internal utility method, not exposed to LangChain.
        Call directly if needed: tool._get_schema_info()

        Returns:
            Complete database schema information
        """
        try:
            schema_info = asyncio.run(self._mcp_service.get_schema_info())
            return f"Database schema:\n{json.dumps(schema_info, indent=2, default=str)}"
        except Exception as e:
            logger.error(f"Failed to get schema info: {e}")
            return f"Error getting schema info: {str(e)}"
