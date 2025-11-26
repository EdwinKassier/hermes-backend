"""
Supabase Database Service - PostgREST API Implementation
Uses Supabase Python client to query the database via PostgREST.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from supabase import Client, create_client

logger = logging.getLogger(__name__)


class SupabaseDatabaseServiceError(Exception):
    """Base exception for Supabase Database Service errors."""

    pass


class SupabaseConnectionError(SupabaseDatabaseServiceError):
    """Raised when Supabase connection fails."""

    pass


class SupabaseAuthenticationError(SupabaseDatabaseServiceError):
    """Raised when Supabase authentication fails."""

    pass


class SupabaseTimeoutError(SupabaseDatabaseServiceError):
    """Raised when Supabase request times out."""

    pass


class SupabaseValidationError(SupabaseDatabaseServiceError):
    """Raised when input validation fails."""

    pass


class SupabaseDatabaseService:
    """
    Supabase Database Service using PostgREST API.

    Provides database query capabilities using Supabase's PostgREST interface.
    """

    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
    ):
        """
        Initialize Supabase Database Service.

        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase service role key
        """
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.client: Optional[Client] = None

        try:
            self.client = create_client(supabase_url, supabase_key)
            logger.info(f"Supabase Database Service initialized for {supabase_url}")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise SupabaseConnectionError(f"Initialization failed: {e}") from e

    async def execute_query(
        self, query: str, include_analysis: bool = True, use_cache: bool = True
    ) -> str:
        """
        Execute natural language query via Supabase.

        Uses an LLM to convert natural language to PostgREST queries.
        Enforces read-only access - no INSERT, UPDATE, DELETE operations.

        Args:
            query: Natural language query description
            include_analysis: Whether to include analysis
            use_cache: Whether to use cached results

        Returns:
            Formatted query results
        """
        # Input validation
        if not query or not isinstance(query, str):
            raise SupabaseValidationError("Query must be a non-empty string")

        query = query.strip()
        if len(query) < 3:
            raise SupabaseValidationError("Query too short (minimum 3 characters)")

        if len(query) > 10000:
            raise SupabaseValidationError("Query too long (maximum 10000 characters)")

        # Read-only enforcement - check for write operations
        query_lower = query.lower()
        write_keywords = [
            "insert",
            "update",
            "delete",
            "drop",
            "create",
            "alter",
            "truncate",
            "grant",
            "revoke",
        ]
        if any(keyword in query_lower for keyword in write_keywords):
            raise SupabaseValidationError(
                "Write operations are not allowed. This tool is read-only. "
                f"Detected forbidden keyword in query."
            )

        try:
            # Use LLM to convert natural language to PostgREST query
            result = await self._execute_natural_language_query(query, include_analysis)
            return result

        except SupabaseValidationError:
            raise
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise SupabaseDatabaseServiceError(f"Query execution failed: {e}") from e

    async def _execute_natural_language_query(
        self, query: str, include_analysis: bool
    ) -> str:
        """
        Convert natural language query to PostgREST and execute.

        Args:
            query: Natural language query
            include_analysis: Whether to include AI analysis

        Returns:
            Formatted results
        """
        from app.shared.services.GeminiService import GeminiService

        # Get available tables
        tables = await self._get_table_list()

        # Create prompt for LLM to convert to PostgREST query
        prompt = f"""Convert this natural language database query to a JSON query plan.

Available tables: {', '.join(tables[:20])}

User query: "{query}"

Return ONLY valid JSON (no markdown, no explanations) in this exact format:

For listing tables:
{{"operation": "list_tables"}}

For describing a table:
{{"operation": "describe", "table": "table_name"}}

For counting records:
{{"operation": "count", "table": "table_name"}}

For selecting data:
{{
  "operation": "select",
  "table": "table_name",
  "columns": "*",
  "filters": [],
  "limit": 10,
  "order_by": null,
  "explanation": "Brief description"
}}

Examples:
- "Show me projects" -> {{"operation": "select", "table": "project", "columns": "*", "limit": 10}}
- "Describe project table" -> {{"operation": "describe", "table": "project"}}
- "Count workflows" -> {{"operation": "count", "table": "workflow_entity"}}
- "Find completed test runs" -> {{"operation": "select", "table": "test_run", "columns": "*", "filters": [{{"column": "status", "operator": "eq", "value": "completed"}}], "limit": 10}}

Return ONLY the JSON object, nothing else."""

        gemini = GeminiService()
        llm_response = await asyncio.create_task(
            asyncio.to_thread(gemini.generate_gemini_response, prompt, "hermes")
        )

        # Parse LLM response
        import json
        import re

        try:
            # Remove markdown code blocks if present
            cleaned_response = llm_response.strip()
            if cleaned_response.startswith("```"):
                # Extract content between code fences
                match = re.search(
                    r"```(?:json)?\s*(\{.*?\})\s*```", cleaned_response, re.DOTALL
                )
                if match:
                    cleaned_response = match.group(1)
                else:
                    # Try to find JSON without fences
                    cleaned_response = re.sub(r"```(?:json)?", "", cleaned_response)
                    cleaned_response = cleaned_response.strip()

            # Extract JSON from response
            json_start = cleaned_response.find("{")
            json_end = cleaned_response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = cleaned_response[json_start:json_end]
                query_plan = json.loads(json_str)
                logger.info(f"Parsed query plan: {query_plan}")
            else:
                logger.error(f"No JSON found in LLM response: {llm_response[:200]}")
                query_plan = {"operation": "error", "message": "Could not parse query"}
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}, Response: {llm_response[:200]}")
            query_plan = {
                "operation": "error",
                "message": f"Invalid query plan: {str(e)}",
            }

        # Execute based on operation
        operation = query_plan.get("operation", "select")

        if operation == "list_tables":
            return await self._list_tables_impl()
        elif operation == "describe":
            table = query_plan.get("table")
            if table:
                return await self._describe_table_impl(table)
            return "Error: No table specified"
        elif operation == "count":
            table = query_plan.get("table")
            if table and table in tables:
                result = self.client.table(table).select("*", count="exact").execute()
                count = result.count if hasattr(result, "count") else len(result.data)
                return f"Table '{table}' has {count} records."
            return f"Error: Table '{table}' not found"
        elif operation == "select":
            return await self._execute_select_query(
                query_plan, tables, include_analysis
            )
        else:
            return f"Query: '{query}'\n\nCould not determine how to execute this query. Try rephrasing or being more specific."

    async def _execute_select_query(
        self, query_plan: Dict[str, Any], tables: List[str], include_analysis: bool
    ) -> str:
        """Execute a SELECT query based on the query plan."""
        table = query_plan.get("table")

        if not table or table not in tables:
            return f"Error: Table '{table}' not found. Available tables: {', '.join(tables[:10])}"

        try:
            # Build PostgREST query
            columns = query_plan.get("columns", "*")
            if isinstance(columns, list):
                columns = ",".join(columns)

            query_builder = self.client.table(table).select(columns)

            # Apply filters
            filters = query_plan.get("filters", [])
            for f in filters:
                col = f.get("column")
                op = f.get("operator", "eq")
                val = f.get("value")

                if op == "eq":
                    query_builder = query_builder.eq(col, val)
                elif op == "gt":
                    query_builder = query_builder.gt(col, val)
                elif op == "lt":
                    query_builder = query_builder.lt(col, val)
                elif op == "like":
                    query_builder = query_builder.like(col, f"%{val}%")

            # Apply limit
            limit = query_plan.get("limit", 10)
            query_builder = query_builder.limit(min(limit, 100))  # Cap at 100

            # Apply ordering
            order_by = query_plan.get("order_by")
            if order_by:
                query_builder = query_builder.order(order_by)

            # Execute query
            response = query_builder.execute()

            # Format results
            if not response.data:
                return f"No results found in table '{table}'"

            result = f"Query Results from '{table}' ({len(response.data)} records):\n\n"

            # Format as table
            if response.data:
                # Get column names
                cols = list(response.data[0].keys())

                # Add explanation if available
                explanation = query_plan.get("explanation")
                if explanation:
                    result += f"Query: {explanation}\n\n"

                # Add data
                for i, row in enumerate(response.data[:10], 1):  # Show max 10 rows
                    result += f"Record {i}:\n"
                    for col in cols[:8]:  # Show max 8 columns
                        value = row.get(col)
                        # Truncate long values
                        if isinstance(value, str) and len(value) > 100:
                            value = value[:100] + "..."
                        result += f"  {col}: {value}\n"
                    result += "\n"

                if len(response.data) > 10:
                    result += f"... and {len(response.data) - 10} more records\n"

            return result

        except Exception as e:
            logger.error(f"Failed to execute select query: {e}")
            return f"Error executing query on table '{table}': {str(e)}"

    async def _get_table_list(self) -> List[str]:
        """Get list of all tables from PostgREST OpenAPI spec."""
        try:
            # Get the OpenAPI spec from PostgREST root endpoint
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.supabase_url}/rest/v1/",
                    headers={
                        "apikey": self.supabase_key,
                        "Authorization": f"Bearer {self.supabase_key}",
                    },
                )

                if response.status_code == 200:
                    spec = response.json()
                    # Extract table names from definitions
                    if "definitions" in spec:
                        tables = list(spec["definitions"].keys())
                        # Filter out non-table definitions
                        tables = [t for t in tables if not t.startswith("_")]
                        return sorted(tables)

            return []

        except Exception as e:
            logger.warning(f"Could not fetch table list from OpenAPI spec: {e}")
            return []

    async def _list_tables_impl(self) -> str:
        """List all available tables."""
        try:
            tables = await self._get_table_list()

            if not tables:
                # If we can't get tables from RPC, try a different approach
                # List some common tables or provide a helpful message
                return (
                    "Unable to retrieve table list directly.\n\n"
                    "To view tables, you can:\n"
                    "1. Check your Supabase dashboard\n"
                    "2. Query specific tables directly if you know their names\n"
                    "3. Use Supabase SQL Editor for custom queries"
                )

            result = f"Found {len(tables)} tables in the database:\n\n"
            for i, table in enumerate(tables, 1):
                result += f"{i}. {table}\n"

            return result

        except Exception as e:
            logger.error(f"Failed to list tables: {e}")
            raise SupabaseDatabaseServiceError(f"Failed to list tables: {e}") from e

    async def list_tables(self) -> List[str]:
        """List all available tables."""
        return await self._get_table_list()

    async def describe_table(self, table_name: str) -> Dict[str, Any]:
        """Get table schema information."""
        try:
            result_str = await self._describe_table_impl(table_name)
            return {"schema": result_str}
        except Exception as e:
            logger.error(f"Failed to describe table: {e}")
            raise SupabaseDatabaseServiceError(f"Failed to describe table: {e}") from e

    async def _describe_table_impl(self, table_name: str) -> str:
        """Get table schema information as formatted string."""
        try:
            # Query the table with limit 1 to get column structure
            response = self.client.table(table_name).select("*").limit(1).execute()

            if response.data and len(response.data) > 0:
                columns = list(response.data[0].keys())
                result = f"Table: {table_name}\n\n"
                result += f"Columns ({len(columns)}):\n"
                for col in columns:
                    # Get sample value to infer type
                    sample = response.data[0].get(col)
                    col_type = (
                        type(sample).__name__ if sample is not None else "unknown"
                    )
                    result += f"  - {col} ({col_type})\n"
                return result
            else:
                return f"Table '{table_name}' exists but has no data to infer schema."

        except Exception as e:
            return f"Error describing table '{table_name}': {str(e)}"

    async def _get_schema_info_impl(self) -> str:
        """Get comprehensive schema information."""
        try:
            tables = await self._get_table_list()

            if not tables:
                return "Unable to retrieve schema information."

            result = f"Database Schema Overview\n"
            result += f"=" * 50 + "\n\n"
            result += f"Total Tables: {len(tables)}\n\n"
            result += "Tables:\n"
            for table in tables:
                result += f"  - {table}\n"

            return result

        except Exception as e:
            logger.error(f"Failed to get schema info: {e}")
            raise SupabaseDatabaseServiceError(f"Failed to get schema info: {e}") from e

    async def get_schema_info(self) -> Dict[str, Any]:
        """Get comprehensive schema information."""
        try:
            schema_str = await self._get_schema_info_impl()
            return {"schema": schema_str}
        except Exception as e:
            logger.error(f"Failed to get schema info: {e}")
            raise SupabaseDatabaseServiceError(f"Failed to get schema info: {e}") from e

    async def query_table(
        self, table_name: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Query a specific table.

        Args:
            table_name: Name of the table to query
            limit: Maximum number of rows to return

        Returns:
            List of rows as dictionaries
        """
        try:
            response = self.client.table(table_name).select("*").limit(limit).execute()
            return response.data
        except Exception as e:
            logger.error(f"Failed to query table {table_name}: {e}")
            raise SupabaseDatabaseServiceError(f"Failed to query table: {e}") from e


# Singleton instance
_supabase_service: Optional[SupabaseDatabaseService] = None


def get_supabase_database_service() -> SupabaseDatabaseService:
    """
    Get or create Supabase Database Service instance (singleton pattern).

    Returns:
        SupabaseDatabaseService instance
    """
    global _supabase_service

    if _supabase_service is None:
        from app.config.environment import get_env, validate_mcp_config

        # We still use validate_mcp_config for now as it checks Supabase env vars
        validate_mcp_config()

        # Use SUPABASE_PROJECT_URL and SUPABASE_SERVICE_ROLE_KEY
        supabase_url = get_env("SUPABASE_PROJECT_URL")
        supabase_key = get_env("SUPABASE_SERVICE_ROLE_KEY")

        if not supabase_url or not supabase_key:
            raise ValueError(
                "SUPABASE_PROJECT_URL and SUPABASE_SERVICE_ROLE_KEY environment variables are required"
            )

        _supabase_service = SupabaseDatabaseService(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
        )

    return _supabase_service


def close_supabase_database_service():
    """Close the Supabase Database Service singleton instance."""
    global _supabase_service
    _supabase_service = None
