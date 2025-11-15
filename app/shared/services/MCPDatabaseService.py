"""
MCP Database Service - HTTP client for Official Supabase MCP Server
Provides secure, performant database query capabilities with AI-powered analysis.
"""

import asyncio
import hashlib
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

import httpx

from app.config.environment import get_env

logger = logging.getLogger(__name__)


class MCPDatabaseServiceError(Exception):
    """Base exception for MCP Database Service errors."""

    pass


class MCPConnectionError(MCPDatabaseServiceError):
    """Raised when MCP server connection fails."""

    pass


class MCPAuthenticationError(MCPDatabaseServiceError):
    """Raised when MCP server authentication fails."""

    pass


class MCPTimeoutError(MCPDatabaseServiceError):
    """Raised when MCP server request times out."""

    pass


class MCPValidationError(MCPDatabaseServiceError):
    """Raised when input validation fails."""

    pass


class MCPDatabaseService:
    """
    HTTP client for Official Supabase MCP Server.

    Provides:
    - Natural language to SQL conversion
    - Secure query execution
    - AI-powered result analysis
    - Simple caching for performance
    """

    def __init__(
        self,
        mcp_server_url: str,
        api_key: str,
        timeout: float = 30.0,
        enable_cache: bool = True,
        cache_ttl: int = 300,
    ):
        """
        Initialize MCP Database Service.

        Args:
            mcp_server_url: URL of the MCP server
            api_key: API key for authentication
            timeout: Request timeout in seconds
            enable_cache: Enable query result caching
            cache_ttl: Cache TTL in seconds
        """
        self.mcp_server_url = mcp_server_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl

        # Simple in-memory cache
        self._cache: Dict[str, tuple] = {}

        # Rate limiting
        self._request_times: List[float] = []
        self._max_requests_per_minute = 60

        # HTTP client configuration with security headers and performance optimization
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "hermes-backend-mcp-client/1.0",
                "X-Requested-With": "XMLHttpRequest",
            },
            # Security settings
            verify=True,  # Verify SSL certificates
            follow_redirects=False,  # Don't follow redirects for security
            # Performance settings
            limits=httpx.Limits(
                max_keepalive_connections=5, max_connections=10, keepalive_expiry=30.0
            ),
            # Connection pooling
            http2=True,  # Enable HTTP/2 for better performance
            # Timeout settings
            timeout=httpx.Timeout(
                connect=10.0,  # Connection timeout
                read=30.0,  # Read timeout
                write=10.0,  # Write timeout
                pool=5.0,  # Pool timeout
            ),
        )

        logger.info(f"MCP Database Service initialized for {mcp_server_url}")

    async def execute_query(
        self, query: str, include_analysis: bool = True, use_cache: bool = True
    ) -> str:
        """
        Execute natural language query with optional AI analysis.

        Args:
            query: Natural language query description
            include_analysis: Whether to include AI-powered analysis
            use_cache: Whether to use cached results

        Returns:
            Formatted query results with optional analysis
        """
        # Input validation and sanitization
        if not query or not isinstance(query, str):
            raise MCPValidationError("Query must be a non-empty string")

        if len(query.strip()) == 0:
            raise MCPValidationError("Query cannot be empty or whitespace only")

        if len(query) > 10000:  # Reasonable limit
            raise MCPValidationError("Query too long (max 10000 characters)")

        # Sanitize query to prevent injection attacks
        query = self._sanitize_query(query)

        # Rate limiting check
        self._check_rate_limit()

        # Check cache first
        if use_cache and self.enable_cache:
            cached_result = self._get_cached_result(query)
            if cached_result:
                logger.info("Returning cached query result")
                return cached_result

        try:
            # Execute query via MCP server
            result = await self._execute_mcp_query(query)

            # Cache result
            if use_cache and self.enable_cache:
                self._cache_result(query, result)

            # Add AI analysis if requested
            if include_analysis and result:
                analysis = await self._analyze_results(query, result)
                return f"{analysis}\n\n--- Raw Data ---\n{result}"

            return result

        except MCPValidationError:
            raise  # Re-raise validation errors
        except MCPConnectionError:
            raise  # Re-raise connection errors
        except MCPAuthenticationError:
            raise  # Re-raise authentication errors
        except MCPTimeoutError:
            raise  # Re-raise timeout errors
        except Exception as e:
            logger.error(f"MCP query execution failed: {e}")
            raise MCPDatabaseServiceError(f"Query execution failed: {e}") from e

    async def list_tables(self) -> List[str]:
        """List all available tables."""
        try:
            response = await self.client.get(f"{self.mcp_server_url}/tables")
            response.raise_for_status()
            data = response.json()
            return data.get("tables", [])
        except httpx.HTTPError as e:
            logger.error(f"Failed to list tables: {e}")
            raise MCPDatabaseServiceError(f"Failed to list tables: {e}") from e

    async def describe_table(self, table_name: str) -> Dict[str, Any]:
        """Get table schema information."""
        try:
            response = await self.client.get(
                f"{self.mcp_server_url}/tables/{table_name}"
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to describe table {table_name}: {e}")
            raise MCPDatabaseServiceError(f"Failed to describe table: {e}") from e

    async def get_schema_info(self) -> Dict[str, Any]:
        """Get comprehensive schema information."""
        try:
            response = await self.client.get(f"{self.mcp_server_url}/schema")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to get schema info: {e}")
            raise MCPDatabaseServiceError(f"Failed to get schema info: {e}") from e

    async def _execute_mcp_query(self, query: str) -> str:
        """Execute query via MCP server HTTP API."""
        try:
            response = await self.client.post(
                f"{self.mcp_server_url}/query", json={"query": query, "format": "json"}
            )
            response.raise_for_status()

            data = response.json()

            # Handle different response formats
            if "result" in data:
                result = data["result"]
            elif "data" in data:
                result = data["data"]
            else:
                result = data

            # Format result as string
            if isinstance(result, (dict, list)):
                return json.dumps(result, indent=2, default=str)
            else:
                return str(result)

        except httpx.TimeoutException as e:
            raise MCPTimeoutError(f"Request timed out: {e}") from e
        except httpx.ConnectError as e:
            raise MCPConnectionError(f"Connection failed: {e}") from e
        except httpx.HTTPError as e:
            if hasattr(e, "response") and e.response:
                status_code = e.response.status_code
                if status_code == 401:
                    raise MCPAuthenticationError(
                        "Authentication failed - check API key"
                    ) from e
                elif status_code == 403:
                    raise MCPAuthenticationError(
                        "Access forbidden - check API key permissions"
                    ) from e
                elif status_code == 404:
                    raise MCPConnectionError(
                        "MCP server not found - check server URL"
                    ) from e
                elif status_code == 500:
                    raise MCPDatabaseServiceError("MCP server internal error") from e
                elif status_code == 503:
                    raise MCPDatabaseServiceError(
                        "MCP server temporarily unavailable"
                    ) from e
                else:
                    raise MCPDatabaseServiceError(
                        f"HTTP error {status_code}: {e}"
                    ) from e
            else:
                raise MCPDatabaseServiceError(f"HTTP error: {e}") from e

    async def _analyze_results(self, query: str, results: str) -> str:
        """Use AI to analyze query results."""
        try:
            # Import GeminiService here to avoid circular imports
            from app.shared.services.GeminiService import GeminiService

            gemini_service = GeminiService()

            prompt = f"""
            Analyze these database query results and provide insights.

            Original Query: "{query}"

            Results:
            {results}

            Provide:
            1. Executive summary of what the data shows
            2. Key insights and patterns
            3. Notable trends or outliers
            4. Business implications
            5. Simple, non-technical explanation

            Keep it concise but comprehensive. Focus on actionable insights.
            """

            # Use GeminiService to generate analysis
            analysis = gemini_service.generate_gemini_response(
                prompt=prompt, persona="default"
            )

            return analysis

        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
            return "Analysis unavailable - AI service error"

    def _get_cached_result(self, query: str) -> Optional[str]:
        """Get cached query result."""
        query_hash = hashlib.sha256(query.encode()).hexdigest()

        if query_hash in self._cache:
            result, timestamp = self._cache[query_hash]
            if time.time() - timestamp < self.cache_ttl:
                return result
            else:
                del self._cache[query_hash]

        return None

    def _cache_result(self, query: str, result: str):
        """Cache query result."""
        query_hash = hashlib.sha256(query.encode()).hexdigest()

        # Simple LRU: keep last 100 queries
        if len(self._cache) >= 100:
            oldest_key = min(self._cache.items(), key=lambda x: x[1][1])[0]
            del self._cache[oldest_key]

        self._cache[query_hash] = (result, time.time())

    def _sanitize_query(self, query: str) -> str:
        """Sanitize query input to prevent injection attacks."""
        import re

        # Remove potentially dangerous characters
        dangerous_patterns = [
            r'[<>"\']',  # HTML/XML injection
            r"[;\\]",  # SQL injection
            r"[{}]",  # JSON injection
            r"[()]",  # Command injection
        ]

        sanitized = query
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, "", sanitized)

        # Normalize whitespace
        sanitized = " ".join(sanitized.split())

        # Check for suspicious patterns
        suspicious_patterns = [
            r"(?i)(drop|delete|insert|update|alter|create|truncate)",
            r"(?i)(union|select|from|where)",
            r"(?i)(script|javascript|vbscript)",
            r"(?i)(exec|execute|eval)",
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, sanitized):
                logger.warning(f"Suspicious query pattern detected: {pattern}")
                # Don't raise error, just log and continue
                # The MCP server should handle SQL generation safely

        return sanitized

    def _check_rate_limit(self):
        """Check if request is within rate limit."""
        current_time = time.time()

        # Remove requests older than 1 minute
        self._request_times = [t for t in self._request_times if current_time - t < 60]

        # Check if we're at the limit
        if len(self._request_times) >= self._max_requests_per_minute:
            raise MCPDatabaseServiceError(
                f"Rate limit exceeded: {self._max_requests_per_minute} requests per minute"
            )

        # Add current request time
        self._request_times.append(current_time)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    def __del__(self):
        """Cleanup on deletion."""
        # Note: Cannot use asyncio in __del__, so we just log a warning
        # The client should be properly closed via the close() method
        if hasattr(self, "client") and not self.client.is_closed:
            logger.warning(
                "MCPDatabaseService client not properly closed. Call close() explicitly."
            )


# Singleton instance with thread safety
_mcp_service: Optional[MCPDatabaseService] = None
_lock = threading.Lock()


def get_mcp_database_service() -> MCPDatabaseService:
    """
    Get or create MCP Database Service instance (thread-safe singleton pattern).

    Returns:
        MCPDatabaseService instance
    """
    global _mcp_service

    if _mcp_service is None:
        with _lock:
            # Double-checked locking pattern
            if _mcp_service is None:
                # Validate MCP configuration
                from app.config.environment import validate_mcp_config

                validate_mcp_config()

                mcp_server_url = get_env("SUPABASE_MCP_SERVER_URL")
                api_key = get_env("SUPABASE_MCP_API_KEY")

                if not mcp_server_url or not api_key:
                    raise ValueError(
                        "SUPABASE_MCP_SERVER_URL and SUPABASE_MCP_API_KEY environment variables are required"
                    )

                _mcp_service = MCPDatabaseService(
                    mcp_server_url=mcp_server_url,
                    api_key=api_key,
                    enable_cache=True,
                    cache_ttl=300,
                )

    return _mcp_service


def close_mcp_database_service():
    """Close the MCP Database Service singleton instance."""
    global _mcp_service

    if _mcp_service is not None:
        try:
            # Run the async close in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_mcp_service.close())
            loop.close()
        except Exception as e:
            logger.error(f"Error closing MCP Database Service: {e}")
        finally:
            _mcp_service = None
