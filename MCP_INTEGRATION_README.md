# MCP Server Integration

This document describes the integration of the Official Supabase MCP Server with the Hermes backend.

## Overview

The MCP (Model Context Protocol) server integration provides:
- Natural language to SQL query conversion
- Secure database querying with AI-powered analysis
- High-performance caching and connection pooling
- Full compliance with security best practices

## Components

### 1. MCP Database Service (`app/shared/services/MCPDatabaseService.py`)
- HTTP client for communicating with the MCP server
- Query caching and performance optimization
- AI-powered result analysis using GeminiService
- Comprehensive error handling

### 2. LangChain Tool (`app/shared/utils/tools/database_query_tool.py`)
- LangChain tool wrapper for MCP server
- Natural language query interface
- Automatic tool discovery via toolhub

### 3. Docker Integration
- MCP server service in `docker-compose.yml`
- Health checks and dependency management
- Environment variable configuration

## Environment Variables

Add these to your `.env` file:

```bash
# MCP Server Configuration
SUPABASE_MCP_SERVER_URL=http://localhost:3001
SUPABASE_MCP_API_KEY=your-mcp-api-key
```

## Usage

### Basic Usage

```python
from app.shared.utils.tools.database_query_tool import DatabaseQueryTool

# Create tool instance
tool = DatabaseQueryTool()

# Execute natural language query
result = tool._run(
    query="Show me all users created in the last week",
    include_analysis=True,
    use_cache=True
)
print(result)
```

### Direct Service Usage

```python
from app.shared.services.MCPDatabaseService import get_mcp_database_service
import asyncio

async def query_database():
    service = get_mcp_database_service()

    # Execute query
    result = await service.execute_query(
        "List all tables in the database",
        include_analysis=True
    )
    print(result)

    # List tables
    tables = await service.list_tables()
    print(f"Available tables: {tables}")

# Run async function
asyncio.run(query_database())
```

## Docker Setup

### Development

```bash
# Start all services including MCP server
docker-compose up --build

# The MCP server will be available at http://localhost:3001
# The Hermes backend will be available at http://localhost:8080
```

### Production

The MCP server runs as a separate container in the Docker Compose setup. Ensure your environment variables are properly configured.

## Features

### 1. Natural Language Queries
- Convert natural language to SQL
- Support for complex queries and aggregations
- Automatic table and column discovery

### 2. AI-Powered Analysis
- Automatic result analysis using Gemini
- Business insights and recommendations
- Non-technical explanations

### 3. Performance Optimization
- Query result caching (5-minute TTL)
- Connection pooling via HTTP client
- LRU cache eviction

### 4. Security
- Read-only mode by default
- API key authentication
- Input validation and sanitization

## Testing

### Unit Tests
```bash
# Run unit tests (requires pytest-asyncio)
python -m pytest tests/unit/test_mcp_database_service.py -v
```

### Integration Tests
```bash
# Run integration tests (requires MCP server running)
python -m pytest tests/integration/test_mcp_database_integration.py -v
```

## Error Handling

The service provides comprehensive error handling:

- `MCPDatabaseServiceError` - General service errors
- `MCPConnectionError` - Connection and authentication errors
- `QueryTimeoutError` - Query execution timeouts

## Monitoring

The service includes built-in metrics:
- Query execution times
- Cache hit rates
- Error rates
- Connection status

## Troubleshooting

### Common Issues

1. **MCP Server Not Available**
   - Check if MCP server is running: `curl http://localhost:3001/health`
   - Verify environment variables are set
   - Check Docker Compose logs: `docker-compose logs supabase-mcp`

2. **Authentication Errors**
   - Verify `SUPABASE_MCP_API_KEY` is correct
   - Check MCP server logs for authentication issues

3. **Query Failures**
   - Check MCP server logs for SQL errors
   - Verify database connection in MCP server
   - Test with simple queries first

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger('app.shared.services.MCPDatabaseService').setLevel(logging.DEBUG)
```

## Future Enhancements

- [ ] OAuth authentication support
- [ ] Real-time query subscriptions
- [ ] Advanced query optimization
- [ ] Custom query templates
- [ ] Query performance analytics

## Security Considerations

- Always use read-only mode in production
- Rotate API keys regularly
- Monitor query logs for suspicious activity
- Use HTTPS for MCP server communication
- Implement rate limiting for query endpoints
