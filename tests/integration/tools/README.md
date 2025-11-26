# Integration Test Configuration

This directory contains **real integration tests** that actually execute the tools with real APIs and services.

## Test Files

### `test_real_tool_execution.py`
Real integration tests that call actual tool implementations:
- **Time Tool**: Tests real timezone conversions and formatting
- **Web Search Tool**: Tests real Firecrawl API calls (requires API key)
- **Database Tool**: Tests real MCP database queries (requires MCP server)

## Running Tests

### Run All Integration Tests
```bash
pytest tests/integration/tools/ -v -s
```

### Run Specific Test Class
```bash
# Time tool tests (no API key needed)
pytest tests/integration/tools/test_real_tool_execution.py::TestTimeToolRealExecution -v -s

# Web search tests (requires FIRECRAWL_API_KEY)
pytest tests/integration/tools/test_real_tool_execution.py::TestWebSearchToolRealExecution -v -s

# Database tests (requires MCP server)
pytest tests/integration/tools/test_real_tool_execution.py::TestDatabaseQueryToolRealExecution -v -s
```

### Run Specific Test
```bash
pytest tests/integration/tools/test_real_tool_execution.py::TestTimeToolRealExecution::test_get_utc_time_real -v -s
```

## Requirements

### Time Tool Tests
- ✅ No external dependencies
- ✅ Always runnable

### Web Search Tool Tests
- ⚠️ Requires `FIRECRAWL_API_KEY` environment variable
- ⚠️ Requires `firecrawl-py` package: `pip install firecrawl-py`
- Tests will be skipped if API key not set

### Database Tool Tests
- ⚠️ Requires `SUPABASE_MCP_SERVER_URL` environment variable
- ⚠️ Requires `SUPABASE_MCP_API_KEY` environment variable
- ⚠️ Requires MCP server to be running
- Tests will be skipped if not configured

## Environment Setup

Create a `.env` file or export variables:

```bash
# For web search tests
export FIRECRAWL_API_KEY="your_firecrawl_api_key"

# For database tests
export SUPABASE_MCP_SERVER_URL="http://localhost:3000"
export SUPABASE_MCP_API_KEY="your_mcp_api_key"
```

## Test Output

Tests use `-s` flag to show print statements, so you'll see:
- Actual tool outputs
- API responses
- Formatted results
- Success/failure indicators

## What These Tests Verify

### Time Tool
- ✅ Real timezone conversions work
- ✅ Output formatting is correct
- ✅ Context information is provided
- ✅ Multiple timezones can be queried
- ✅ Error handling for invalid timezones

### Web Search Tool
- ✅ Real Firecrawl API integration works
- ✅ Search results are formatted correctly
- ✅ Multiple searches can be performed
- ✅ API key validation works
- ✅ Error handling for missing dependencies

### Database Tool
- ✅ Real MCP server integration works
- ✅ Queries are executed correctly
- ✅ Input validation works
- ✅ Helper methods work
- ✅ Error handling for invalid inputs

### Tool System
- ✅ All tools load from toolhub
- ✅ All tools are callable
- ✅ Tool count is correct
- ✅ Tool names are unique

## CI/CD Considerations

For CI/CD pipelines:

```yaml
# GitHub Actions example
- name: Run Integration Tests
  env:
    FIRECRAWL_API_KEY: ${{ secrets.FIRECRAWL_API_KEY }}
    SUPABASE_MCP_SERVER_URL: ${{ secrets.MCP_SERVER_URL }}
    SUPABASE_MCP_API_KEY: ${{ secrets.MCP_API_KEY }}
  run: |
    pytest tests/integration/tools/ -v -s
```

Tests will automatically skip if credentials not available.
