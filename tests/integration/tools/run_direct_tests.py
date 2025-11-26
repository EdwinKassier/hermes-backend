#!/usr/bin/env python3
"""
Direct tool execution test - runs tools directly without pytest.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def test_time_tool():
    """Test time tool execution."""
    print("\n" + "=" * 60)
    print("Testing Time Tool")
    print("=" * 60)

    try:
        from app.shared.utils.tools.time_tool import TimeInfoTool

        tool = TimeInfoTool()
        result = tool._run(timezone="UTC")

        print(result)
        print("\n✅ Time tool test PASSED")
        return True
    except Exception as e:
        print(f"\n❌ Time tool test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_web_search_tool():
    """Test web search tool execution."""
    print("\n" + "=" * 60)
    print("Testing Web Search Tool")
    print("=" * 60)

    try:
        from app.shared.utils.tools.web_search_tool import WebSearchTool

        tool = WebSearchTool()

        if not tool.api_key:
            print("⚠️  FIRECRAWL_API_KEY not set - skipping real search")
            print("Testing error handling instead...")
            result = tool._run("test query")
            print(result)
            assert "unavailable" in result
            print("\n✅ Web search tool error handling test PASSED")
            return True

        print("API key found - running real search...")
        result = tool._run("Python programming", max_results=3)

        print(result[:500] + "..." if len(result) > 500 else result)
        print("\n✅ Web search tool test PASSED")
        return True
    except Exception as e:
        print(f"\n❌ Web search tool test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_database_tool():
    """Test database tool execution."""
    print("\n" + "=" * 60)
    print("Testing Database Tool")
    print("=" * 60)

    try:
        from app.shared.utils.tools.database_query_tool import DatabaseQueryTool

        tool = DatabaseQueryTool()

        # Test 1: Input validation
        print("\n--- Test 1: Input Validation ---")
        result = tool._run("")
        assert "Error" in result
        print(f"✓ Empty query validation: {result}")

        result = tool._run("ab")
        assert "Error" in result
        print(f"✓ Short query validation: {result}")

        result = tool._run("a" * 1001)
        assert "Error" in result
        print(f"✓ Long query validation: {result}")

        # Test 2: Real database query - List tables
        print("\n--- Test 2: Real Query - List Tables ---")
        result = tool._run(
            "List all tables in the database", include_analysis=False, use_cache=False
        )
        print(f"Query: 'List all tables in the database'")
        print(f"Result:\n{result}\n")

        # Verify we got a real response
        if result.startswith("Error:"):
            print(f"⚠️  Query returned error: {result}")
            return False

        # Test 3: Real database query - Count query
        print("\n--- Test 3: Real Query - Count Records ---")
        result = tool._run(
            "How many tables are in the database?",
            include_analysis=False,
            use_cache=False,
        )
        print(f"Query: 'How many tables are in the database?'")
        print(f"Result:\n{result}\n")

        # Test 4: Helper method - List tables
        print("\n--- Test 4: Helper Method - List Tables ---")
        result = tool._list_tables()
        print(f"List tables result:\n{result}\n")

        print("✅ Database tool test PASSED")
        return True
    except Exception as e:
        print(f"\n❌ Database tool test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_tool_loading():
    """Test that all tools load correctly."""
    print("\n" + "=" * 60)
    print("Testing Tool Loading")
    print("=" * 60)

    try:
        from app.shared.utils.toolhub import get_all_tools

        tools = get_all_tools()

        print(f"Loaded {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:60]}...")

        assert len(tools) == 3, f"Expected 3 tools, got {len(tools)}"

        tool_names = {tool.name for tool in tools}
        expected = {"database_query", "time_info", "web_search"}
        assert tool_names == expected, f"Expected {expected}, got {tool_names}"

        print("\n✅ Tool loading test PASSED")
        return True
    except Exception as e:
        print(f"\n❌ Tool loading test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("REAL TOOL INTEGRATION TESTS")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Time Tool", test_time_tool()))
    results.append(("Web Search Tool", test_web_search_tool()))
    results.append(("Database Tool", test_database_tool()))
    results.append(("Tool Loading", test_tool_loading()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
