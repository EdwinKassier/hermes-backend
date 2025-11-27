#!/usr/bin/env python3
"""
Test complex database queries with the enhanced database tool.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def test_complex_queries():
    """Test complex database queries."""
    print("\n" + "=" * 60)
    print("Testing Complex Database Queries")
    print("=" * 60)

    from app.shared.utils.tools.database_query_tool import DatabaseQueryTool

    tool = DatabaseQueryTool()

    # Test 1: Read-only enforcement
    print("\n--- Test 1: Read-Only Enforcement ---")
    write_queries = [
        "INSERT INTO users VALUES (1, 'test')",
        "UPDATE users SET name='test'",
        "DELETE FROM users WHERE id=1",
        "DROP TABLE users",
        "CREATE TABLE test (id int)",
    ]

    for query in write_queries:
        result = tool._run(query)
        print(f"Query: {query[:50]}...")
        print(f"Result: {result[:100]}...")
        assert "not allowed" in result.lower() or "forbidden" in result.lower()
        print("✓ Correctly blocked\n")

    # Test 2: List tables
    print("\n--- Test 2: List Tables ---")
    result = tool._run("List all tables in the database")
    print(result[:500] + "..." if len(result) > 500 else result)
    assert (
        "tables" in result.lower() and "hermes_vectors" in result.lower()
    ), "Should list actual tables"
    print("✓ PASSED - Listed tables successfully\n")

    # Test 3: Count records
    print("\n--- Test 3: Count Records in Table ---")
    result = tool._run("How many records are in the hermes_vectors table?")
    print(result)
    assert "has" in result and "records" in result, "Should return record count"
    assert "could not determine" not in result.lower(), f"Query failed: {result}"
    print("✓ PASSED - Got record count\n")

    # Test 4: Describe table
    print("\n--- Test 4: Describe Table Schema ---")
    result = tool._run("Describe the structure of the project table")
    print(result)
    assert (
        "columns" in result.lower() or "table:" in result.lower()
    ), f"Should describe table schema, got: {result}"
    assert "could not determine" not in result.lower(), f"Query failed: {result}"
    print("✓ PASSED - Described table schema\n")

    # Test 5: Complex SELECT query
    print("\n--- Test 5: Complex SELECT Query ---")
    result = tool._run("Show me the first 5 projects from the project table")
    print(result[:800] + "..." if len(result) > 800 else result)
    assert (
        "record" in result.lower() or "project" in result.lower()
    ), f"Should return project data, got: {result}"
    assert "could not determine" not in result.lower(), f"Query failed: {result}"
    print("✓ PASSED - Retrieved project data\n")

    # Test 6: Filtered query
    print("\n--- Test 6: Filtered Query ---")
    result = tool._run(
        "Find all records in the test_run table where status equals 'completed'"
    )
    print(result[:800] + "..." if len(result) > 800 else result)
    # This query might legitimately return no results, so just check it executed
    assert "could not determine" not in result.lower(), f"Query failed: {result}"
    assert (
        "error" not in result.lower() or "no results" in result.lower()
    ), f"Query had error: {result}"
    print("✓ PASSED - Executed filtered query\n")

    # Test 7: Count query
    print("\n--- Test 7: Count Query ---")
    result = tool._run("Count how many workflows are in the workflow_entity table")
    print(result)
    assert "has" in result and (
        "record" in result or "workflow" in result
    ), f"Should return count, got: {result}"
    assert "could not determine" not in result.lower(), f"Query failed: {result}"
    print("✓ PASSED - Got workflow count\n")

    print("\n" + "=" * 60)
    print("✅ All complex query tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_complex_queries()
