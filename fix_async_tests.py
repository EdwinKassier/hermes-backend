#!/usr/bin/env python3
"""
Script to add @pytest.mark.asyncio decorators to async test functions.
"""
import re
import sys
from pathlib import Path


def fix_async_tests(file_path):
    """Add @pytest.mark.asyncio to async test functions if missing."""
    with open(file_path, "r") as f:
        content = f.read()

    original_content = content
    lines = content.split("\n")
    new_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if this is an async test function
        if re.match(r"\s*async def test_", line):
            # Check if previous line already has the decorator
            if i > 0 and "@pytest.mark.asyncio" not in lines[i - 1]:
                # Get the indentation of the function
                indent = len(line) - len(line.lstrip())
                decorator = " " * indent + "@pytest.mark.asyncio"

                # Add decorator before the function
                new_lines.append(decorator)

        new_lines.append(line)
        i += 1

    new_content = "\n".join(new_lines)

    if new_content != original_content:
        with open(file_path, "w") as f:
            f.write(new_content)
        return True
    return False


def main():
    test_files = [
        "tests/unit/hermes/legion/intelligence/test_adaptive_synthesizer.py",
        "tests/unit/hermes/legion/intelligence/test_query_analyzer.py",
        "tests/unit/hermes/legion/intelligence/test_routing_intelligence.py",
        "tests/unit/hermes/legion/intelligence/test_tool_intelligence.py",
        "tests/unit/hermes/legion/intelligence/test_worker_planner.py",
        "tests/unit/test_supabase_database_service.py",
    ]

    fixed_count = 0
    for file_path in test_files:
        path = Path(file_path)
        if path.exists():
            if fix_async_tests(path):
                print(f"✓ Fixed {file_path}")
                fixed_count += 1
            else:
                print(f"- No changes needed for {file_path}")
        else:
            print(f"✗ File not found: {file_path}")

    print(f"\nFixed {fixed_count} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
