#!/usr/bin/env python3
"""
Debug Firecrawl API to see what's happening.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()


def test_firecrawl_api():
    """Test Firecrawl API directly."""
    api_key = os.environ.get("FIRECRAWL_API_KEY")

    print(f"API Key present: {bool(api_key)}")
    print(f"API Key (first 10 chars): {api_key[:10] if api_key else 'None'}...")

    try:
        from firecrawl import FirecrawlApp

        print("✅ Firecrawl imported successfully")

        app = FirecrawlApp(api_key=api_key)
        print("✅ FirecrawlApp initialized")

        # Test search
        print("\n--- Testing search ---")
        query = "Python programming"
        print(f"Query: {query}")

        result = app.search(query, limit=3)

        print(f"\nResult type: {type(result)}")
        print(
            f"Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}"
        )
        print(f"\nFull result:")
        import json

        print(json.dumps(result, indent=2, default=str))

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_firecrawl_api()
