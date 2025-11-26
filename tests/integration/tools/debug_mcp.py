#!/usr/bin/env python3
"""Debug MCP connection"""

import asyncio
import logging
import os
import sys

logging.basicConfig(level=logging.DEBUG)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dotenv import load_dotenv

load_dotenv()


async def test_mcp():
    """Test MCP connection"""
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    url = os.getenv("SUPABASE_MCP_SERVER_URL")
    api_key = os.getenv("SUPABASE_MCP_API_KEY")

    print(f"URL: {url}")
    print(f"API Key: {api_key[:20]}...")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print(f"Headers: {list(headers.keys())}")

    try:
        print("\n1. Creating SSE client with auth headers...")
        async with sse_client(url, headers=headers) as (read, write):
            print("2. SSE client created")

            print("3. Creating ClientSession...")
            async with ClientSession(read, write) as session:
                print("4. Session created")

                print("5. Initializing session...")
                await session.initialize()
                print("6. Session initialized!")

                print("\n7. Listing tools...")
                tools = await session.list_tools()
                print(f"Available tools: {[t.name for t in tools.tools]}")

                if tools.tools:
                    tool = tools.tools[0]
                    print(f"\n8. Calling tool: {tool.name}")
                    result = await session.call_tool(tool.name, arguments={})
                    print(f"Result: {result}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_mcp())
