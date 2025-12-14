#!/usr/bin/env python3
"""
Test script to verify code generation improvements.
"""

import asyncio
import os
import sys

from app.shared.services.GeminiService import GeminiService

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))


async def test_code_generation():
    """Test code generation with Legion personas."""
    try:
        # Initialize service
        service = GeminiService()

        # Test with pragmatic_developer persona
        prompt = """Generate a simple Python function that calculates the factorial of a number.
Include proper error handling and documentation."""

        print("Testing code generation with pragmatic_developer persona...")
        response = service.generate_gemini_response(
            prompt=prompt, persona="pragmatic_developer", user_id="test_user"
        )

        print("Response received:")
        print("=" * 50)
        print(response)
        print("=" * 50)

        # Check for basic code formatting
        if "```" in response:
            print("✅ Code blocks detected")
        else:
            print("❌ No code blocks found")

        # Check for closing brackets (basic check)
        if response.count("(") == response.count(")") and response.count(
            "["
        ) == response.count("]"):
            print("✅ Bracket matching looks good")
        else:
            print("⚠️  Bracket mismatch detected")

        # Check for basic Python function structure
        if "def factorial" in response.lower():
            print("✅ Function definition detected")
        else:
            print("⚠️  No function definition found")

        print("\nTest completed successfully!")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_code_generation())
