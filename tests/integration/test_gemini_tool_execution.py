"""
Integration tests for GeminiService tool execution with real LLM responses.

These tests verify that the GeminiService correctly:
1. Processes tool calls from the real LLM
2. Executes tools and returns results
3. Generates natural language responses incorporating tool results
4. Handles tool execution errors gracefully

Note: These tests require real API keys and will make actual calls to Gemini.
"""

import os
import time
from datetime import datetime

import pytest
from google.api_core import exceptions as google_exceptions

from app.shared.services.GeminiService import GeminiService, PersonaConfig


@pytest.mark.integration
@pytest.mark.requires_api_key
class TestGeminiToolExecution:
    """Test GeminiService tool execution with real LLM responses."""

    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Set up test environment with required API keys."""
        required_keys = ["GOOGLE_API_KEY"]
        missing_keys = [key for key in required_keys if not os.environ.get(key)]

        if missing_keys:
            pytest.skip(f"Missing required environment variables: {missing_keys}")

        # Initialize service with test persona that has access to time tool
        self.service = GeminiService()

        # Create a test persona with time tool access
        test_persona = PersonaConfig(
            name="test_tool_execution",
            base_prompt="You are a helpful assistant that can use tools when needed. Always use the time_tool when asked about the current time.",
            model_name="gemini-2.5-flash",
            temperature=0.1,  # Low temperature for consistent responses
            timeout=30,
            max_retries=2,
            allowed_tools=["time_info"],
            error_message_template="I encountered an error while processing your request.",
        )

        self.service.add_persona(test_persona)

    def _call_api_safely(self, func, *args, **kwargs):
        """Helper method to call API methods with error handling."""
        try:
            return func(*args, **kwargs)
        except google_exceptions.ResourceExhausted as e:
            pytest.skip(f"API quota exceeded: {e}")
        except google_exceptions.PermissionDenied as e:
            error_msg = str(e).lower()
            if "billing" in error_msg or "403" in error_msg:
                pytest.skip(f"Billing/permission issue: {e}")
            raise
        except Exception as e:
            error_msg = str(e).lower()
            if (
                "quota" in error_msg
                or "429" in error_msg
                or "resourceexhausted" in error_msg
            ):
                pytest.skip(f"API quota exceeded: {e}")
            elif "billing" in error_msg or "403" in error_msg:
                pytest.skip(f"Billing not enabled: {e}")
            raise

    def test_time_tool_execution_with_real_llm(self):
        """Test that the service correctly executes the time tool with real LLM."""
        # Test prompt that should trigger time tool usage
        test_prompt = "What is the current time?"

        # Get response from real LLM
        response = self._call_api_safely(
            self.service.generate_gemini_response,
            prompt=test_prompt,
            persona="test_tool_execution",
        )

        # Verify response is not empty
        assert response is not None
        assert len(response.strip()) > 0

        # Verify response contains time-related information
        # The response should include actual time information from the tool
        assert any(
            keyword in response.lower()
            for keyword in [
                "time",
                "current",
                "now",
                "clock",
                "hour",
                "minute",
                "second",
                "pm",
                "am",
                "utc",
            ]
        ), f"Response should contain time information: {response}"

        # Verify response is natural language (not raw tool output)
        assert not response.startswith("time_tool:")
        assert ":" in response  # Should be formatted as natural language

        print(f"✅ Time tool execution successful. Response: {response}")

    def test_time_tool_with_specific_request(self):
        """Test time tool with more specific requests."""
        test_prompts = [
            "Can you tell me what time it is right now?",
            "I need to know the current time",
            "What's the time?",
            "Please get the current time for me",
        ]

        for prompt in test_prompts:
            response = self._call_api_safely(
                self.service.generate_gemini_response,
                prompt=prompt,
                persona="test_tool_execution",
            )

            # Verify response is meaningful
            assert response is not None
            assert len(response.strip()) > 5  # Should be more than just "OK"

            # Verify it contains time information
            assert any(
                keyword in response.lower()
                for keyword in ["time", "current", "now", "clock", "pm", "am", "utc"]
            ), f"Response to '{prompt}' should contain time info: {response}"

            print(f"✅ Prompt '{prompt}' -> Response: {response}")

            # Small delay to avoid rate limiting
            time.sleep(1)

    def test_tool_execution_error_handling(self):
        """Test that tool execution errors are handled gracefully."""
        # Create a persona with no tools to test error handling
        no_tools_persona = PersonaConfig(
            name="no_tools_persona",
            base_prompt="You are a helpful assistant.",
            model_name="gemini-2.5-flash",
            temperature=0.1,
            timeout=30,
            max_retries=2,
            allowed_tools=[],  # No tools available
            error_message_template="I'm sorry, I can't help with that request.",
        )

        self.service.add_persona(no_tools_persona)

        # Test with a prompt that might request tools
        response = self._call_api_safely(
            self.service.generate_gemini_response,
            prompt="What time is it?",
            persona="no_tools_persona",
        )

        # Should still get a response (even if it can't use tools)
        assert response is not None
        assert len(response.strip()) > 0

        print(f"✅ Error handling test successful. Response: {response}")

    def test_tool_execution_with_context(self):
        """Test tool execution with conversational context."""
        # First, establish some context
        context_prompt = "Hi, I'm working on a project and need to track time."
        context_response = self._call_api_safely(
            self.service.generate_gemini_response,
            prompt=context_prompt,
            persona="test_tool_execution",
        )

        assert context_response is not None
        print(f"✅ Context established: {context_response}")

        # Then ask for time with context
        time_prompt = "What's the current time so I can log it?"
        time_response = self._call_api_safely(
            self.service.generate_gemini_response,
            prompt=time_prompt,
            persona="test_tool_execution",
        )

        # Should get time information
        assert time_response is not None
        assert any(
            keyword in time_response.lower()
            for keyword in ["time", "current", "now", "pm", "am", "utc"]
        ), f"Should get time info: {time_response}"

        print(f"✅ Contextual time request successful: {time_response}")

    def test_multiple_tool_calls_in_conversation(self):
        """Test multiple tool calls in a single conversation."""
        # Ask for time multiple times
        prompts = [
            "What time is it?",
            "Can you check the time again?",
            "I need the current time one more time",
        ]

        responses = []
        for prompt in prompts:
            response = self._call_api_safely(
                self.service.generate_gemini_response,
                prompt=prompt,
                persona="test_tool_execution",
            )

            assert response is not None
            assert len(response.strip()) > 0

            # Each response should contain time information
            assert any(
                keyword in response.lower()
                for keyword in ["time", "current", "now", "pm", "am", "utc"]
            ), f"Response should contain time info: {response}"

            responses.append(response)
            print(f"✅ Multiple tool call {len(responses)}: {response}")

            # Small delay between requests
            time.sleep(1)

        # Verify we got different responses (not cached)
        assert (
            len(set(responses)) > 1
        ), "Should get varied responses for multiple time requests"

    def test_tool_execution_performance(self):
        """Test tool execution performance and timing."""
        start_time = time.time()

        response = self._call_api_safely(
            self.service.generate_gemini_response,
            prompt="What is the current time?",
            persona="test_tool_execution",
        )

        end_time = time.time()
        execution_time = end_time - start_time

        # Verify response quality
        assert response is not None
        assert len(response.strip()) > 0
        assert any(
            keyword in response.lower()
            for keyword in ["time", "current", "now", "pm", "am", "utc"]
        )

        # Verify reasonable performance (should complete within 30 seconds)
        assert (
            execution_time < 30
        ), f"Tool execution took too long: {execution_time:.2f}s"

        print(f"✅ Tool execution performance: {execution_time:.2f}s")
        print(f"✅ Response: {response}")

    def test_tool_execution_with_different_personas(self):
        """Test tool execution works across different persona configurations."""
        # Create another persona with different settings
        alternative_persona = PersonaConfig(
            name="alternative_tool_persona",
            base_prompt="You are a precise timekeeper assistant. Always provide accurate time information.",
            model_name="gemini-2.5-flash",
            temperature=0.0,  # Very low temperature for consistency
            timeout=30,
            max_retries=2,
            allowed_tools=["time_info"],
            error_message_template="I'm unable to provide time information at the moment.",
        )

        self.service.add_persona(alternative_persona)

        # Test with both personas
        personas_to_test = ["test_tool_execution", "alternative_tool_persona"]

        for persona in personas_to_test:
            response = self._call_api_safely(
                self.service.generate_gemini_response,
                prompt="What time is it?",
                persona=persona,
            )

            assert response is not None
            assert len(response.strip()) > 0
            assert any(
                keyword in response.lower()
                for keyword in ["time", "current", "now", "pm", "am", "utc"]
            ), f"Persona '{persona}' should provide time info: {response}"

            print(f"✅ Persona '{persona}' tool execution successful: {response}")
            time.sleep(1)

    def test_tool_execution_error_recovery(self):
        """Test that the service recovers gracefully from tool execution issues."""
        # This test verifies that even if there are issues, the service continues to work

        # First, get a successful response
        response1 = self._call_api_safely(
            self.service.generate_gemini_response,
            prompt="What time is it?",
            persona="test_tool_execution",
        )

        assert response1 is not None
        print(f"✅ First response successful: {response1}")

        # Then try another request to ensure service is still working
        response2 = self._call_api_safely(
            self.service.generate_gemini_response,
            prompt="Can you tell me the current time again?",
            persona="test_tool_execution",
        )

        assert response2 is not None
        assert len(response2.strip()) > 0

        print(f"✅ Second response successful: {response2}")

        # Both responses should contain time information
        for i, response in enumerate([response1, response2], 1):
            assert any(
                keyword in response.lower()
                for keyword in ["time", "current", "now", "pm", "am", "utc"]
            ), f"Response {i} should contain time info: {response}"


@pytest.mark.integration
@pytest.mark.requires_api_key
class TestGeminiToolExecutionEdgeCases:
    """Test edge cases for GeminiService tool execution."""

    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Set up test environment."""
        if not os.environ.get("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not available")

        self.service = GeminiService()

    def _call_api_safely(self, func, *args, **kwargs):
        """Helper method to call API methods with error handling."""
        try:
            return func(*args, **kwargs)
        except google_exceptions.ResourceExhausted as e:
            pytest.skip(f"API quota exceeded: {e}")
        except google_exceptions.PermissionDenied as e:
            error_msg = str(e).lower()
            if "billing" in error_msg or "403" in error_msg:
                pytest.skip(f"Billing/permission issue: {e}")
            raise
        except Exception as e:
            error_msg = str(e).lower()
            if (
                "quota" in error_msg
                or "429" in error_msg
                or "resourceexhausted" in error_msg
            ):
                pytest.skip(f"API quota exceeded: {e}")
            elif "billing" in error_msg or "403" in error_msg:
                pytest.skip(f"Billing not enabled: {e}")
            raise

    def test_tool_execution_with_empty_prompt(self):
        """Test tool execution with empty or minimal prompts."""
        test_persona = PersonaConfig(
            name="edge_case_persona",
            base_prompt="You are a helpful assistant.",
            allowed_tools=["time_info"],
        )

        self.service.add_persona(test_persona)

        # Test with minimal prompt
        response = self._call_api_safely(
            self.service.generate_gemini_response,
            prompt="time",
            persona="edge_case_persona",
        )

        assert response is not None
        print(f"✅ Minimal prompt test: {response}")

    def test_tool_execution_with_very_long_prompt(self):
        """Test tool execution with very long prompts."""
        test_persona = PersonaConfig(
            name="long_prompt_persona",
            base_prompt="You are a helpful assistant that can use tools.",
            allowed_tools=["time_info"],
        )

        self.service.add_persona(test_persona)

        # Create a very long prompt
        long_prompt = (
            "I need to know the current time because "
            + "I have many things to do " * 50
            + "and I need to schedule them properly. What time is it?"
        )

        response = self._call_api_safely(
            self.service.generate_gemini_response,
            prompt=long_prompt,
            persona="long_prompt_persona",
        )

        assert response is not None
        assert len(response.strip()) > 0

        print(f"✅ Long prompt test successful: {response[:100]}...")

    def test_tool_execution_with_special_characters(self):
        """Test tool execution with special characters in prompts."""
        test_persona = PersonaConfig(
            name="special_chars_persona",
            base_prompt="You are a helpful assistant.",
            allowed_tools=["time_info"],
        )

        self.service.add_persona(test_persona)

        special_prompts = [
            "What's the time? (urgent!)",
            "Time? Please! 🕐",
            "Current time: needed ASAP",
            "Time check - priority HIGH",
        ]

        for prompt in special_prompts:
            response = self._call_api_safely(
                self.service.generate_gemini_response,
                prompt=prompt,
                persona="special_chars_persona",
            )

            assert response is not None
            print(f"✅ Special chars test '{prompt}': {response}")
            time.sleep(1)


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])
