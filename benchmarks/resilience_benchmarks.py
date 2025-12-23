"""
Resilience & Chaos Benchmarks.

This module tests system robustness by injecting faults:
1. Chaos Testing (Simulated LLM failures: Timeouts, 500s, Garbage)
2. Fuzz Testing (Edge case inputs)
3. Tool Resilience (Simulated Tool Failures)
"""

import asyncio
import logging
import random
import time
from unittest.mock import MagicMock, patch

from langchain.tools import BaseTool

from app.hermes.legion.graph_service import LegionGraphService
from app.hermes.models import ResponseMode, UserIdentity
from app.shared.services.AsyncLLMService import AsyncLLMService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("resilience_benchmarks")


class FaultyLLMService:
    """Simulates a flaky LLM provider."""

    def __init__(
        self, failure_rate: float = 0.0, latency_ms: int = 0, error_type: str = "random"
    ):
        self.failure_rate = failure_rate
        self.latency_ms = latency_ms
        self.error_type = error_type  # "timeout", "rate_limit", "garbage_json"

    async def generate_async(self, prompt: str, persona: str = "default", **kwargs):
        # Simulate Latency
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000)

        # Simulate Failure
        if random.random() < self.failure_rate:
            if self.error_type == "timeout":
                raise asyncio.TimeoutError("Simulated LLM Timeout")
            elif self.error_type == "rate_limit":
                raise Exception("429: Too Many Requests")
            elif self.error_type == "garbage":
                return "This is not valid JSON { broken ... "
            else:
                raise Exception("500: Internal Server Error")

        # Valid Response (Mock)
        # return a generic valid routing decision or text
        if "routing" in prompt.lower():
            # Return a valid routing JSON to allow progression unless we want to fail routing
            return '{"action": "simple_response", "reasoning": "Mock pass", "confidence": 1.0, "requires_agents": false}'

        return "Resilience test valid response."


class BrokenTool(BaseTool):
    name: str = "broken_tool"
    description: str = "A tool that always fails."

    def _run(self, query: str):
        raise Exception("Simulated Tool Crash (Network Error)")

    async def _arun(self, query: str):
        raise Exception("Simulated Tool Crash (Network Error)")


class ResilienceBenchmarker:
    def __init__(self):
        self.user = UserIdentity(
            user_id="chaos_user", ip_address="127.0.0.1", user_agent="Chaos/1.0"
        )

    async def run_chaos_test(
        self, scenario_name: str, failure_rate: float, error_type: str
    ):
        """Run a workflow with injected faults."""
        print(f"\n🌪️  CHAOS TEST: {scenario_name}")
        print(f"    Failure Rate: {failure_rate*100}% | Type: {error_type}")

        # Patch the get_async_llm_service to return our faulty one
        faulty_service = FaultyLLMService(
            failure_rate=failure_rate, latency_ms=100, error_type=error_type
        )

        with patch(
            "app.shared.utils.service_loader.get_async_llm_service",
            return_value=faulty_service,
        ):
            service = LegionGraphService()

            start = time.perf_counter()
            try:
                # We expect this might fail, but checking for CRASH vs GRACEFUL ERROR
                result = await service.process_request(
                    text="Resilience Test Query",
                    user_identity=self.user,
                    response_mode=ResponseMode.TEXT,
                    persona="legion",
                )
                duration = (time.perf_counter() - start) * 1000
                print(f"    Result: SUCCESS (Handled gracefully)")
                print(f"    Response: {str(result)[:50]}...")
                return True

            except Exception as e:
                duration = (time.perf_counter() - start) * 1000
                print(f"    Result: FAILED (Exception raised)")
                print(f"    Error: {e}")
                # We want to know if it CRASHED hard (bad) or raised a controlled error
                return False

    async def run_fuzz_test(self):
        """Run fuzz testing on inputs."""
        print("\n" + "=" * 60)
        print(" 🧪 FUZZ TESTING")
        print("=" * 60)

        service = LegionGraphService()

        fuzz_inputs = [
            ("Empty String", ""),
            ("Giant String", "A" * 10000),
            ("SQL Injection-ish", "DROP TABLE users; --"),
            ("JSON Injection", '{"action": "hack"}'),
        ]

        for name, inp in fuzz_inputs:
            print(f"\nTesting: {name}")
            try:
                # Mock LLM to avoid costs/noise
                mock_llm = MagicMock()
                mock_llm.generate_async.return_value = "Protected response"

                with patch(
                    "app.shared.utils.service_loader.get_async_llm_service",
                    return_value=mock_llm,
                ):
                    result = await service.process_request(
                        text=inp,
                        user_identity=self.user,
                        response_mode=ResponseMode.TEXT,
                        persona="legion",
                    )
                    print(f"  -> Survived: ✅")
            except Exception as e:
                print(f"  -> Crashed: ❌ ({e})")

    async def run_tool_resilience_test(self):
        """Test tool failure handling."""
        print("\n" + "=" * 60)
        print(" 🛠️  TOOL RESILIENCE TESTING")
        print("=" * 60)

        # We need to manually register a broken tool and try to execute it
        from app.hermes.legion.utils.tool_registry import get_tool_registry
        from app.shared.services.LLMService import LLMService, ToolExecutionError

        service = LLMService()
        registry = get_tool_registry()

        # 1. Register Broken Tool
        broken = BrokenTool()
        registry.register_tool(broken, "broken_tool")

        print(f"Registered 'broken_tool'. Simulating failure...")

        # 2. Execute it via LLMService (using internal method for direct test)
        # We simulate what the LLM would return: a tool call
        tool_cal = {"name": "broken_tool", "args": {"query": "fail check"}}

        try:
            # We fail 3 times to trigger degradation
            for i in range(4):
                print(f"  Attempt {i+1}: Executing broken tool...")
                results = service._execute_tools([tool_cal])
                print(f"    Result: {results[0]}")

                # Check status
                healthy = registry.is_tool_healthy("broken_tool")
                print(f"    Tool Healthy? {'✅' if healthy else '❌ DEGRADED'}")

        except ToolExecutionError:
            print(
                "  ❌ CRITCAL: ToolExecutionError was raised! Circuit breaker failed."
            )
        except Exception as e:
            print(f"  ❌ CRITCAL: Unexpected exception: {e}")

        # Cleanup
        registry.unregister_tool("broken_tool")


async def main():
    benchmarker = ResilienceBenchmarker()

    print("============================================================")
    print(" RESILIENCE BENCHMARKS")
    print("============================================================")

    # Tool Resilience (New Test)
    await benchmarker.run_tool_resilience_test()

    # Chaos Tests (Mocking Failures)
    # await benchmarker.run_chaos_test("Unstable Network (Timeouts)", 0.5, "timeout")

    # Fuzz Tests
    # await benchmarker.run_fuzz_test()

    print("\n✅ Resilience Benchmarks Complete")


if __name__ == "__main__":
    asyncio.run(main())
