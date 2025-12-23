"""
Agent & Routing Intelligence Benchmarks.

This module benchmarks the "Brain" of the Legion system:
1. Routing Intelligence (Classification latency & accuracy)
2. Task Agent Planner (Planning latency & plan quality)
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.hermes.legion.agents.task_agent_planner import TaskAgentPlanner
from app.hermes.legion.intelligence.routing_intelligence import (
    RoutingAction,
    RoutingDecision,
)
from app.hermes.legion.intelligence.routing_service import RoutingIntelligence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent_benchmarks")


@dataclass
class RoutingTestCase:
    input_text: str
    expected_action: RoutingAction
    complexity: str  # "simple", "complex", "ambiguous"


@dataclass
class AgentPlanningTestCase:
    task_description: str
    complexity_estimate: str
    min_agents: int


class AgentBenchmarker:
    def __init__(self):
        self.router = RoutingIntelligence()
        self.planner = TaskAgentPlanner()

    async def run_routing_benchmarks(self) -> List[Dict[str, Any]]:
        """Benchmark Routing Intelligence."""
        print("\n" + "=" * 60)
        print(" 🧠 ROUTING INTELLIGENCE BENCHMARKS")
        print("=" * 60)

        test_cases = [
            # SIMPLE
            RoutingTestCase("Hello there", RoutingAction.SIMPLE_RESPONSE, "simple"),
            RoutingTestCase(
                "What time is it?", RoutingAction.SIMPLE_RESPONSE, "simple"
            ),
            # ORCHESTRATE
            RoutingTestCase(
                "Write a Python script to scrape a website",
                RoutingAction.ORCHESTRATE,
                "complex",
            ),
            RoutingTestCase(
                "Research the impact of quantum computing on crypto",
                RoutingAction.ORCHESTRATE,
                "complex",
            ),
            # GATHER_INFO / AMBIGUOUS (Might be inferred or gather)
            RoutingTestCase("Do it", RoutingAction.GATHER_INFO, "ambiguous"),
        ]

        results = []

        for case in test_cases:
            print(f"\n🔍 Testing: '{case.input_text}'")
            start = time.perf_counter()
            try:
                decision = await self.router.analyze(
                    case.input_text, conversation_history=[]
                )
                latency_ms = (time.perf_counter() - start) * 1000

                # Check accuracy
                # Note: Semantic equivalent actions might be acceptable, but strict check for now
                is_correct = decision.action == case.expected_action

                # If ambiguous, we might accept orchestrate as well if reasoning is good,
                # but for benchmark we stick to expected.

                print(
                    f"   -> Decision: {decision.action.value} (Conf: {decision.confidence:.2f})"
                )
                print(f"   -> Latency:  {latency_ms:.0f}ms")
                print(f"   -> Match:    {'✅' if is_correct else '❌'}")

                results.append(
                    {
                        "input": case.input_text,
                        "expected": case.expected_action.value,
                        "actual": decision.action.value,
                        "latency_ms": latency_ms,
                        "correct": is_correct,
                        "complexity": case.complexity,
                    }
                )

            except Exception as e:
                logger.error(f"Routing failed for '{case.input_text}': {e}")
                results.append(
                    {"input": case.input_text, "error": str(e), "correct": False}
                )

        return results

    async def run_planning_benchmarks(self) -> List[Dict[str, Any]]:
        """Benchmark Task Agent Planner."""
        print("\n" + "=" * 60)
        print(" 📋 TASK PLANNING BENCHMARKS")
        print("=" * 60)

        test_cases = [
            AgentPlanningTestCase("Summarize this short text about cats.", "simple", 1),
            AgentPlanningTestCase(
                "Build a full e-commerce backend with user auth, cart, and stripe integration.",
                "complex",
                2,  # Should be at least 2 (coder, architect, etc)
            ),
        ]

        results = []

        for case in test_cases:
            print(f"\n🏗️  Planning Task: '{case.task_description[:50]}...'")
            start = time.perf_counter()
            try:
                # Use async version to match typical usage
                analysis = await self.planner.analyze_task_and_plan_agents_async(
                    case.task_description, complexity_estimate=case.complexity_estimate
                )
                latency_ms = (time.perf_counter() - start) * 1000

                agent_plan = analysis.get("agent_plan", [])
                agent_count = len(agent_plan)

                print(f"   -> Agents Created: {agent_count}")
                print(f"   -> Latency:        {latency_ms:.0f}ms")

                # Basic quality check
                passes_agents = agent_count >= case.min_agents
                print(
                    f"   -> Quality Check:  {'✅' if passes_agents else '⚠️'} (Min {case.min_agents})"
                )

                for agent in agent_plan:
                    print(
                        f"      - {agent.get('agent_id')} ({agent.get('agent_type')})"
                    )

                results.append(
                    {
                        "task": case.task_description,
                        "latency_ms": latency_ms,
                        "agent_count": agent_count,
                        "passes_min_agents": passes_agents,
                    }
                )

            except Exception as e:
                logger.error(f"Planning failed: {e}")

        return results


async def main():
    benchmarker = AgentBenchmarker()

    # 1. Routing
    routing_results = await benchmarker.run_routing_benchmarks()

    # 2. Planning
    planning_results = await benchmarker.run_planning_benchmarks()

    # Summary
    print("\n" + "=" * 60)
    print(" 🏁 AGENT BENCHMARK SUMMARY")
    print("=" * 60)

    # Routing Stats
    total_routing = len(routing_results)
    correct_routing = sum(1 for r in routing_results if r.get("correct"))
    avg_routing_lat = (
        sum(r.get("latency_ms", 0) for r in routing_results) / total_routing
        if total_routing
        else 0
    )

    print(
        f"Routing Accuracy: {correct_routing}/{total_routing} ({correct_routing/total_routing*100:.0f}%)"
    )
    print(f"Avg Routing Latency: {avg_routing_lat:.0f}ms")

    # Planning Stats
    total_planning = len(planning_results)
    avg_planning_lat = (
        sum(r.get("latency_ms", 0) for r in planning_results) / total_planning
        if total_planning
        else 0
    )

    print(f"Avg Planning Latency: {avg_planning_lat:.0f}ms")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
