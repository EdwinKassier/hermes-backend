#!/usr/bin/env python3
"""
Bottleneck Analyzer for Hermes & Legion.

This module provides deep analysis of system performance to identify
optimization opportunities with specific, actionable recommendations.

Features:
1. Component latency breakdown
2. Hot path identification
3. Caching opportunity detection
4. Async improvement analysis
5. Memory pressure analysis
6. Comparative benchmarking

Usage:
    python -m benchmarks.bottleneck_analyzer
    python -m benchmarks.bottleneck_analyzer --report-only
"""

import asyncio
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class ComponentMetrics:
    """Metrics for a system component."""

    name: str
    call_count: int = 0
    total_time_ms: float = 0
    min_time_ms: float = float("inf")
    max_time_ms: float = 0
    times: List[float] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0

    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / self.call_count if self.call_count > 0 else 0

    @property
    def std_dev_ms(self) -> float:
        return statistics.stdev(self.times) if len(self.times) > 1 else 0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0

    @property
    def error_rate(self) -> float:
        total = self.call_count + self.errors
        return (self.errors / total * 100) if total > 0 else 0

    def record(
        self, duration_ms: float, cache_hit: Optional[bool] = None, error: bool = False
    ):
        """Record a measurement."""
        if error:
            self.errors += 1
            return

        self.call_count += 1
        self.total_time_ms += duration_ms
        self.min_time_ms = min(self.min_time_ms, duration_ms)
        self.max_time_ms = max(self.max_time_ms, duration_ms)
        self.times.append(duration_ms)

        if cache_hit is True:
            self.cache_hits += 1
        elif cache_hit is False:
            self.cache_misses += 1


@dataclass
class OptimizationOpportunity:
    """An identified optimization opportunity."""

    category: str  # 'caching', 'async', 'latency', 'memory', 'error'
    severity: str  # 'critical', 'high', 'medium', 'low'
    component: str
    issue: str
    recommendation: str
    estimated_impact: str
    implementation_effort: str  # 'low', 'medium', 'high'
    code_location: Optional[str] = None

    @property
    def priority_score(self) -> int:
        """Calculate priority score (higher = more important)."""
        severity_scores = {"critical": 40, "high": 30, "medium": 20, "low": 10}
        effort_scores = {"low": 10, "medium": 5, "high": 1}
        return severity_scores.get(self.severity, 0) + effort_scores.get(
            self.implementation_effort, 0
        )


class BottleneckAnalyzer:
    """
    Comprehensive bottleneck analyzer for identifying optimization opportunities.
    """

    def __init__(self):
        self.components: Dict[str, ComponentMetrics] = {}
        self.opportunities: List[OptimizationOpportunity] = []
        self.analysis_start: float = 0
        self.analysis_end: float = 0

    def get_or_create_component(self, name: str) -> ComponentMetrics:
        """Get or create a component metrics tracker."""
        if name not in self.components:
            self.components[name] = ComponentMetrics(name=name)
        return self.components[name]

    def analyze_service_loading(self) -> List[OptimizationOpportunity]:
        """Analyze service loading patterns."""
        opportunities = []

        try:
            from app.shared.utils.service_loader import get_llm_service

            # Test cold start
            get_llm_service.cache_clear()

            start = time.perf_counter()
            _ = get_llm_service()
            cold_time = (time.perf_counter() - start) * 1000

            component = self.get_or_create_component("service_loader")
            component.record(cold_time, cache_hit=False)

            # Test hot start
            start = time.perf_counter()
            _ = get_llm_service()
            hot_time = (time.perf_counter() - start) * 1000

            component.record(hot_time, cache_hit=True)

            # Analyze
            if cold_time > 1000:
                opportunities.append(
                    OptimizationOpportunity(
                        category="latency",
                        severity="medium",
                        component="LLMService",
                        issue=f"Cold start takes {cold_time:.0f}ms",
                        recommendation="Consider lazy initialization of heavy dependencies (embeddings, vector stores)",
                        estimated_impact="Reduce cold start by 30-50%",
                        implementation_effort="medium",
                        code_location="app/shared/services/LLMService.py:__init__",
                    )
                )

            speedup = cold_time / hot_time if hot_time > 0 else 0
            print(
                f"  Service loading: cold={cold_time:.0f}ms, hot={hot_time:.4f}ms, speedup={speedup:.0f}x"
            )

        except Exception as e:
            print(f"  ⚠️ Service loading analysis failed: {e}")

        return opportunities

    def analyze_tool_registry(self) -> List[OptimizationOpportunity]:
        """Analyze tool registry performance."""
        opportunities = []

        try:
            from app.shared.utils.toolhub import (
                clear_tools_cache,
                get_all_tools,
                get_tool_by_name,
            )

            component = self.get_or_create_component("tool_registry")

            # Clear cache
            clear_tools_cache()

            # Cold load
            start = time.perf_counter()
            tools = get_all_tools()
            cold_time = (time.perf_counter() - start) * 1000
            component.record(cold_time, cache_hit=False)

            # Hot load
            start = time.perf_counter()
            _ = get_all_tools()
            hot_time = (time.perf_counter() - start) * 1000
            component.record(hot_time, cache_hit=True)

            # Lookup performance
            if tools and hasattr(tools[0], "name"):
                lookup_component = self.get_or_create_component("tool_lookup")

                for tool in tools[:5]:  # Test first 5 tools
                    start = time.perf_counter()
                    _ = get_tool_by_name(tool.name)
                    lookup_time = (time.perf_counter() - start) * 1000
                    lookup_component.record(lookup_time)

            print(
                f"  Tool registry: {len(tools)} tools, cold={cold_time:.0f}ms, hot={hot_time:.4f}ms"
            )

            if cold_time > 500:
                opportunities.append(
                    OptimizationOpportunity(
                        category="latency",
                        severity="low",
                        component="ToolRegistry",
                        issue=f"Tool loading takes {cold_time:.0f}ms",
                        recommendation="Consider preloading tools at server startup",
                        estimated_impact="Eliminate tool loading from request path",
                        implementation_effort="low",
                        code_location="app/shared/utils/toolhub.py",
                    )
                )

        except Exception as e:
            print(f"  ⚠️ Tool registry analysis failed: {e}")

        return opportunities

    async def analyze_llm_latency(self) -> List[OptimizationOpportunity]:
        """Analyze LLM call latency patterns."""
        opportunities = []

        try:
            from app.shared.utils.service_loader import (
                get_async_llm_service,
                get_llm_service,
            )

            llm_component = self.get_or_create_component("llm_calls")
            async_component = self.get_or_create_component("async_llm_calls")

            sync_service = get_llm_service()
            async_service = get_async_llm_service()

            # Test sync calls
            sync_times = []
            for i in range(2):
                start = time.perf_counter()
                _ = sync_service.generate_response(
                    prompt=f"What is {i+1}?", persona="hermes", user_id=f"analyzer_{i}"
                )
                duration = (time.perf_counter() - start) * 1000
                sync_times.append(duration)
                llm_component.record(duration)

            # Test async calls
            async_times = []
            for i in range(2):
                start = time.perf_counter()
                _ = await async_service.generate_async(
                    prompt=f"What is {i+2}?", persona="hermes"
                )
                duration = (time.perf_counter() - start) * 1000
                async_times.append(duration)
                async_component.record(duration)

            # Test parallel async
            parallel_component = self.get_or_create_component("parallel_llm")
            start = time.perf_counter()
            await asyncio.gather(
                async_service.generate_async("Q1?", persona="hermes"),
                async_service.generate_async("Q2?", persona="hermes"),
            )
            parallel_time = (time.perf_counter() - start) * 1000
            parallel_component.record(parallel_time)

            sequential_time = sum(async_times[:2])
            parallel_speedup = (
                (sequential_time - parallel_time) / sequential_time * 100
                if sequential_time > 0
                else 0
            )

            print(f"  LLM sync: avg={statistics.mean(sync_times):.0f}ms")
            print(f"  LLM async: avg={statistics.mean(async_times):.0f}ms")
            print(
                f"  LLM parallel: 2 calls in {parallel_time:.0f}ms (speedup: {parallel_speedup:.1f}%)"
            )

            # Detect opportunities
            avg_latency = statistics.mean(sync_times + async_times)

            if avg_latency > 2000:
                opportunities.append(
                    OptimizationOpportunity(
                        category="latency",
                        severity="high",
                        component="LLM API",
                        issue=f"Average LLM latency is {avg_latency:.0f}ms",
                        recommendation=(
                            "1. Implement response caching for identical prompts\n"
                            "2. Use streaming for long responses\n"
                            "3. Consider model selection based on query complexity"
                        ),
                        estimated_impact="Could reduce latency 20-50% for cached queries",
                        implementation_effort="medium",
                        code_location="app/shared/services/LLMService.py",
                    )
                )

            if parallel_speedup < 30:
                opportunities.append(
                    OptimizationOpportunity(
                        category="async",
                        severity="medium",
                        component="Async Execution",
                        issue=f"Parallel speedup is only {parallel_speedup:.1f}%",
                        recommendation="Check for blocking operations in async code path",
                        estimated_impact="Improve parallel efficiency",
                        implementation_effort="medium",
                    )
                )

        except Exception as e:
            print(f"  ⚠️ LLM latency analysis failed: {e}")

        return opportunities

    async def analyze_legion_graph(self) -> List[OptimizationOpportunity]:
        """Analyze Legion graph execution."""
        opportunities = []

        try:
            from app.hermes.legion.graph_service import LegionGraphService
            from app.hermes.models import ResponseMode, UserIdentity

            graph_component = self.get_or_create_component("legion_graph")

            service = LegionGraphService()
            user_identity = UserIdentity(
                user_id="analyzer_user",
                ip_address="127.0.0.1",
                user_agent="Analyzer/1.0",
            )

            # Simple query
            start = time.perf_counter()
            _ = await service.process_request(
                text="What is 2+2?",
                user_identity=user_identity,
                response_mode=ResponseMode.TEXT,
                persona="legion",
            )
            simple_time = (time.perf_counter() - start) * 1000
            graph_component.record(simple_time)

            print(f"  Legion graph: simple query={simple_time:.0f}ms")

            # Complex query
            start = time.perf_counter()
            _ = await service.process_request(
                text="Compare Python and JavaScript for web development",
                user_identity=user_identity,
                response_mode=ResponseMode.TEXT,
                persona="legion",
            )
            complex_time = (time.perf_counter() - start) * 1000
            graph_component.record(complex_time)

            print(f"  Legion graph: complex query={complex_time:.0f}ms")

            # Analyze routing overhead
            if simple_time > 3000:
                opportunities.append(
                    OptimizationOpportunity(
                        category="latency",
                        severity="medium",
                        component="Legion Routing",
                        issue=f"Simple queries take {simple_time:.0f}ms through Legion",
                        recommendation="Consider faster routing for obvious simple queries",
                        estimated_impact="Reduce simple query latency by 30-50%",
                        implementation_effort="medium",
                        code_location="app/hermes/legion/intelligence/routing_service.py",
                    )
                )

        except Exception as e:
            print(f"  ⚠️ Legion graph analysis failed: {e}")
            import traceback

            traceback.print_exc()

        return opportunities

    def analyze_caching_opportunities(self) -> List[OptimizationOpportunity]:
        """Analyze caching patterns and opportunities."""
        opportunities = []

        # Check components with low cache hit rates
        for name, component in self.components.items():
            if component.cache_misses > 0:
                hit_rate = component.cache_hit_rate

                if hit_rate < 50:
                    opportunities.append(
                        OptimizationOpportunity(
                            category="caching",
                            severity="high" if hit_rate < 25 else "medium",
                            component=name,
                            issue=f"Cache hit rate is only {hit_rate:.1f}%",
                            recommendation="Review cache key design and consider LRU cache size increase",
                            estimated_impact=f"Could improve hit rate to 80%+ saving {component.avg_time_ms:.0f}ms per miss",
                            implementation_effort="low",
                        )
                    )

        return opportunities

    async def run_full_analysis(
        self,
    ) -> Tuple[Dict[str, ComponentMetrics], List[OptimizationOpportunity]]:
        """Run complete bottleneck analysis."""

        self.analysis_start = time.time()
        self.opportunities = []

        print("\n" + "=" * 80)
        print(" BOTTLENECK ANALYSIS")
        print(" Identifying optimization opportunities...")
        print("=" * 80)

        # Service loading
        print("\n🔍 Analyzing service loading...")
        self.opportunities.extend(self.analyze_service_loading())

        # Tool registry
        print("\n🔍 Analyzing tool registry...")
        self.opportunities.extend(self.analyze_tool_registry())

        # LLM latency
        print("\n🔍 Analyzing LLM latency...")
        self.opportunities.extend(await self.analyze_llm_latency())

        # Legion graph
        print("\n🔍 Analyzing Legion graph...")
        self.opportunities.extend(await self.analyze_legion_graph())

        # Caching patterns
        print("\n🔍 Analyzing caching patterns...")
        self.opportunities.extend(self.analyze_caching_opportunities())

        self.analysis_end = time.time()

        return self.components, self.opportunities

    def generate_report(self) -> str:
        """Generate a comprehensive analysis report."""

        lines = []
        lines.append("\n" + "=" * 80)
        lines.append(" OPTIMIZATION OPPORTUNITY REPORT")
        lines.append(f" Generated: {datetime.now().isoformat()}")
        lines.append(
            f" Analysis Duration: {(self.analysis_end - self.analysis_start) * 1000:.0f}ms"
        )
        lines.append("=" * 80)

        # Component metrics
        lines.append("\n📊 COMPONENT METRICS")
        lines.append("-" * 80)
        lines.append(
            f"  {'Component':<30} | {'Calls':>6} | {'Avg (ms)':>10} | {'Cache Hit':>10}"
        )
        lines.append("  " + "-" * 70)

        for name, comp in sorted(self.components.items()):
            cache_str = (
                f"{comp.cache_hit_rate:.0f}%" if comp.cache_misses > 0 else "N/A"
            )
            lines.append(
                f"  {name:<30} | {comp.call_count:>6} | {comp.avg_time_ms:>10.1f} | {cache_str:>10}"
            )

        # Opportunities by priority
        lines.append("\n\n🎯 OPTIMIZATION OPPORTUNITIES (by priority)")
        lines.append("-" * 80)

        # Sort by priority
        sorted_opportunities = sorted(
            self.opportunities, key=lambda x: -x.priority_score
        )

        for i, opp in enumerate(sorted_opportunities, 1):
            severity_icon = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🟢",
            }.get(opp.severity, "⚪")

            lines.append(
                f"\n{i}. {severity_icon} [{opp.category.upper()}] {opp.component}"
            )
            lines.append(f"   Issue: {opp.issue}")
            lines.append(f"   Recommendation:")
            for rec_line in opp.recommendation.split("\n"):
                lines.append(f"      {rec_line}")
            lines.append(f"   Impact: {opp.estimated_impact}")
            lines.append(f"   Effort: {opp.implementation_effort}")
            if opp.code_location:
                lines.append(f"   Location: {opp.code_location}")

        # Summary
        lines.append("\n\n📈 SUMMARY")
        lines.append("-" * 80)

        by_severity = defaultdict(int)
        by_category = defaultdict(int)
        for opp in self.opportunities:
            by_severity[opp.severity] += 1
            by_category[opp.category] += 1

        lines.append(f"  Total opportunities: {len(self.opportunities)}")
        lines.append(f"  By severity: {dict(by_severity)}")
        lines.append(f"  By category: {dict(by_category)}")

        # Quick wins
        quick_wins = [o for o in self.opportunities if o.implementation_effort == "low"]
        if quick_wins:
            lines.append(f"\n  🏆 Quick Wins (low effort):")
            for qw in quick_wins[:3]:
                lines.append(f"     • {qw.component}: {qw.issue[:50]}...")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)

    def save_report(self, filepath: str = None):
        """Save report to file."""
        if filepath is None:
            os.makedirs("benchmarks/reports", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"benchmarks/reports/bottleneck_report_{timestamp}.txt"

        report = self.generate_report()

        with open(filepath, "w") as f:
            f.write(report)

        print(f"\n📁 Report saved to: {filepath}")

        # Also save JSON version
        json_path = filepath.replace(".txt", ".json")
        with open(json_path, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "components": {
                        name: {
                            "call_count": c.call_count,
                            "avg_time_ms": c.avg_time_ms,
                            "min_time_ms": (
                                c.min_time_ms if c.min_time_ms != float("inf") else 0
                            ),
                            "max_time_ms": c.max_time_ms,
                            "cache_hit_rate": c.cache_hit_rate,
                            "error_rate": c.error_rate,
                        }
                        for name, c in self.components.items()
                    },
                    "opportunities": [
                        {
                            "category": o.category,
                            "severity": o.severity,
                            "component": o.component,
                            "issue": o.issue,
                            "recommendation": o.recommendation,
                            "estimated_impact": o.estimated_impact,
                            "implementation_effort": o.implementation_effort,
                            "code_location": o.code_location,
                            "priority_score": o.priority_score,
                        }
                        for o in self.opportunities
                    ],
                },
                f,
                indent=2,
            )

        print(f"📁 JSON data saved to: {json_path}")


async def main():
    """Run bottleneck analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Bottleneck Analyzer")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate report without new analysis",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick analysis (fewer iterations)"
    )

    args = parser.parse_args()

    analyzer = BottleneckAnalyzer()

    # Run analysis
    await analyzer.run_full_analysis()

    # Generate and print report
    report = analyzer.generate_report()
    print(report)

    # Save report
    analyzer.save_report()


if __name__ == "__main__":
    asyncio.run(main())
