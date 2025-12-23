#!/usr/bin/env python3
"""
Advanced Profiling System for Hermes & Legion.

This module provides deep instrumentation and profiling capabilities to identify
optimization opportunities in the system. It measures:

1. Component-level timing (routing, workers, synthesis, etc.)
2. Memory usage patterns
3. Network latency vs processing time
4. Cache hit/miss rates
5. Bottleneck detection with recommendations

Usage:
    python -m benchmarks.profiler
    python -m benchmarks.profiler --detailed
    python -m benchmarks.profiler --memory
"""

import asyncio
import gc
import json
import os
import statistics
import sys
import time
import tracemalloc
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TimingEntry:
    """Single timing measurement."""

    name: str
    duration_ms: float
    start_time: float
    end_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List["TimingEntry"] = field(default_factory=list)

    @property
    def self_time_ms(self) -> float:
        """Time spent in this operation excluding children."""
        child_time = sum(c.duration_ms for c in self.children)
        return max(0, self.duration_ms - child_time)


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""

    timestamp: float
    current_mb: float
    peak_mb: float
    label: str


@dataclass
class ProfileResult:
    """Complete profiling result."""

    name: str
    total_duration_ms: float
    timings: List[TimingEntry]
    memory_snapshots: List[MemorySnapshot]
    cache_stats: Dict[str, Dict[str, int]]
    bottlenecks: List[Dict[str, Any]]
    recommendations: List[str]


class Profiler:
    """
    Advanced profiler for deep system instrumentation.

    Features:
    - Hierarchical timing with nested spans
    - Memory tracking
    - Cache statistics
    - Bottleneck detection
    - Optimization recommendations
    """

    def __init__(self, name: str = "profiler"):
        self.name = name
        self.timings: List[TimingEntry] = []
        self.memory_snapshots: List[MemorySnapshot] = []
        self.cache_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"hits": 0, "misses": 0}
        )
        self._timing_stack: List[TimingEntry] = []
        self._start_time: float = 0
        self._memory_tracking = False

    def start(self):
        """Start profiling session."""
        self._start_time = time.perf_counter()
        self.timings = []
        self.memory_snapshots = []
        self._timing_stack = []

    def stop(self) -> ProfileResult:
        """Stop profiling and generate results."""
        total_duration = (time.perf_counter() - self._start_time) * 1000

        bottlenecks = self._detect_bottlenecks()
        recommendations = self._generate_recommendations(bottlenecks)

        return ProfileResult(
            name=self.name,
            total_duration_ms=total_duration,
            timings=self.timings,
            memory_snapshots=self.memory_snapshots,
            cache_stats=dict(self.cache_stats),
            bottlenecks=bottlenecks,
            recommendations=recommendations,
        )

    @contextmanager
    def span(self, name: str, **metadata):
        """Create a timing span (can be nested)."""
        start = time.perf_counter()
        entry = TimingEntry(
            name=name,
            duration_ms=0,
            start_time=start,
            end_time=0,
            metadata=metadata,
        )

        # Add to parent's children if nested
        if self._timing_stack:
            self._timing_stack[-1].children.append(entry)
        else:
            self.timings.append(entry)

        self._timing_stack.append(entry)

        try:
            yield entry
        finally:
            end = time.perf_counter()
            entry.end_time = end
            entry.duration_ms = (end - start) * 1000
            self._timing_stack.pop()

    def record_cache_hit(self, cache_name: str):
        """Record a cache hit."""
        self.cache_stats[cache_name]["hits"] += 1

    def record_cache_miss(self, cache_name: str):
        """Record a cache miss."""
        self.cache_stats[cache_name]["misses"] += 1

    def snapshot_memory(self, label: str = ""):
        """Take a memory snapshot."""
        if not self._memory_tracking:
            tracemalloc.start()
            self._memory_tracking = True

        current, peak = tracemalloc.get_traced_memory()
        self.memory_snapshots.append(
            MemorySnapshot(
                timestamp=time.perf_counter(),
                current_mb=current / 1024 / 1024,
                peak_mb=peak / 1024 / 1024,
                label=label,
            )
        )

    def _detect_bottlenecks(self) -> List[Dict[str, Any]]:
        """Detect performance bottlenecks."""
        bottlenecks = []

        # Analyze timing data
        flat_timings = self._flatten_timings(self.timings)

        if not flat_timings:
            return bottlenecks

        total_time = sum(t.duration_ms for t in self.timings)

        # Find operations that take > 30% of total time
        for timing in flat_timings:
            pct = (timing.duration_ms / total_time * 100) if total_time > 0 else 0
            if pct > 30:
                bottlenecks.append(
                    {
                        "type": "time_dominant",
                        "operation": timing.name,
                        "duration_ms": timing.duration_ms,
                        "percentage": pct,
                        "severity": "high" if pct > 50 else "medium",
                        "description": f"'{timing.name}' takes {pct:.1f}% of total execution time",
                    }
                )

        # Find slow individual operations (> 1000ms)
        for timing in flat_timings:
            if timing.duration_ms > 1000 and timing.name not in [
                b["operation"] for b in bottlenecks
            ]:
                bottlenecks.append(
                    {
                        "type": "slow_operation",
                        "operation": timing.name,
                        "duration_ms": timing.duration_ms,
                        "severity": "high" if timing.duration_ms > 3000 else "medium",
                        "description": f"'{timing.name}' took {timing.duration_ms:.0f}ms",
                    }
                )

        # Check cache efficiency
        for cache_name, stats in self.cache_stats.items():
            total = stats["hits"] + stats["misses"]
            if total > 0:
                hit_rate = stats["hits"] / total * 100
                if hit_rate < 50:
                    bottlenecks.append(
                        {
                            "type": "cache_miss",
                            "cache": cache_name,
                            "hit_rate": hit_rate,
                            "severity": "medium" if hit_rate > 25 else "high",
                            "description": f"Cache '{cache_name}' has only {hit_rate:.1f}% hit rate",
                        }
                    )

        # Check memory growth
        if len(self.memory_snapshots) >= 2:
            first = self.memory_snapshots[0]
            last = self.memory_snapshots[-1]
            growth = last.current_mb - first.current_mb
            if growth > 50:  # More than 50MB growth
                bottlenecks.append(
                    {
                        "type": "memory_growth",
                        "growth_mb": growth,
                        "severity": "high" if growth > 100 else "medium",
                        "description": f"Memory grew by {growth:.1f}MB during execution",
                    }
                )

        return sorted(
            bottlenecks,
            key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(
                x.get("severity", "low"), 2
            ),
        )

    def _generate_recommendations(self, bottlenecks: List[Dict[str, Any]]) -> List[str]:
        """Generate optimization recommendations based on bottlenecks."""
        recommendations = []

        for bottleneck in bottlenecks:
            btype = bottleneck["type"]

            if btype == "time_dominant":
                op = bottleneck["operation"]
                if "llm" in op.lower() or "gemini" in op.lower():
                    recommendations.append(
                        f"🔸 LLM call '{op}' dominates execution time. Consider:\n"
                        f"   - Caching responses for identical prompts\n"
                        f"   - Using streaming for long responses\n"
                        f"   - Parallel execution where possible"
                    )
                elif "routing" in op.lower():
                    recommendations.append(
                        f"🔸 Routing '{op}' is slow. Consider:\n"
                        f"   - Caching routing decisions for similar queries\n"
                        f"   - Using simpler heuristics before LLM routing"
                    )
                else:
                    recommendations.append(
                        f"🔸 '{op}' takes {bottleneck['percentage']:.1f}% of time. Profile deeper to optimize."
                    )

            elif btype == "slow_operation":
                op = bottleneck["operation"]
                recommendations.append(
                    f"🔸 Slow operation '{op}' ({bottleneck['duration_ms']:.0f}ms). Consider:\n"
                    f"   - Adding caching layer\n"
                    f"   - Async execution if blocking\n"
                    f"   - Breaking into smaller units"
                )

            elif btype == "cache_miss":
                recommendations.append(
                    f"🔸 Low cache hit rate for '{bottleneck['cache']}' ({bottleneck['hit_rate']:.1f}%). Consider:\n"
                    f"   - Increasing cache size\n"
                    f"   - Improving cache key design\n"
                    f"   - Pre-warming cache on startup"
                )

            elif btype == "memory_growth":
                recommendations.append(
                    f"🔸 Memory grew by {bottleneck['growth_mb']:.1f}MB. Consider:\n"
                    f"   - Implementing object pooling\n"
                    f"   - Using __slots__ for data classes\n"
                    f"   - Clearing large objects after use"
                )

        if not recommendations:
            recommendations.append("✅ No significant bottlenecks detected!")

        return recommendations

    def _flatten_timings(self, timings: List[TimingEntry]) -> List[TimingEntry]:
        """Flatten nested timings for analysis."""
        result = []
        for t in timings:
            result.append(t)
            result.extend(self._flatten_timings(t.children))
        return result


class LegionProfiler(Profiler):
    """
    Specialized profiler for Legion graph execution.

    Automatically instruments:
    - Routing decisions
    - Worker execution
    - Synthesis steps
    - LLM calls
    """

    def __init__(self):
        super().__init__("legion_profiler")
        self.llm_calls: List[Dict[str, Any]] = []
        self.routing_decisions: List[Dict[str, Any]] = []
        self.worker_executions: List[Dict[str, Any]] = []

    def record_llm_call(
        self,
        prompt_length: int,
        response_length: int,
        duration_ms: float,
        persona: str,
        model: str,
    ):
        """Record an LLM call."""
        self.llm_calls.append(
            {
                "prompt_length": prompt_length,
                "response_length": response_length,
                "duration_ms": duration_ms,
                "persona": persona,
                "model": model,
                "tokens_estimated": (prompt_length + response_length) // 4,
                "tokens_per_sec": (
                    (response_length // 4) / (duration_ms / 1000)
                    if duration_ms > 0
                    else 0
                ),
            }
        )

    def record_routing_decision(
        self, query: str, decision: str, confidence: float, duration_ms: float
    ):
        """Record a routing decision."""
        self.routing_decisions.append(
            {
                "query_length": len(query),
                "decision": decision,
                "confidence": confidence,
                "duration_ms": duration_ms,
            }
        )

    def record_worker_execution(
        self, worker_id: str, task: str, duration_ms: float, success: bool
    ):
        """Record a worker execution."""
        self.worker_executions.append(
            {
                "worker_id": worker_id,
                "task_length": len(task),
                "duration_ms": duration_ms,
                "success": success,
            }
        )

    def get_llm_stats(self) -> Dict[str, Any]:
        """Get LLM call statistics."""
        if not self.llm_calls:
            return {}

        durations = [c["duration_ms"] for c in self.llm_calls]
        return {
            "total_calls": len(self.llm_calls),
            "total_time_ms": sum(durations),
            "avg_time_ms": statistics.mean(durations),
            "min_time_ms": min(durations),
            "max_time_ms": max(durations),
            "total_tokens_estimated": sum(
                c["tokens_estimated"] for c in self.llm_calls
            ),
            "avg_tokens_per_sec": statistics.mean(
                [c["tokens_per_sec"] for c in self.llm_calls]
            ),
        }

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        if not self.routing_decisions:
            return {}

        durations = [r["duration_ms"] for r in self.routing_decisions]
        decisions = [r["decision"] for r in self.routing_decisions]

        return {
            "total_decisions": len(self.routing_decisions),
            "avg_time_ms": statistics.mean(durations),
            "avg_confidence": statistics.mean(
                [r["confidence"] for r in self.routing_decisions]
            ),
            "decision_distribution": {d: decisions.count(d) for d in set(decisions)},
        }


def profile_function(profiler: Profiler = None):
    """Decorator to profile a function."""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            p = profiler or Profiler()
            with p.span(func.__name__):
                return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            p = profiler or Profiler()
            with p.span(func.__name__):
                return await func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


# ============================================================================
# Profile Analysis and Reporting
# ============================================================================


def print_profile_report(result: ProfileResult, detailed: bool = False):
    """Print a detailed profile report."""

    print("\n" + "=" * 80)
    print(f" PROFILE REPORT: {result.name}")
    print(f" Total Duration: {result.total_duration_ms:.0f}ms")
    print("=" * 80)

    # Timing breakdown
    print("\n📊 TIMING BREAKDOWN")
    print("-" * 80)
    _print_timing_tree(result.timings, indent=0, total_time=result.total_duration_ms)

    # Cache stats
    if result.cache_stats:
        print("\n💾 CACHE STATISTICS")
        print("-" * 80)
        for name, stats in result.cache_stats.items():
            total = stats["hits"] + stats["misses"]
            hit_rate = (stats["hits"] / total * 100) if total > 0 else 0
            status = "✅" if hit_rate > 80 else "⚠️" if hit_rate > 50 else "❌"
            print(
                f"  {status} {name}: {hit_rate:.1f}% hit rate ({stats['hits']}/{total})"
            )

    # Memory
    if result.memory_snapshots:
        print("\n🧠 MEMORY USAGE")
        print("-" * 80)
        for snap in result.memory_snapshots:
            print(
                f"  {snap.label or 'snapshot'}: {snap.current_mb:.2f}MB (peak: {snap.peak_mb:.2f}MB)"
            )

    # Bottlenecks
    if result.bottlenecks:
        print("\n🚨 BOTTLENECKS DETECTED")
        print("-" * 80)
        for b in result.bottlenecks:
            severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                b.get("severity", "low"), "⚪"
            )
            print(f"  {severity_icon} [{b['type'].upper()}] {b['description']}")

    # Recommendations
    print("\n💡 OPTIMIZATION RECOMMENDATIONS")
    print("-" * 80)
    for rec in result.recommendations:
        print(f"  {rec}")

    print("\n" + "=" * 80)


def _print_timing_tree(timings: List[TimingEntry], indent: int, total_time: float):
    """Print timing tree with percentages."""
    for timing in timings:
        pct = (timing.duration_ms / total_time * 100) if total_time > 0 else 0
        bar_len = int(pct / 2)  # Max 50 chars
        bar = "█" * bar_len + "░" * (50 - bar_len)

        prefix = "  " * indent
        self_pct = (timing.self_time_ms / total_time * 100) if total_time > 0 else 0

        if timing.children:
            print(
                f"{prefix}├─ {timing.name}: {timing.duration_ms:.0f}ms ({pct:.1f}%) [self: {timing.self_time_ms:.0f}ms ({self_pct:.1f}%)]"
            )
        else:
            print(f"{prefix}├─ {timing.name}: {timing.duration_ms:.0f}ms ({pct:.1f}%)")

        if timing.children:
            _print_timing_tree(timing.children, indent + 1, total_time)


def save_profile_json(result: ProfileResult, filepath: str):
    """Save profile result to JSON for further analysis."""
    data = {
        "name": result.name,
        "total_duration_ms": result.total_duration_ms,
        "timestamp": datetime.now().isoformat(),
        "timings": _timings_to_dict(result.timings),
        "memory_snapshots": [
            {"label": s.label, "current_mb": s.current_mb, "peak_mb": s.peak_mb}
            for s in result.memory_snapshots
        ],
        "cache_stats": result.cache_stats,
        "bottlenecks": result.bottlenecks,
        "recommendations": result.recommendations,
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n📁 Profile saved to: {filepath}")


def _timings_to_dict(timings: List[TimingEntry]) -> List[Dict]:
    """Convert timings to dictionary."""
    return [
        {
            "name": t.name,
            "duration_ms": t.duration_ms,
            "self_time_ms": t.self_time_ms,
            "metadata": t.metadata,
            "children": _timings_to_dict(t.children),
        }
        for t in timings
    ]


# ============================================================================
# Legion-Specific Profiling Functions
# ============================================================================


async def profile_legion_request(query: str, detailed: bool = False) -> ProfileResult:
    """Profile a complete Legion request with deep instrumentation."""

    from app.hermes.legion.graph_service import LegionGraphService
    from app.hermes.models import ResponseMode, UserIdentity

    profiler = LegionProfiler()
    profiler.start()

    # Memory snapshot at start
    profiler.snapshot_memory("start")

    user_identity = UserIdentity(
        user_id="profiler_user", ip_address="127.0.0.1", user_agent="Profiler/1.0"
    )

    with profiler.span("total_request", query=query[:50]):

        with profiler.span("service_initialization"):
            service = LegionGraphService()

        with profiler.span("process_request"):
            result = await service.process_request(
                text=query,
                user_identity=user_identity,
                response_mode=ResponseMode.TEXT,
                persona="legion",
            )

        profiler.snapshot_memory("after_request")

    return profiler.stop()


async def profile_llm_service(iterations: int = 3) -> ProfileResult:
    """Profile LLM service with detailed timing."""

    from app.shared.utils.service_loader import get_async_llm_service, get_llm_service

    profiler = LegionProfiler()
    profiler.start()
    profiler.snapshot_memory("start")

    with profiler.span("llm_service_profiling"):

        # Test sync service
        with profiler.span("sync_service"):
            sync_service = get_llm_service()

            for i in range(iterations):
                with profiler.span(f"sync_call_{i+1}"):
                    # Check if model is cached
                    cache_key = "hermes:gemini-2.5-flash:0.3"
                    if (
                        hasattr(sync_service, "_model_cache")
                        and cache_key in sync_service._model_cache
                    ):
                        profiler.record_cache_hit("model_cache")
                    else:
                        profiler.record_cache_miss("model_cache")

                    start = time.perf_counter()
                    response = sync_service.generate_gemini_response(
                        prompt=f"What is {i+1} + {i+1}?",
                        persona="hermes",
                        user_id=f"profiler_{i}",
                    )
                    duration = (time.perf_counter() - start) * 1000

                    profiler.record_llm_call(
                        prompt_length=20,
                        response_length=len(response),
                        duration_ms=duration,
                        persona="hermes",
                        model="gemini-2.5-flash",
                    )

        profiler.snapshot_memory("after_sync")

        # Test async service
        with profiler.span("async_service"):
            async_service = get_async_llm_service()

            for i in range(iterations):
                with profiler.span(f"async_call_{i+1}"):
                    start = time.perf_counter()
                    response = await async_service.generate_async(
                        prompt=f"What is {i+2} * 2?", persona="hermes"
                    )
                    duration = (time.perf_counter() - start) * 1000

                    profiler.record_llm_call(
                        prompt_length=20,
                        response_length=len(response),
                        duration_ms=duration,
                        persona="hermes",
                        model="gemini-2.5-flash",
                    )

        profiler.snapshot_memory("after_async")

        # Test parallel async
        with profiler.span("parallel_async"):
            start = time.perf_counter()
            tasks = [
                async_service.generate_async(f"What is {i}?", persona="hermes")
                for i in range(3)
            ]
            await asyncio.gather(*tasks)
            duration = (time.perf_counter() - start) * 1000

        profiler.snapshot_memory("end")

    result = profiler.stop()

    # Add LLM-specific stats
    llm_stats = profiler.get_llm_stats()
    if llm_stats:
        result.bottlenecks.append(
            {
                "type": "llm_stats",
                "description": f"LLM calls: {llm_stats['total_calls']}, avg: {llm_stats['avg_time_ms']:.0f}ms, tokens/sec: {llm_stats['avg_tokens_per_sec']:.1f}",
                "severity": "info",
            }
        )

    return result


# ============================================================================
# Main Profiling Runner
# ============================================================================


async def run_full_profiling(detailed: bool = False, save_json: bool = True):
    """Run complete profiling suite."""

    print("\n" + "=" * 80)
    print(" HERMES & LEGION PROFILING SYSTEM")
    print(" Deep instrumentation for optimization discovery")
    print(f" Started: {datetime.now().isoformat()}")
    print("=" * 80)

    results = []

    # Profile LLM Service
    print("\n\n📊 Profiling LLM Service...")
    llm_result = await profile_llm_service(iterations=2)
    print_profile_report(llm_result, detailed=detailed)
    results.append(llm_result)

    # Profile Legion Request
    print("\n\n📊 Profiling Legion Request...")
    legion_result = await profile_legion_request(
        "What are the benefits of microservices architecture?", detailed=detailed
    )
    print_profile_report(legion_result, detailed=detailed)
    results.append(legion_result)

    # Save results if requested
    if save_json:
        os.makedirs("benchmarks/reports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for result in results:
            filepath = f"benchmarks/reports/profile_{result.name}_{timestamp}.json"
            save_profile_json(result, filepath)

    # Summary
    print("\n" + "=" * 80)
    print(" PROFILING COMPLETE")
    print("=" * 80)

    total_bottlenecks = sum(len(r.bottlenecks) for r in results)
    total_recommendations = sum(len(r.recommendations) for r in results)

    print(f"\n  📊 Profiles generated: {len(results)}")
    print(f"  🚨 Bottlenecks found: {total_bottlenecks}")
    print(f"  💡 Recommendations: {total_recommendations}")

    return results


def main():
    """Entry point for profiling."""
    import argparse

    parser = argparse.ArgumentParser(description="Profile Hermes & Legion")
    parser.add_argument("--detailed", action="store_true", help="Show detailed output")
    parser.add_argument(
        "--no-save", action="store_true", help="Don't save JSON reports"
    )
    parser.add_argument(
        "--memory", action="store_true", help="Focus on memory profiling"
    )

    args = parser.parse_args()

    asyncio.run(run_full_profiling(detailed=args.detailed, save_json=not args.no_save))


if __name__ == "__main__":
    main()
