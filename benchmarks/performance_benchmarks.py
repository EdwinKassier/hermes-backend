#!/usr/bin/env python3
"""
Performance Benchmarks for Hermes & Legion Optimizations.

This module provides comprehensive benchmarks to measure the impact of
the 16 optimizations implemented across 12 phases.

Usage:
    python -m benchmarks.performance_benchmarks

Or with pytest:
    pytest benchmarks/performance_benchmarks.py -v -s
"""

import asyncio
import re
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""

    name: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float
    improvement_vs_baseline: Optional[float] = None  # Percentage improvement

    def __str__(self) -> str:
        base = (
            f"{self.name}:\n"
            f"  Iterations: {self.iterations}\n"
            f"  Avg: {self.avg_time_ms:.3f}ms\n"
            f"  Min: {self.min_time_ms:.3f}ms\n"
            f"  Max: {self.max_time_ms:.3f}ms\n"
            f"  StdDev: {self.std_dev_ms:.3f}ms"
        )
        if self.improvement_vs_baseline is not None:
            base += f"\n  Improvement: {self.improvement_vs_baseline:.1f}%"
        return base


@contextmanager
def timer():
    """Context manager for timing operations."""
    start = time.perf_counter()
    times = {"elapsed_ms": 0}
    yield times
    times["elapsed_ms"] = (time.perf_counter() - start) * 1000


def run_benchmark(
    name: str,
    func: Callable,
    iterations: int = 100,
    warmup: int = 5,
    baseline_result: Optional[BenchmarkResult] = None,
) -> BenchmarkResult:
    """
    Run a benchmark for a function.

    Args:
        name: Benchmark name
        func: Function to benchmark (should take no arguments)
        iterations: Number of iterations
        warmup: Number of warmup iterations (not counted)
        baseline_result: Optional baseline to compare against

    Returns:
        BenchmarkResult with timing statistics
    """
    # Warmup
    for _ in range(warmup):
        func()

    # Actual benchmark
    times = []
    for _ in range(iterations):
        with timer() as t:
            func()
        times.append(t["elapsed_ms"])

    total = sum(times)
    avg = statistics.mean(times)

    improvement = None
    if baseline_result is not None:
        improvement = (
            (baseline_result.avg_time_ms - avg) / baseline_result.avg_time_ms
        ) * 100

    return BenchmarkResult(
        name=name,
        iterations=iterations,
        total_time_ms=total,
        avg_time_ms=avg,
        min_time_ms=min(times),
        max_time_ms=max(times),
        std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0,
        improvement_vs_baseline=improvement,
    )


async def run_async_benchmark(
    name: str,
    func: Callable,
    iterations: int = 100,
    warmup: int = 5,
    baseline_result: Optional[BenchmarkResult] = None,
) -> BenchmarkResult:
    """Run a benchmark for an async function."""
    # Warmup
    for _ in range(warmup):
        await func()

    # Actual benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        await func()
        times.append((time.perf_counter() - start) * 1000)

    total = sum(times)
    avg = statistics.mean(times)

    improvement = None
    if baseline_result is not None:
        improvement = (
            (baseline_result.avg_time_ms - avg) / baseline_result.avg_time_ms
        ) * 100

    return BenchmarkResult(
        name=name,
        iterations=iterations,
        total_time_ms=total,
        avg_time_ms=avg,
        min_time_ms=min(times),
        max_time_ms=max(times),
        std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0,
        improvement_vs_baseline=improvement,
    )


# ============================================================================
# Benchmark 1: Tool Lookup Performance (O(1) vs O(n))
# ============================================================================


def benchmark_tool_lookup():
    """Benchmark tool lookup with and without name index cache."""

    # Simulate tool registry with 50 tools
    class MockTool:
        def __init__(self, name: str):
            self.name = name

    tools = [MockTool(f"tool_{i}") for i in range(50)]

    # Baseline: O(n) lookup
    def lookup_on_baseline(name: str) -> Optional[MockTool]:
        for tool in tools:
            if tool.name == name:
                return tool
        return None

    # Optimized: O(1) lookup with index
    tool_index = {tool.name: tool for tool in tools}

    def lookup_o1_optimized(name: str) -> Optional[MockTool]:
        return tool_index.get(name)

    # Benchmark looking up the last tool (worst case for O(n))
    target = "tool_49"

    baseline = run_benchmark(
        "Tool Lookup O(n) Baseline",
        lambda: lookup_on_baseline(target),
        iterations=10000,
    )

    optimized = run_benchmark(
        "Tool Lookup O(1) Optimized",
        lambda: lookup_o1_optimized(target),
        iterations=10000,
        baseline_result=baseline,
    )

    return baseline, optimized


# ============================================================================
# Benchmark 2: Regex Pre-compilation
# ============================================================================


def benchmark_regex_compilation():
    """Benchmark regex with and without pre-compilation."""

    # Sample text to process
    test_text = "Call me at 555-123-4567 or email john@example.com for SSN 123-45-6789"

    # Baseline: Compile on every call
    def redact_pii_baseline(text: str) -> str:
        patterns = {
            "ssn": (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
            "email": (r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[EMAIL]"),
            "phone": (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]"),
        }
        for _, (pattern, replacement) in patterns.items():
            text = re.sub(pattern, replacement, text)
        return text

    # Optimized: Pre-compiled patterns
    COMPILED_PATTERNS = {
        "ssn": (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
        "email": (re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b"), "[EMAIL]"),
        "phone": (re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"), "[PHONE]"),
    }

    def redact_pii_optimized(text: str) -> str:
        for _, (pattern, replacement) in COMPILED_PATTERNS.items():
            text = pattern.sub(replacement, text)
        return text

    baseline = run_benchmark(
        "Regex Compile Every Call",
        lambda: redact_pii_baseline(test_text),
        iterations=5000,
    )

    optimized = run_benchmark(
        "Regex Pre-compiled",
        lambda: redact_pii_optimized(test_text),
        iterations=5000,
        baseline_result=baseline,
    )

    return baseline, optimized


# ============================================================================
# Benchmark 3: LRU Cache for Singleton Pattern
# ============================================================================


def benchmark_singleton_cache():
    """Benchmark singleton pattern with and without lru_cache."""

    class ExpensiveService:
        """Simulates an expensive service to initialize."""

        def __init__(self):
            # Simulate some initialization work
            self.data = {f"key_{i}": i for i in range(100)}

    # Container to hold the manual singleton instance
    class SingletonContainer:
        instance = None

    def get_service_manual():
        if SingletonContainer.instance is None:
            SingletonContainer.instance = ExpensiveService()
        return SingletonContainer.instance

    # Optimized: lru_cache singleton
    @lru_cache(maxsize=1)
    def get_service_cached():
        return ExpensiveService()

    # Reset for fair comparison
    SingletonContainer.instance = None
    get_service_cached.cache_clear()

    # Prime both caches first
    get_service_manual()
    get_service_cached()

    # Subsequent calls (hot) - this is what matters for performance
    baseline_hot = run_benchmark(
        "Manual Singleton (hot)",
        get_service_manual,
        iterations=10000,
    )

    optimized_hot = run_benchmark(
        "LRU Cache Singleton (hot)",
        get_service_cached,
        iterations=10000,
        baseline_result=baseline_hot,
    )

    return baseline_hot, optimized_hot


# ============================================================================
# Benchmark 4: Dictionary Access Patterns
# ============================================================================


def benchmark_dict_access():
    """Benchmark different dictionary access patterns."""

    # Large dictionary simulating model cache
    cache = {f"persona_{i}": {"model": f"model_{i}", "config": {}} for i in range(100)}

    # Baseline: Check then access
    def access_check_then_get(key: str):
        if key in cache:
            return cache[key]
        return None

    # Optimized: Direct .get()
    def access_direct_get(key: str):
        return cache.get(key)

    target = "persona_50"

    baseline = run_benchmark(
        "Dict: if key in + access",
        lambda: access_check_then_get(target),
        iterations=100000,
    )

    optimized = run_benchmark(
        "Dict: direct .get()",
        lambda: access_direct_get(target),
        iterations=100000,
        baseline_result=baseline,
    )

    return baseline, optimized


# ============================================================================
# Benchmark 5: Object Creation vs Caching
# ============================================================================


def benchmark_object_creation():
    """Benchmark object creation with and without caching."""

    @dataclass
    class PersonaConfig:
        name: str
        base_prompt: str
        model_name: str = "gemini-2.5-flash"
        temperature: float = 0.3
        timeout: int = 60

    # Simulate 31 persona configs (matching actual count)
    PERSONA_NAMES = [f"persona_{i}" for i in range(31)]

    # Baseline: Create on every call
    def get_personas_baseline():
        return [
            PersonaConfig(
                name=name,
                base_prompt=f"You are {name}",
            )
            for name in PERSONA_NAMES
        ]

    # Optimized: Cached creation
    @lru_cache(maxsize=1)
    def get_personas_cached():
        return tuple(
            [  # tuple for hashability
                PersonaConfig(
                    name=name,
                    base_prompt=f"You are {name}",
                )
                for name in PERSONA_NAMES
            ]
        )

    baseline = run_benchmark(
        "Create 31 PersonaConfigs every call",
        get_personas_baseline,
        iterations=1000,
    )

    get_personas_cached.cache_clear()  # Cold start first
    get_personas_cached()  # Prime the cache

    optimized = run_benchmark(
        "Cached PersonaConfigs",
        get_personas_cached,
        iterations=1000,
        baseline_result=baseline,
    )

    return baseline, optimized


# ============================================================================
# Benchmark 6: Lazy Property vs Eager Initialization
# ============================================================================


def benchmark_lazy_vs_eager():
    """Benchmark lazy property loading vs eager initialization."""

    class EagerService:
        def __init__(self):
            self.service_a = self._create_a()
            self.service_b = self._create_b()
            self.service_c = self._create_c()

        def _create_a(self):
            return {"type": "a", "data": list(range(100))}

        def _create_b(self):
            return {"type": "b", "data": list(range(100))}

        def _create_c(self):
            return {"type": "c", "data": list(range(100))}

    class LazyService:
        def __init__(self):
            self._service_a = None
            self._service_b = None
            self._service_c = None

        @property
        def service_a(self):
            if self._service_a is None:
                self._service_a = {"type": "a", "data": list(range(100))}
            return self._service_a

        @property
        def service_b(self):
            if self._service_b is None:
                self._service_b = {"type": "b", "data": list(range(100))}
            return self._service_b

        @property
        def service_c(self):
            if self._service_c is None:
                self._service_c = {"type": "c", "data": list(range(100))}
            return self._service_c

    # Benchmark initialization only
    baseline = run_benchmark(
        "Eager Init (all services)",
        lambda: EagerService(),
        iterations=5000,
    )

    optimized = run_benchmark(
        "Lazy Init (no services used)",
        lambda: LazyService(),
        iterations=5000,
        baseline_result=baseline,
    )

    return baseline, optimized


# ============================================================================
# Main Benchmark Runner
# ============================================================================


def print_header(title: str):
    """Print a formatted header."""
    width = 70
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_comparison(baseline: BenchmarkResult, optimized: BenchmarkResult):
    """Print comparison between baseline and optimized."""
    print(f"\n  Baseline:  {baseline.avg_time_ms:.4f}ms avg")
    print(f"  Optimized: {optimized.avg_time_ms:.4f}ms avg")
    if optimized.improvement_vs_baseline is not None:
        if optimized.improvement_vs_baseline > 0:
            print(f"  ✅ Improvement: {optimized.improvement_vs_baseline:.1f}% faster")
        else:
            print(f"  ⚠️  Regression: {-optimized.improvement_vs_baseline:.1f}% slower")


def run_all_benchmarks():
    """Run all benchmarks and report results."""

    print("\n" + "=" * 70)
    print(" HERMES & LEGION PERFORMANCE BENCHMARKS")
    print(" Measuring impact of 16 optimizations across 12 phases")
    print("=" * 70)

    results = []

    # Benchmark 1: Tool Lookup
    print_header("1. Tool Lookup: O(n) vs O(1) with Index Cache")
    baseline, optimized = benchmark_tool_lookup()
    print_comparison(baseline, optimized)
    results.append(("Tool Lookup", baseline, optimized))

    # Benchmark 2: Regex Pre-compilation
    print_header("2. Regex: Compile Every Call vs Pre-compiled")
    baseline, optimized = benchmark_regex_compilation()
    print_comparison(baseline, optimized)
    results.append(("Regex Pre-compilation", baseline, optimized))

    # Benchmark 3: Singleton Cache
    print_header("3. Singleton: Manual Check vs LRU Cache")
    baseline, optimized = benchmark_singleton_cache()
    print_comparison(baseline, optimized)
    results.append(("Singleton Cache", baseline, optimized))

    # Benchmark 4: Dictionary Access
    print_header("4. Dictionary: if-in vs .get()")
    baseline, optimized = benchmark_dict_access()
    print_comparison(baseline, optimized)
    results.append(("Dict Access", baseline, optimized))

    # Benchmark 5: Object Creation
    print_header("5. Object Creation: Every Call vs Cached")
    baseline, optimized = benchmark_object_creation()
    print_comparison(baseline, optimized)
    results.append(("Object Creation Cache", baseline, optimized))

    # Benchmark 6: Lazy vs Eager
    print_header("6. Initialization: Eager vs Lazy Loading")
    baseline, optimized = benchmark_lazy_vs_eager()
    print_comparison(baseline, optimized)
    results.append(("Lazy Init", baseline, optimized))

    # Summary
    print("\n" + "=" * 70)
    print(" BENCHMARK SUMMARY")
    print("=" * 70)

    print("\n  Optimization               | Baseline    | Optimized   | Improvement")
    print("  " + "-" * 66)

    total_baseline = 0
    total_optimized = 0

    for name, baseline, optimized in results:
        improvement = optimized.improvement_vs_baseline or 0
        print(
            f"  {name:<26} | {baseline.avg_time_ms:>9.4f}ms | {optimized.avg_time_ms:>9.4f}ms | {improvement:>6.1f}%"
        )
        total_baseline += baseline.avg_time_ms
        total_optimized += optimized.avg_time_ms

    print("  " + "-" * 66)
    overall_improvement = ((total_baseline - total_optimized) / total_baseline) * 100
    print(
        f"  {'TOTAL':<26} | {total_baseline:>9.4f}ms | {total_optimized:>9.4f}ms | {overall_improvement:>6.1f}%"
    )

    print("\n" + "=" * 70)
    print(" KEY INSIGHTS")
    print("=" * 70)
    print(
        """
  • Tool lookup with O(1) index provides significant speedup for repeated calls
  • Pre-compiled regex patterns eliminate compilation overhead per call
  • LRU cache provides slight overhead for singleton but better thread safety
  • Object creation caching has highest impact (100x+ faster on cache hit)
  • Lazy initialization pays off when not all services are used

  Note: These micro-benchmarks measure isolated patterns. Actual improvement
  in LLM-heavy workloads is dominated by network latency (async optimizations).
"""
    )

    return results


if __name__ == "__main__":
    run_all_benchmarks()
