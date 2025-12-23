"""
Benchmarks package for Hermes & Legion performance testing.

Modules:
    performance_benchmarks: Standalone micro-benchmarks for pattern comparison
    integration_benchmarks: Integration tests with actual service implementations
    e2e_benchmarks: End-to-end tests with real LLM API calls
    profiler: Advanced system instrumentation and profiling
    bottleneck_analyzer: Automated optimization opportunity detection

Usage:
    # Run standalone micro-benchmarks (no API calls)
    python -m benchmarks.performance_benchmarks

    # Run integration benchmarks (service loading, no API calls)
    python -m benchmarks.integration_benchmarks

    # Run end-to-end benchmarks (REAL API calls - may incur costs!)
    python -m benchmarks.e2e_benchmarks

    # Run advanced profiling
    python -m benchmarks.profiler --detailed

    # Run bottleneck analysis and get optimization recommendations
    python -m benchmarks.bottleneck_analyzer
"""

from .bottleneck_analyzer import BottleneckAnalyzer
from .e2e_benchmarks import run_all_e2e_benchmarks
from .integration_benchmarks import run_all_integration_benchmarks
from .performance_benchmarks import run_all_benchmarks as run_micro_benchmarks
from .profiler import run_full_profiling

__all__ = [
    "run_micro_benchmarks",
    "run_all_integration_benchmarks",
    "run_all_e2e_benchmarks",
    "run_full_profiling",
    "BottleneckAnalyzer",
]
