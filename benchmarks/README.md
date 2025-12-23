# Hermes & Legion Benchmarking Suite 🚀

This directory contains a comprehensive benchmarking and profiling system for the Hermes & Legion architecture. It is designed to verify optimizations, identify performance bottlenecks, and prevent regressions.

## 📊 Modules

| Module | Purpose | Impact Target |
|--------|---------|---------------|
| `performance_benchmarks.py` | Micro-benchmarks for specific coding patterns (regex, caching, etc.) | CPU / Efficiency |
| `integration_benchmarks.py` | Integration tests verifying service loading and interactions | Startup / Config |
| `e2e_benchmarks.py` | End-to-end tests making **REAL API CALLS** to measure system latency | Latency / Accuracy |
| `profiler.py` | Advanced system instrumentation (hierarchical timing, memory tracking) | Deep Analysis |
| `bottleneck_analyzer.py` | Automated analysis tool that identifies issues and suggests fixes | Optimization |

## 🚀 Quick Start

### Run the Bottleneck Analyzer (Recommended)
This runs a full analysis and provides prioritized optimization recommendations:
```bash
python -m benchmarks.bottleneck_analyzer
```

### Run Micro-Benchmarks
Test the efficiency of internal patterns (0 cost):
```bash
python -m benchmarks.performance_benchmarks
```

### Run E2E Benchmarks
Test the full system latency and success rate (costs API tokens):
```bash
python -m benchmarks.e2e_benchmarks
```

## 🔍 Advanced Usage

### Profiling a Specific Request
To profile a specific request with deep instrumentation:
```bash
python -m benchmarks.profiler --detailed
```

This generates a JSON report in `benchmarks/reports/` containing:
- Hierarchical timing tree
- Memory usage snapshots
- Cache hit/miss statistics
- Detected bottlenecks

### Optimization Maturity Verification
Run the verification suite to confirm the system meets optimization standards:
```bash
python -m benchmarks.performance_benchmarks
python -m benchmarks.integration_benchmarks
```

## 📈 Key Metrics Monitored

- **Service Initialization**: Cold start vs Hot start time (Target: <1ms hot)
- **Tool Registry**: Lookup efficiency (Target: O(1))
- **LLM Latency**: Async parallelization speedup (Target: >30%)
- **Legion Routing**: Overhead for simple queries (Target: <5000ms)
- **Memory Optimization**: Growth per request
- **Cache Efficiency**: Hit rates for models, tools, and personas

## 🛡️ Best Practices

1. **Run before/after optimization**: Always capture baseline metrics before making performance changes.
2. **Watch costs**: E2E benchmarks make real LLM calls. Use sparingly or with smaller models.
3. **Interpret carefully**: Micro-benchmarks measure localized speedups; E2E benchmarks measure real user experience which is often dominated by network latency.
