"""
Benchmark Scenarios for Quality vs Latency Testing.

This module defines test scenarios ranging from simple factual queries to complex
multi-step reasoning tasks, providing ground truth or expectations for evaluation.

Includes categories:
- trivial: Greetings, yes/no, single-word answers (target: <1s)
- simple: Quick facts, basic calculations (target: <2s)
- code_simple: Basic code snippets (target: <3s)
- medium: Standard analysis, comparisons (target: <5s)
- code_medium: Multi-function code (target: <5s)
- complex: Deep analysis, multi-domain synthesis (target: <15s)
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BenchmarkScenario:
    name: str
    query: str
    complexity: (
        str  # 'trivial', 'simple', 'code_simple', 'medium', 'code_medium', 'complex'
    )
    expected_key_points: List[str]
    category: str
    target_latency_ms: int = 5000  # Target latency in milliseconds


# Dataset of benchmarks
SCENARIOS = [
    # --- TRIVIAL (Should be instant, <1s) ---
    BenchmarkScenario(
        name="Greeting",
        query="Hello!",
        complexity="trivial",
        category="greeting",
        expected_key_points=["Hello", "Hi", "Hey"],
        target_latency_ms=1000,
    ),
    BenchmarkScenario(
        name="Yes/No Question",
        query="Is Python a programming language?",
        complexity="trivial",
        category="yes_no",
        expected_key_points=["Yes"],
        target_latency_ms=1000,
    ),
    BenchmarkScenario(
        name="Single Word Answer",
        query="What color is the sky on a clear day?",
        complexity="trivial",
        category="fact_simple",
        expected_key_points=["Blue"],
        target_latency_ms=1000,
    ),
    # --- SIMPLE (Quick facts, basic math, <2s) ---
    BenchmarkScenario(
        name="Simple Calculation",
        query="What is 15 * 12?",
        complexity="simple",
        category="math",
        expected_key_points=["180"],
        target_latency_ms=2000,
    ),
    BenchmarkScenario(
        name="Direct Fact",
        query="Who is the CEO of Google?",
        complexity="simple",
        category="fact",
        expected_key_points=["Sundar Pichai"],
        target_latency_ms=2000,
    ),
    BenchmarkScenario(
        name="Definition",
        query="What is an API?",
        complexity="simple",
        category="definition",
        expected_key_points=["Application Programming Interface"],
        target_latency_ms=2500,
    ),
    # --- CODE SIMPLE (Basic snippets, <3s) ---
    BenchmarkScenario(
        name="Hello World JS",
        query="Give me a basic JavaScript hello world",
        complexity="code_simple",
        category="code_js",
        expected_key_points=["console.log", "Hello"],
        target_latency_ms=3000,
    ),
    BenchmarkScenario(
        name="Hello World Python",
        query="Write a Python hello world",
        complexity="code_simple",
        category="code_python",
        expected_key_points=["print", "Hello"],
        target_latency_ms=3000,
    ),
    BenchmarkScenario(
        name="Simple Function",
        query="Write a JavaScript function that adds two numbers",
        complexity="code_simple",
        category="code_js",
        expected_key_points=["function", "return", "+"],
        target_latency_ms=3000,
    ),
    BenchmarkScenario(
        name="Array Sum",
        query="Write a Python function to sum a list of numbers",
        complexity="code_simple",
        category="code_python",
        expected_key_points=["def", "sum", "return"],
        target_latency_ms=3000,
    ),
    # --- MEDIUM (Standard RAG/Tool Usage, <5s) ---
    BenchmarkScenario(
        name="Tool Usage",
        query="What is the square root of 256 multiplied by 4?",
        complexity="medium",
        category="math_tool",
        expected_key_points=["64"],  # 16 * 4
        target_latency_ms=5000,
    ),
    BenchmarkScenario(
        name="Tech Comparison",
        query="Briefly compare Python and Rust for systems programming.",
        complexity="medium",
        category="analysis",
        expected_key_points=[
            "Python: slower, interpreted, garbage collected",
            "Rust: faster, compiled, ownership model (memory safety)",
            "Python: high level ease of use",
            "Rust: steep learning curve",
        ],
        target_latency_ms=5000,
    ),
    BenchmarkScenario(
        name="Explain Concept",
        query="Explain what a REST API is in 3 sentences.",
        complexity="medium",
        category="explanation",
        expected_key_points=["HTTP", "client-server", "stateless", "resources"],
        target_latency_ms=4000,
    ),
    # --- CODE MEDIUM (Multi-function code, <8s) ---
    BenchmarkScenario(
        name="Fibonacci Function",
        query="Write a JavaScript function that returns the nth Fibonacci number",
        complexity="code_medium",
        category="code_js",
        expected_key_points=["function", "fibonacci", "return", "recursive or loop"],
        target_latency_ms=5000,
    ),
    BenchmarkScenario(
        name="Sort Algorithm",
        query="Implement bubble sort in Python",
        complexity="code_medium",
        category="code_python",
        expected_key_points=["def", "for", "swap", "sorted"],
        target_latency_ms=5000,
    ),
    BenchmarkScenario(
        name="API Handler",
        query="Write an Express.js route handler that accepts POST data and returns JSON",
        complexity="code_medium",
        category="code_js",
        expected_key_points=["app.post", "req.body", "res.json"],
        target_latency_ms=6000,
    ),
    # --- COMPLEX (Legion Territory, <15s) ---
    BenchmarkScenario(
        name="Complex Reasoning",
        query="Analyze the potential impact of quantum computing on modern cryptography. Recommend 3 steps companies should take to prepare.",
        complexity="complex",
        category="deep_analysis",
        expected_key_points=[
            "Threat to RSA/ECC",
            "Shor's algorithm",
            "Post-quantum cryptography (PQC)",
            "Crypto-agility",
            "Audit current inventory",
            "Hybrid implementation",
        ],
        target_latency_ms=15000,
    ),
    BenchmarkScenario(
        name="Multi-Domain Synthesis",
        query="Explain how the discovery of CRISPR relates to the concept of 'playing god' in bioethics, citing both potential medical cures and eugenic risks.",
        complexity="complex",
        category="synthesis",
        expected_key_points=[
            "Gene editing technology",
            "Curing genetic diseases (positive)",
            "Designer babies / Germline editing (risk)",
            "Ethical responsibility",
            "Unintended consequences",
        ],
        target_latency_ms=15000,
    ),
    BenchmarkScenario(
        name="Architecture Design",
        query="Design a microservices architecture for an e-commerce platform. Include service boundaries, data flow, and key technologies.",
        complexity="complex",
        category="architecture",
        expected_key_points=[
            "Product service",
            "Order service",
            "User/Auth service",
            "API Gateway",
            "Message queue",
            "Database per service",
        ],
        target_latency_ms=20000,
    ),
]


def get_scenarios(complexity: Optional[str] = None) -> List[BenchmarkScenario]:
    """Get scenarios, optionally filtered by complexity."""
    if not complexity:
        return SCENARIOS
    return [s for s in SCENARIOS if s.complexity == complexity]


def get_scenarios_by_target_latency(max_latency_ms: int) -> List[BenchmarkScenario]:
    """Get scenarios that should complete within the specified latency."""
    return [s for s in SCENARIOS if s.target_latency_ms <= max_latency_ms]


def get_quick_scenarios() -> List[BenchmarkScenario]:
    """Get only trivial and simple scenarios for quick latency testing."""
    return [
        s for s in SCENARIOS if s.complexity in ["trivial", "simple", "code_simple"]
    ]
