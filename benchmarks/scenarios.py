"""
Benchmark Scenarios for Quality vs Latency Testing.

This module defines test scenarios ranging from simple factual queries to complex
multi-step reasoning tasks, providing ground truth or expectations for evaluation.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BenchmarkScenario:
    name: str
    query: str
    complexity: str  # 'simple', 'medium', 'complex'
    expected_key_points: List[str]
    category: str


# Dataset of benchmarks
SCENARIOS = [
    # --- SIMPLE (Latency Sensitive) ---
    BenchmarkScenario(
        name="Simple Calculation",
        query="What is 15 * 12?",
        complexity="simple",
        category="math",
        expected_key_points=["180"],
    ),
    BenchmarkScenario(
        name="Direct Fact",
        query="Who is the CEO of Google?",
        complexity="simple",
        category="fact",
        expected_key_points=["Sundar Pichai"],
    ),
    # --- MEDIUM (Standard RAG/Tool Usage) ---
    BenchmarkScenario(
        name="Tool Usage",
        query="What is the square root of 256 multiplied by 4?",
        complexity="medium",
        category="math_tool",
        expected_key_points=["64"],  # 16 * 4
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
    ),
    # --- COMPLEX (Legion Territory) ---
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
    ),
]


def get_scenarios(complexity: Optional[str] = None) -> List[BenchmarkScenario]:
    """Get scenarios, optionally filtered by complexity."""
    if not complexity:
        return SCENARIOS
    return [s for s in SCENARIOS if s.complexity == complexity]
