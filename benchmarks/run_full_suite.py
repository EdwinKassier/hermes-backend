"""
Unified Benchmark Suite Runner.

This script executes all available benchmarks:
1. Bottleneck Analysis (Static/Dynamic analysis)
2. E2E Performance & Quality (Real API calls)
3. Agent Intelligence (Routing/Planning)
4. System Resilience (Chaos/Fuzzing)

It aggregates the results into a comprehensive master report.
"""

import asyncio
import contextlib
import datetime
import io
import os

# Import benchmark modules
# We import them to run their main logic or specific functions
# Note: Some scripts use argparse, so we might need to invoke them as subprocesses
# or careful function calls. For stability, subprocess is cleaner for capturing independent outputs.
import subprocess
import sys
from pathlib import Path

REPORT_DIR = Path("benchmarks/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


class BenchmarkRunner:
    def __init__(self):
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.master_report_path = REPORT_DIR / f"MASTER_BENCHMARK_{self.timestamp}.md"
        self.summaries = []

    def run_command(self, title: str, command: list, working_dir: str = ".") -> str:
        """Run a benchmark command and capture output."""
        print(f"\n🚀 Running: {title}...")
        try:
            result = subprocess.run(
                command,
                cwd=working_dir,
                capture_output=True,
                text=True,
                check=False,  # Don't throw on non-zero, capture it
            )

            status = "✅ PASSED" if result.returncode == 0 else "❌ FAILED"
            print(f"   Status: {status}")

            output = result.stdout
            if result.stderr:
                output += "\n\nSTDERR:\n" + result.stderr

            return {
                "title": title,
                "status": status,
                "output": output,
                "return_code": result.returncode,
            }
        except Exception as e:
            print(f"   Execution Error: {e}")
            return {
                "title": title,
                "status": "❌ ERROR",
                "output": str(e),
                "return_code": -1,
            }

    def save_master_report(self, results: list):
        """Generate and save Markdown report."""
        with open(self.master_report_path, "w") as f:
            f.write(f"# 📊 Hermes & Legion Benchmark Master Report\n")
            f.write(f"**Date:** {datetime.datetime.now().isoformat()}\n\n")

            f.write("## 1. Executive Summary\n")
            f.write("| Benchmark Suite | Status |\n")
            f.write("|---|---|\n")
            for r in results:
                f.write(f"| {r['title']} | {r['status']} |\n")

            f.write("\n---\n")

            for r in results:
                f.write(f"\n## {r['title']}\n")
                f.write(f"**Status:** {r['status']}\n\n")

                # Format output code block
                f.write("```text\n")
                # Truncate if too long? For now keep full for "full report"
                f.write(r["output"])
                f.write("\n```\n")

        print(f"\n✨ Master Report Saved: {self.master_report_path}")
        return self.master_report_path

    def run_all(self):
        results = []

        # 1. Bottleneck Analysis
        results.append(
            self.run_command(
                "Bottleneck Analysis",
                ["uv", "run", "python", "-m", "benchmarks.bottleneck_analyzer"],
            )
        )

        # 2. Agent Intelligence
        results.append(
            self.run_command(
                "Agent Intelligence (Routing & Planning)",
                ["uv", "run", "python", "-m", "benchmarks.agent_benchmarks"],
            )
        )

        # 3. Resilience (Chaos)
        results.append(
            self.run_command(
                "Resilience & Security",
                ["uv", "run", "python", "-m", "benchmarks.resilience_benchmarks"],
            )
        )

        # 4. E2E Quality (Most expensive, run last)
        results.append(
            self.run_command(
                "E2E Quality vs Latency",
                ["uv", "run", "python", "-m", "benchmarks.e2e_benchmarks", "--quality"],
            )
        )

        return self.save_master_report(results)


if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run_all()
