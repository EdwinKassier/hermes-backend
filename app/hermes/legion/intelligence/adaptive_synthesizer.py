"""
Adaptive Synthesizer Service.

Synthesizes results from Legion workers with quality assessment and adaptive strategies.
"""

import logging
from typing import Any, Dict

from app.shared.utils.service_loader import get_async_llm_service

from ..models import QualityMetrics
from ..utils.llm_utils import extract_json_from_llm_response
from ..utils.persona_context import get_current_legion_persona

logger = logging.getLogger(__name__)


class AdaptiveSynthesizer:
    """Quality-aware synthesizer for Legion orchestration."""

    def __init__(self):
        self.llm_service = get_async_llm_service()

    async def assess_result_quality(self, results: Dict[str, Any]) -> QualityMetrics:
        """
        Assess the quality of the worker results.
        """
        try:
            # Prepare results summary
            summary = []
            for worker_id, data in results.items():
                status = data.get("status", "unknown")
                result_preview = str(data.get("result", ""))[:200] + "..."
                summary.append(f"Worker {worker_id} ({status}): {result_preview}")

            summary_text = "\n".join(summary)

            prompt = f"""
            Assess the quality of these AI agent results.

            Results Summary:
            {summary_text}

            Rate the following metrics (0.0 to 1.0):
            - completeness: Do the results cover the requested tasks?
            - coherence: Are the results logical and consistent?
            - relevance: Are the results relevant to the likely user intent?
            - confidence: How confident are you in these results?
            - agreement: Do the agents agree? (If applicable)

            Return ONLY valid JSON:
            {{
                "completeness": 0.8,
                "coherence": 0.9,
                "relevance": 0.9,
                "confidence": 0.8,
                "agreement": 0.7
            }}
            """

            response = await self.llm_service.generate_async(
                prompt, persona=get_current_legion_persona()
            )
            data = extract_json_from_llm_response(response)

            return QualityMetrics(**data)

        except Exception as e:
            logger.error(f"Error assessing quality: {e}")
            return QualityMetrics(
                completeness=0.5,
                coherence=0.5,
                relevance=0.5,
                confidence=0.5,
                agreement=0.0,
            )

    async def synthesize_adaptively(
        self,
        original_query: str,
        results: Dict[str, Any],
        quality: QualityMetrics,
        strategy: str,
        persona: str,
    ) -> str:
        """
        Synthesize results using an adaptive strategy based on quality.
        """
        try:
            # Format full results
            formatted_results = []
            for worker_id, data in results.items():
                formatted_results.append(
                    f"--- Worker: {worker_id} ({data.get('role', 'unknown')}) ---\n{data.get('result', '')}"
                )

            results_text = "\n\n".join(formatted_results)

            # Determine synthesis instructions based on quality and strategy
            instructions = ""
            if quality.completeness < 0.5:
                instructions = "WARNING: Results are incomplete. Acknowledge missing information and synthesize what is available."
            elif quality.agreement < 0.4 and strategy == "council":
                instructions = "Highlight the disagreement between experts. Present the conflicting viewpoints clearly."
            else:
                instructions = "Synthesize a comprehensive and cohesive answer."

            prompt = f"""
            Synthesize the following AI agent results into a final response for the user.

            Original Query: "{original_query}"

            Agent Results:
            {results_text}

            Instructions:
            {instructions}

            Persona: {persona}

            Ensure the tone matches the persona.

            **CRITICAL OUTPUT FORMATTING REQUIREMENTS**:

            Your response MUST use proper markdown that a frontend can parse:

            1. **Code Blocks**: ALL code MUST be wrapped in fenced code blocks:
               - Use triple backticks with language: ```python, ```javascript, etc.
               - NEVER output raw code without proper fencing
               - Closing ``` must be on its own line

            2. **Structure**: Use ## headers for sections, - for bullet lists

            3. **Spacing**:
               - Include double newlines between all sections
               - Use horizontal rules (---) surrounded by blank lines for major separations
               - Ensure blank lines around lists and code blocks

            4. **Inline Code**: Use `backticks` for filenames and function names

            5. **Preserve Formatting**: Maintain markdown from agent outputs
            """

            response = await self.llm_service.generate_async(prompt, persona=persona)
            return response

        except Exception as e:
            logger.error(f"Error synthesizing results: {e}")
            return "I apologize, but I encountered an error synthesizing the results."

    async def synthesize_with_inline_quality(
        self,
        original_query: str,
        results: Dict[str, Any],
        strategy: str,
        persona: str,
    ) -> str:
        """
        Fast-path synthesis that combines quality assessment and synthesis into ONE LLM call.

        This eliminates the separate quality assessment LLM call, reducing latency by ~1-2s.
        Quality metrics are assessed inline as part of the synthesis prompt.

        Args:
            original_query: Original user query
            results: Worker results to synthesize
            strategy: Strategy name (council, parallel, intelligent)
            persona: Persona for response style

        Returns:
            Synthesized response string
        """
        try:
            # Format full results
            formatted_results = []
            for worker_id, data in results.items():
                status = data.get("status", "success")
                role = data.get("role", "unknown")
                result_text = data.get("result", "")
                formatted_results.append(
                    f"--- Worker: {worker_id} (Role: {role}, Status: {status}) ---\n{result_text}"
                )

            results_text = "\n\n".join(formatted_results)

            # Count success vs partial/failed results for inline quality assessment
            success_count = sum(
                1 for d in results.values() if d.get("status") == "success"
            )
            total_count = len(results)
            quality_hint = ""

            if success_count < total_count:
                partial_count = total_count - success_count
                quality_hint = f"\n\nNOTE: {partial_count}/{total_count} workers returned partial or degraded results. Acknowledge limitations where relevant."

            # Detect potential disagreement for council strategy
            if strategy == "council" and total_count > 1:
                quality_hint += "\n\nFor council strategy: If agents present conflicting viewpoints, highlight the disagreement and present both sides."

            prompt = f"""Synthesize these AI agent results into a comprehensive final response.

Original Query: "{original_query}"

Agent Results:
{results_text}
{quality_hint}

**Synthesis Guidelines:**
1. **Executive Summary First**: Start with a 2-3 sentence summary of the key findings
2. Integrate all relevant insights into a cohesive narrative by themes/topics
3. Remove redundancy while preserving unique insights
4. If any results are incomplete/partial, acknowledge limitations
5. Write naturally for persona: {persona}

**FORMATTING REQUIREMENTS:**
- Code blocks: Use ```language syntax with closing ``` on its own line
- Headers: Use ## for sections
- Lists: Use - for bullets with blank lines before/after
- Inline code: Use `backticks` for function/file names
- Spacing: Double newlines between major sections
- Tables: Use | Col1 | Col2 | syntax for comparisons

**ANTI-PATTERNS TO AVOID:**
- Do NOT list agent names or mention "Agent 1 said...", "Worker contributed..."
- Do NOT include meta-commentary about the synthesis process
- Do NOT use placeholder text like "[insert details here]"
- Do NOT repeat the same information in different sections

Provide a complete, well-formatted response that directly addresses the user's query."""

            response = await self.llm_service.generate_async(prompt, persona=persona)
            logger.info("Synthesized with inline quality assessment (fast-path)")
            return response

        except Exception as e:
            logger.error(f"Error in fast-path synthesis: {e}")
            return "I apologize, but I encountered an error synthesizing the results."
