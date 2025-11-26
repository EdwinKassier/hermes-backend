"""
Adaptive Synthesizer Service.

Synthesizes results from Legion workers with quality assessment and adaptive strategies.
"""

import logging
from typing import Any, Dict

from app.shared.utils.service_loader import get_async_llm_service

from ..models import QualityMetrics
from ..utils.llm_utils import extract_json_from_llm_response

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

            response = await self.llm_service.generate_async(prompt, persona="hermes")
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
            """

            response = await self.llm_service.generate_async(prompt, persona=persona)
            return response

        except Exception as e:
            logger.error(f"Error synthesizing results: {e}")
            return "I apologize, but I encountered an error synthesizing the results."
