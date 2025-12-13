"""Council strategy implementation."""

import logging
from typing import Any, Dict, List

from app.shared.utils.service_loader import get_async_llm_service

from ..parallel.result_synthesizer import ResultSynthesizer
from ..utils.llm_utils import extract_json_from_llm_response
from .base import LegionStrategy

logger = logging.getLogger(__name__)


class CouncilStrategy(LegionStrategy):
    """
    Council strategy: Generates diverse expert personas to analyze a problem
    from multiple perspectives.
    """

    def __init__(self, persona_generator=None):
        """Initialize council strategy with dependencies."""
        # Council strategy uses predefined personas, so it doesn't need persona generation
        # But we keep the parameter for consistency
        pass

    async def generate_workers(
        self, query: str, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate diverse expert personas."""
        try:
            llm_service = get_async_llm_service()
            prompt = f"""Generate 3 diverse expert personas to answer: "{query}".
            Return JSON: {{ "personas": [ {{ "name": "...", "description": "...", "perspective": "..." }} ] }}"""

            # Properly await async LLM call
            response = await llm_service.generate_async(prompt, persona="hermes")
            data = extract_json_from_llm_response(response)
            personas = data.get("personas", [])

            workers = []
            for p in personas:
                workers.append(
                    {
                        "worker_id": f"council_{p['name']}",
                        "role": "analyst",  # Use valid agent type for council members
                        "persona": p["name"],  # Use the generated council persona
                        "task_description": f"Analyze this question from the perspective of {p['name']} ({p['description']}). Focus on: {p['perspective']}.\nQuestion: {query}",
                        "tools": ["web_search"],  # Default tool for council
                    }
                )
            return workers

        except Exception as e:
            logger.error(f"Failed to generate council workers: {e}")
            # Fallback
            return [
                {
                    "worker_id": "council_optimist",
                    "role": "analyst",  # Use valid agent type
                    "persona": "optimist",  # Use council persona
                    "task_description": f"Analyze optimistically: {query}",
                    "tools": ["web_search"],
                },
                {
                    "worker_id": "council_critic",
                    "role": "analyst",  # Use valid agent type
                    "persona": "critic",  # Use council persona
                    "task_description": f"Analyze critically: {query}",
                    "tools": ["web_search"],
                },
            ]

    async def synthesize_results(
        self, original_query: str, results: Dict[str, Any], persona: str
    ) -> str:
        """Synthesize diverse perspectives."""
        # Format results for synthesizer
        formatted_results = {}
        for worker_id, data in results.items():
            formatted_results[worker_id] = {
                "agent_id": worker_id,
                "result": data["result"],
                "status": data["status"],
                "agent_type": data["role"],
            }

        synthesizer = ResultSynthesizer()
        # The ResultSynthesizer handles the prompt construction internally,
        # but we could subclass it or pass specific instructions if the API supported it.
        # For now, we use the standard synthesis which is quite robust.
        return synthesizer.synthesize_results(
            original_query=original_query,
            agent_results=formatted_results,
            persona=persona,
        )
