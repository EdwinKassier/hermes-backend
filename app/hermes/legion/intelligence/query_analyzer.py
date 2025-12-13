"""
Query Analyzer Service.

Analyzes user queries to determine complexity, required domains, and constraints.
This service powers the intelligent planning of the Legion swarm.
"""

import json
import logging
from typing import Any, Dict, List

from app.shared.utils.service_loader import get_async_llm_service

from ..models import Domain, QueryComplexity
from ..utils.llm_utils import extract_json_from_llm_response
from ..utils.persona_context import get_current_legion_persona

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """Analyzes user queries for Legion orchestration."""

    def __init__(self):
        self.llm_service = get_async_llm_service()

    async def analyze_complexity(self, query: str) -> QueryComplexity:
        """
        Analyze the complexity of a query to determine resource allocation.

        Returns a QueryComplexity object with score, dimensions, and suggestions.
        """
        try:
            prompt = f"""
            Analyze the complexity of the following user query for an AI agent swarm.

            Query: "{query}"

            Determine:
            1. Complexity Score (0.0 to 1.0): How difficult is this task?
               - 0.0-0.3: Simple factual question or basic task
               - 0.4-0.7: Multi-step reasoning or research required
               - 0.8-1.0: Complex, open-ended, or requires deep expertise/synthesis

            2. Dimensions: Rate complexity in these areas (0.0 to 1.0):
               - technical: Requires coding or technical knowledge
               - creative: Requires generation of new content
               - reasoning: Requires logical deduction or planning
               - context: Depends heavily on external context

            3. Suggested Workers: How many parallel agents are optimal? (1-5)

            4. Estimated Time: Expected execution time in seconds (approximate).

            Return ONLY valid JSON in this format:
            {{
                "score": 0.5,
                "dimensions": {{
                    "technical": 0.2,
                    "creative": 0.8,
                    "reasoning": 0.4,
                    "context": 0.1
                }},
                "suggested_workers": 2,
                "estimated_time_seconds": 30.0
            }}
            """

            # Properly await async LLM call
            response = await self.llm_service.generate_async(
                prompt, persona=get_current_legion_persona()
            )
            data = extract_json_from_llm_response(response)

            return QueryComplexity(**data)

        except Exception as e:
            logger.error(f"Error analyzing query complexity: {e}")
            # Return safe default
            return QueryComplexity(
                score=0.5,
                dimensions={
                    "technical": 0.5,
                    "creative": 0.5,
                    "reasoning": 0.5,
                    "context": 0.5,
                },
                suggested_workers=2,
                estimated_time_seconds=30.0,
            )

    async def identify_domains(self, query: str) -> List[Domain]:
        """
        Identify the knowledge domains required to answer the query.
        """
        try:
            prompt = f"""
            Identify the primary knowledge domains required for this query.

            Query: "{query}"

            Available Domains:
            - research: Searching for information, facts, history
            - coding: Writing code, debugging, technical implementation
            - data_analysis: Analyzing numbers, trends, statistics
            - creative: Writing stories, poems, marketing copy, design
            - planning: Project management, scheduling, step-by-step plans
            - general: General conversation or simple tasks

            Return ONLY valid JSON with a list of domains:
            {{
                "domains": ["research", "data_analysis"]
            }}
            """

            response = await self.llm_service.generate_async(
                prompt, persona=get_current_legion_persona()
            )
            data = extract_json_from_llm_response(response)

            domain_strs = data.get("domains", [])
            domains = []
            for d in domain_strs:
                try:
                    domains.append(Domain(d.lower()))
                except ValueError:
                    logger.warning(f"Unknown domain identified: {d}")

            if not domains:
                domains = [Domain.GENERAL]

            return domains

        except Exception as e:
            logger.error(f"Error identifying domains: {e}")
            return [Domain.GENERAL]
