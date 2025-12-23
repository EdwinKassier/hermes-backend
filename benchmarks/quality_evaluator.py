"""
Quality Evaluation System using LLM-as-a-Judge.

This module provides tools to score the quality of LLM responses based on dimensions
such as relevance, completeness, accuracy, and coherence.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.shared.utils.service_loader import get_async_llm_service

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Quality scores for a response."""

    relevance: float  # 0-10
    completeness: float  # 0-10
    accuracy: float  # 0-10
    coherence: float  # 0-10
    reasoning: str

    @property
    def overall_score(self) -> float:
        """Calculate weighted overall score."""
        return (
            self.relevance * 0.3
            + self.completeness * 0.3
            + self.accuracy * 0.3
            + self.coherence * 0.1
        )


class QualityEvaluator:
    """
    Evaluates response quality using a dedicated LLM instance.
    """

    def __init__(self, model: str = "gemini-1.5-pro"):
        self.model = model
        self.llm_service = get_async_llm_service()

    async def evaluate_response(
        self, query: str, response: str, expected_answer: Optional[str] = None
    ) -> QualityScore:
        """
        Evaluate a response against a query and optional expected answer.

        Args:
            query: The user's original query
            response: The system's response to evaluate
            expected_answer: Optional ground truth or key points to check for

        Returns:
            QualityScore object with dimensional scores
        """

        ground_truth_context = ""
        if expected_answer:
            ground_truth_context = f"\nREFERENCE GROUND TRUTH:\n{expected_answer}\n"

        prompt = f"""You are an expert AI Quality Evaluator.
Role: Evaluate the quality of an AI system's response to a user query.

USER QUERY:
"{query}"

{ground_truth_context}
SYSTEM RESPONSE:
"{response}"

EVALUATION INSTRUCTIONS:
Score the response on these 4 dimensions (0-10 scale):

1. Relevance (0-10): Does it directly address the user's specific question?
   - 10: Perfectly relevant, no fluff
   - 0: Completely irrelevant

2. Completeness (0-10): Does it cover all aspects of the query?
   - 10: Comprehensive, leaves no follow-up needed
   - 0: Missing critical info

3. Accuracy (0-10): Is the information correct and factual?
   - 10: Factually perfect (based on your knowledge/ground truth)
   - 0: Contains hallucinations or major errors

4. Coherence (0-10): Is it well-structured and natural?
   - 10: Clear, professional, easy to read
   - 0: Garbled or confusing

OUTPUT FORMAT:
Return ONLY a valid JSON object:
{{
  "relevance": <float>,
  "completeness": <float>,
  "accuracy": <float>,
  "coherence": <float>,
  "reasoning": "<concise explanation of scores>"
}}
"""

        try:
            # Use 'prism' persona for impartial evaluation
            json_response = await self.llm_service.generate_async(
                prompt, persona="prism"
            )
            parsed = self._parse_json_result(json_response)

            return QualityScore(
                relevance=float(parsed.get("relevance", 0)),
                completeness=float(parsed.get("completeness", 0)),
                accuracy=float(parsed.get("accuracy", 0)),
                coherence=float(parsed.get("coherence", 0)),
                reasoning=parsed.get("reasoning", "No reasoning provided"),
            )

        except Exception as e:
            logger.error(f"Error during quality evaluation: {e}")
            # Return fail-safe score
            return QualityScore(0, 0, 0, 0, f"Evaluation failed: {e}")

    def _parse_json_result(self, text: str) -> Dict[str, Any]:
        """Extract and parse JSON from LLM response."""
        try:
            # Remove markdown code blocks if present
            clean_text = re.sub(r"```json\s*|\s*```", "", text).strip()
            # If plain code block
            clean_text = re.sub(r"```\s*|\s*```", "", clean_text).strip()

            # Simple JSON parse
            return json.loads(clean_text)

        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON evaluation: {text[:100]}...")
            # Fallback regex extraction
            return {
                "relevance": self._extract_score(text, "relevance"),
                "completeness": self._extract_score(text, "completeness"),
                "accuracy": self._extract_score(text, "accuracy"),
                "coherence": self._extract_score(text, "coherence"),
                "reasoning": "Parsed via regex fallback",
            }

    def _extract_score(self, text: str, key: str) -> float:
        """Extract a single score using regex."""
        match = re.search(rf'"{key}"\s*:\s*([\d\.]+)', text)
        if match:
            return float(match.group(1))
        return 0.0
