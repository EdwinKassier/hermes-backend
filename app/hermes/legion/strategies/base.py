"""Base protocol for Legion strategies."""

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class LegionStrategy(Protocol):
    """Protocol defining a Legion orchestration strategy."""

    async def generate_workers(
        self, query: str, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate worker configurations for the given query.

        Args:
            query: The user's query or task description
            context: Shared context including collected info

        Returns:
            List of worker configuration dictionaries
        """
        ...

    async def synthesize_results(
        self, original_query: str, results: Dict[str, Any], persona: str
    ) -> str:
        """
        Synthesize results from workers into a final response.

        Args:
            original_query: The original user query
            results: Dictionary of worker results
            persona: The AI persona to use

        Returns:
            Synthesized response string
        """
        ...
