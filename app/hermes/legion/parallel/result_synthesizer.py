"""Result synthesis for multi-agent orchestration."""

import logging
from typing import Any, Dict, List, Optional

from app.shared.utils.service_loader import get_gemini_service

logger = logging.getLogger(__name__)


class ResultSynthesizer:
    """
    Synthesizes results from multiple agents into a coherent response.

    Combines outputs from parallel agent execution, detects gaps,
    and generates clarifying questions when needed.
    """

    def __init__(self):
        """Initialize result synthesizer."""
        self._gemini_service = None

    @property
    def gemini_service(self):
        """Lazy load Gemini service."""
        if self._gemini_service is None:
            self._gemini_service = get_gemini_service()
        return self._gemini_service

    def synthesize_results(
        self,
        original_query: str,
        agent_results: Dict[str, Dict[str, Any]],
        persona: str = "hermes",
    ) -> str:
        """
        Combine results from multiple agents into coherent response.

        Args:
            original_query: User's original question
            agent_results: Dict of agent_id -> {agent_type, result, task_description}
            persona: AI persona to use

        Returns:
            Synthesized response combining all agent outputs
        """
        if not agent_results:
            return "I couldn't generate a response from the agents."

        if len(agent_results) == 1:
            # Single agent result, return directly
            result = list(agent_results.values())[0]
            return result.get("result", "No result available")

        logger.info(f"Synthesizing results from {len(agent_results)} agents")

        # Build context from all agent results
        results_by_type = {}
        for agent_id, result_data in agent_results.items():
            agent_type = result_data.get("agent_type", "unknown")
            task_desc = result_data.get("task_description", "")
            result = result_data.get("result", "")
            status = result_data.get("status", "unknown")

            if agent_type not in results_by_type:
                results_by_type[agent_type] = []

            results_by_type[agent_type].append(
                {"task": task_desc, "result": result, "status": status}
            )

        # Build structured context
        results_context = []
        for agent_type, type_results in results_by_type.items():
            for item in type_results:
                status_emoji = "✅" if item["status"] == "success" else "⚠️"
                results_context.append(
                    f"{status_emoji} **{agent_type.upper()}** (Task: {item['task']}):\n{item['result']}"
                )

        combined_context = "\n\n---\n\n".join(results_context)

        # Enhanced synthesis prompt
        prompt = f"""You are synthesizing results from multiple specialized AI agents into a single, comprehensive response.

ORIGINAL USER QUESTION: "{original_query}"

AGENT RESULTS:
{combined_context}

YOUR TASK:
1. **Integrate All Perspectives**: Combine insights from all agents into one cohesive narrative
2. **Maintain Structure**: Organize information logically based on the original question
3. **Remove Redundancy**: Eliminate repetitive information while preserving unique insights
4. **Add Transitions**: Use smooth transitions between different agent contributions
5. **Stay Focused**: Directly answer the user's original question
6. **Be Comprehensive**: Include all relevant information from each agent
7. **Natural Tone**: Write conversationally, not like a concatenated document

FORMATTING GUIDELINES:
- Use clear section headings when appropriate
- Employ bullet points for lists
- Include code blocks if code was generated
- Maintain markdown formatting from agent outputs

Provide a well-structured, comprehensive response that naturally weaves together all agent contributions while directly addressing "{original_query}"."""

        try:
            synthesized = self.gemini_service.generate_gemini_response(
                prompt, persona=persona
            )
            return synthesized

        except Exception as e:
            logger.error(f"Result synthesis failed: {e}")
            # Fallback: enhanced concatenation
            return self._enhanced_concatenation(original_query, agent_results)

    def _enhanced_concatenation(
        self, original_query: str, agent_results: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Enhanced fallback synthesis with better formatting.

        Args:
            original_query: User's question
            agent_results: Agent outputs

        Returns:
            Well-formatted concatenated results
        """
        parts = [f"# Response to: {original_query}\n"]

        # Group by agent type
        by_type = {}
        for agent_id, result_data in agent_results.items():
            agent_type = result_data.get("agent_type", "Agent")
            if agent_type not in by_type:
                by_type[agent_type] = []
            by_type[agent_type].append(result_data)

        # Format each type
        for agent_type, results in by_type.items():
            parts.append(f"\n## {agent_type.title()} Findings\n")
            for result_data in results:
                result = result_data.get("result", "")
                if result:
                    parts.append(result)
                    parts.append("\n")

        return "\n".join(parts)

    def detect_gaps(
        self, agent_results: Dict[str, Dict[str, Any]], expected_agents: List[str]
    ) -> List[str]:
        """
        Detect missing or incomplete agent results.

        Args:
            agent_results: Actual results received
            expected_agents: Agents that were supposed to execute

        Returns:
            List of gap descriptions
        """
        gaps = []

        # Check for missing agents
        result_agent_ids = set(agent_results.keys())
        expected_agent_ids = set(expected_agents)
        missing = expected_agent_ids - result_agent_ids

        for agent_id in missing:
            gaps.append(f"Missing result from {agent_id}")

        # Check for empty results
        for agent_id, result_data in agent_results.items():
            result = result_data.get("result", "")
            if not result or len(result.strip()) < 10:
                gaps.append(f"Incomplete result from {agent_id}")

        return gaps

    def generate_clarifying_questions(
        self,
        gaps: List[str],
        original_query: str,
        partial_results: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """
        Generate clarifying questions based on gaps in agent results.

        Args:
            gaps: List of detected gaps
            original_query: User's question
            partial_results: Results that were received

        Returns:
            List of clarifying questions to ask user
        """
        if not gaps:
            return []

        # Simple question generation based on gaps
        questions = []

        for gap in gaps:
            if "code" in gap.lower():
                questions.append("What programming language should I use for the code?")
            elif "analysis" in gap.lower():
                questions.append("What specific aspects should I analyze?")
            elif "research" in gap.lower():
                questions.append("What depth of research do you need?")

        # Remove duplicates
        return list(set(questions))
