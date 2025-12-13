"""Parallel task decomposition for multi-agent orchestration."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.shared.utils.service_loader import get_gemini_service

from ..utils.persona_context import get_current_legion_persona

logger = logging.getLogger(__name__)


class ParallelTaskDecomposer:
    """
    Decomposes complex queries into parallel subtasks for multi-agent execution.

    Detects queries that require multiple specialized agents and breaks them
    into independent tasks that can be executed concurrently.
    """

    def __init__(self):
        """Initialize task decomposer."""
        self._gemini_service = None

    @property
    def gemini_service(self):
        """Lazy load Gemini service."""
        if self._gemini_service is None:
            self._gemini_service = get_gemini_service()
        return self._gemini_service

    def is_multi_agent_task(self, user_message: str) -> bool:
        """
        Determine if a query requires multiple agents using AI.

        Args:
            user_message: User's query

        Returns:
            True if multiple agents needed
        """
        # Validate input
        if not isinstance(user_message, str):
            logger.warning(f"Invalid input type: {type(user_message)}")
            return False

        user_message = user_message.strip()
        if not user_message or len(user_message) < 5:
            return False

        # Limit length to prevent excessive API costs
        if len(user_message) > 1000:
            user_message = user_message[:1000]

        # Use cached version
        return self._cached_multi_agent_check(user_message)

    def _cached_multi_agent_check(self, user_message: str) -> bool:
        """Cached multi-agent detection (LRU cache)."""
        try:
            # Use AI to detect multi-agent tasks
            prompt = self._build_multi_agent_detection_prompt(user_message)
            response = self.gemini_service.generate_gemini_response(
                prompt, persona=get_current_legion_persona()
            )

            # Parse AI response
            is_multi = "MULTI_AGENT" in response.upper()

            logger.info(
                f"AI multi-agent detection: {is_multi} for query: '{user_message[:60]}...'"
            )
            return is_multi

        except (ConnectionError, TimeoutError) as e:
            # Transient network errors - retry once
            logger.warning(f"Network error in AI detection, retrying: {e}")
            try:
                response = self.gemini_service.generate_gemini_response(
                    prompt, persona=get_current_legion_persona()
                )
                is_multi = "MULTI_AGENT" in response.upper()
                logger.info(f"AI retry successful: {is_multi}")
                return is_multi
            except Exception as retry_error:
                logger.error(f"AI retry failed: {retry_error}, using fallback")
                return self._fallback_multi_agent_detection(user_message)

        except (ValueError, KeyError) as e:
            # Response parsing errors
            logger.error(f"AI response parsing failed: {e}, using fallback")
            return self._fallback_multi_agent_detection(user_message)

        except Exception as e:
            # Unexpected errors
            logger.exception(f"Unexpected AI error: {e}, using fallback")
            return self._fallback_multi_agent_detection(user_message)

    def _build_multi_agent_detection_prompt(self, user_message: str) -> str:
        """Build AI prompt for multi-agent task detection."""
        return f"""Analyze if this request requires multiple specialized agents working in parallel.

User request: "{user_message}"

A MULTI_AGENT task has INDEPENDENT subtasks that can be done in parallel by different specialists.

Examples of MULTI_AGENT tasks:
✓ "Research quantum computing AND analyze its market applications" → research + analysis agents
✓ "Find data sources, clean the data, then visualize trends" → data + analysis + code agents
✓ "Investigate AI trends and write a technical report" → research + code agents
✓ "Compare Python and Java, then recommend one" → research + analysis agents

Examples of SINGLE_AGENT tasks:
✗ "Research quantum computing" → just research agent
✗ "Write code to sort a list" → just code agent
✗ "Research and development team structure" → asking about R&D, not requesting both tasks
✗ "Find and summarize AI papers" → single research task with summarization

Respond with ONLY: "MULTI_AGENT" or "SINGLE_AGENT"
"""

    def _fallback_multi_agent_detection(self, user_message: str) -> bool:
        """
        Fallback heuristic for multi-agent detection if AI fails.

        This is the original keyword-based approach, kept as a safety net.
        """
        user_message_lower = user_message.lower()

        # Detect compound task indicators
        compound_indicators = [
            " and ",
            ", and ",
            " then ",
            " also ",
            " along with ",
            " as well as ",
            " plus ",
        ]

        # Must have compound indicator
        has_compound = any(ind in user_message_lower for ind in compound_indicators)
        if not has_compound:
            return False

        # Count distinct action verbs
        action_verbs = [
            "research",
            "investigate",
            "find",
            "search",
            "analyze",
            "evaluate",
            "assess",
            "compare",
            "write",
            "code",
            "generate",
            "create",
            "implement",
            "build",
            "process",
            "transform",
            "convert",
            "extract",
        ]

        verb_count = sum(1 for verb in action_verbs if verb in user_message_lower)

        # Multiple verbs + compound indicator = multi-agent task
        return verb_count >= 2

    def decompose_task(
        self, user_message: str, skip_check: bool = False
    ) -> Optional[List[Dict[str, str]]]:
        """
        Decompose query into parallel subtasks with agent assignments.

        Args:
            user_message: User's query
            skip_check: If True, skip the is_multi_agent_task check

        Returns:
            List of subtasks or None
        """
        if not skip_check and not self.is_multi_agent_task(user_message):
            return None

        logger.info(f"Decomposing multi-agent task: {user_message[:100]}")

        # Use AI to decompose the task
        prompt = f"""Analyze this user query and break it into independent subtasks that can be executed in parallel.

User query: "{user_message}"

For each subtask, identify:
1. A clear task description
2. The agent type needed (research, code, analysis, or data)
3. Key information needed

Respond in this exact JSON format:
{{
  "subtasks": [
    {{
      "description": "task description",
      "agent_type": "research|code|analysis|data",
      "keywords": ["key", "terms"]
    }}
  ]
}}

Agent types:
- research: Finding information, investigating topics
- code: Writing code, implementing algorithms
- analysis: Analyzing data, evaluating trends
- data: Processing, transforming, extracting data

Respond with ONLY the JSON, no other text."""

        try:
            response = self.gemini_service.generate_gemini_response(
                prompt, persona=get_current_legion_persona()
            )

            # Parse JSON response
            # Parse JSON response
            import json

            # Clean response of markdown code blocks
            cleaned_response = response.strip()
            if cleaned_response.startswith("```"):
                # Extract content between code fences
                match = re.search(
                    r"```(?:json)?\s*(\{.*?\})\s*```", cleaned_response, re.DOTALL
                )
                if match:
                    cleaned_response = match.group(1)
                else:
                    # Try to find JSON without fences
                    cleaned_response = re.sub(r"```(?:json)?", "", cleaned_response)
                    cleaned_response = cleaned_response.strip()

            # Extract JSON from response
            json_start = cleaned_response.find("{")
            json_end = cleaned_response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = cleaned_response[json_start:json_end]
                result = json.loads(json_str)
                subtasks = result.get("subtasks", [])

                if subtasks:
                    logger.info(f"Decomposed into {len(subtasks)} parallel subtasks")
                    return subtasks

            logger.warning(
                f"Could not parse decomposition result from: {response[:100]}..."
            )
            return None

        except Exception as e:
            logger.error(f"Task decomposition failed: {e}")
            return None

    def merge_with_dependencies(
        self, subtasks: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        DEPRECATED: Use analyze_task_dependencies() for proper dependency detection.

        This method is kept for backwards compatibility but now delegates to
        the synchronous fallback of dependency detection.

        Args:
            subtasks: List of decomposed subtasks

        Returns:
            Tuple of (parallel_tasks, sequential_tasks)
        """
        # Use simple heuristic for sync context
        parallel_tasks = []
        sequential_tasks = []

        for i, task in enumerate(subtasks):
            desc_lower = task.get("description", "").lower()

            # Detect sequential indicators
            sequential_indicators = [
                "then ",
                "after ",
                "based on ",
                "using the ",
                "with the results",
                "from the previous",
                "analyze the ",  # Often depends on data gathering
            ]

            is_sequential = any(ind in desc_lower for ind in sequential_indicators)

            if is_sequential and parallel_tasks:
                # This task depends on previous, make it sequential
                sequential_tasks.append(task)
            else:
                parallel_tasks.append(task)

        return parallel_tasks, sequential_tasks

    async def analyze_task_dependencies(
        self, subtasks: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Analyze task dependencies using AI-powered detection.

        This is the recommended method for dependency-aware task execution.

        Args:
            subtasks: List of decomposed subtasks

        Returns:
            Dictionary with execution_levels, tasks, and dependency information
        """
        from .task_dependencies import analyze_and_structure_tasks

        return await analyze_and_structure_tasks(subtasks)
