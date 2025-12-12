"""Research agent for conducting research tasks."""

import logging
from typing import Dict, List

from app.shared.utils.service_loader import get_gemini_service

from ..models import RequiredInfoField, SubAgentState
from .base import BaseSubAgent

logger = logging.getLogger(__name__)


class ResearchAgent(BaseSubAgent):
    """Agent for conducting research tasks."""

    @property
    def agent_id(self) -> str:
        """Unique identifier for research agent."""
        return "research_agent"

    @property
    def task_types(self) -> List[str]:
        """Task types this agent can handle."""
        return ["research", "investigation", "analysis"]

    def identify_required_info(
        self, task: str, user_message: str
    ) -> Dict[str, RequiredInfoField]:
        """
        Intelligently determine if any information is genuinely missing.

        Uses LLM to analyze the request and conversation context to infer
        missing details. Only returns required fields when information cannot
        be reasonably inferred.
        """
        try:
            gemini_service = get_gemini_service()

            analysis_prompt = f"""Analyze this research request to determine if any critical information is genuinely missing and cannot be reasonably inferred.

User Request: "{user_message}"
Task: "{task}"

**Your Goal**: Determine what information (if any) is truly needed from the user.

**Inference Philosophy**:
- **Strongly prefer inference over asking questions**
- Use domain knowledge and context to make reasonable assumptions
- Only flag information as "required" if it's genuinely ambiguous or critical

**Potential Information to Consider**:
1. **Time Period**: Can you infer from context? (e.g., "trends" = recent, "history" = comprehensive)
2. **Topics/Focus Areas**: Can you infer from the research subject?
3. **Depth**: Can you infer from query complexity? (brief/moderate/comprehensive)

**When to Request Information**:
✅ Request if: Genuinely ambiguous (e.g., "research it" with no context)
✅ Request if: Contradictory requirements
✅ Request if: Sensitive/critical parameter that must be explicit
❌ Do NOT request if: Can reasonably infer from context
❌ Do NOT request if: Standard defaults apply

**Response Format (JSON)**:
If NO information is needed (you can infer everything):
{{
  "needs_info": false,
  "inferred_values": {{
    "time_period": "<what you'll assume>",
    "topics": "<what you'll focus on>",
    "depth": "<brief|moderate|comprehensive>"
  }},
  "reasoning": "<why you can proceed without asking>"
}}

If information IS needed:
{{
  "needs_info": true,
  "required_fields": [
    {{
      "field_name": "...",
      "field_type": "string|enum|list",
      "question": "...",
      "description": "...",
      "options": ["..."] // only for enum type
    }}
  ],
  "reasoning": "<why you cannot infer this information>"
}}

Analyze and respond with JSON only."""

            # Call LLM directly - already wrapped in try/except with fail-open
            response = gemini_service.generate_gemini_response(
                prompt=analysis_prompt,
                persona=self.persona,
            )

            import json
            import re

            # Extract JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group(0))

                # If no info needed, return empty dict
                if not analysis.get("needs_info", False):
                    logger.info(
                        f"Research agent can proceed without clarification. "
                        f"Inferred: {analysis.get('inferred_values', {})}"
                    )
                    return {}

                # Convert required_fields to RequiredInfoField objects
                required_info = {}
                for field_spec in analysis.get("required_fields", []):
                    field_name = field_spec["field_name"]
                    required_info[field_name] = RequiredInfoField(
                        field_name=field_name,
                        field_type=field_spec["field_type"],
                        question=field_spec["question"],
                        description=field_spec.get("description", ""),
                        options=field_spec.get("options"),
                    )

                logger.info(
                    f"Research agent needs clarification: {list(required_info.keys())}"
                )
                return required_info

        except Exception as e:
            logger.warning(
                f"Failed to analyze info requirements: {e}. Proceeding without clarification."
            )

        # On error, proceed without asking (fail open, not closed)
        return {}

    def execute_task(self, state: SubAgentState) -> str:
        """
        Execute deep research task using multi-phase approach.

        Phase 1: Search for relevant sources (10-15 sources)
        Phase 2: Scrape 7-10 top sources for detailed content
        Phase 3: Synthesize comprehensive research report

        Args:
            state: SubAgentState with task and collected information

        Returns:
            Comprehensive research result with citations
        """
        try:
            # Get GeminiService
            gemini_service = get_gemini_service()
            user_id = state.metadata.get("user_id", "default")

            # Build research context
            time_period = state.collected_info.get("time_period", "all time")
            topics = state.collected_info.get("topics", [])
            depth = state.collected_info.get("depth", "moderate")
            topics_str = ", ".join(topics) if isinstance(topics, list) else str(topics)

            # Use circuit breaker for LLM calls
            from ..utils.resilience import get_llm_circuit_breaker

            circuit_breaker = get_llm_circuit_breaker()

            # Incorporate Judge Feedback
            feedback_context = ""
            if state.judge_feedback:
                logger.info(f"Retrying with judge feedback: {state.judge_feedback}")
                feedback_context = f"""
**CRITICAL FEEDBACK FROM PREVIOUS ATTEMPT**:
The previous attempt was rejected. You MUST address the following feedback:
{state.judge_feedback}
"""

            # PHASE 1: Search for sources
            logger.info(f"Phase 1: Searching for sources on '{state.task}'")
            search_prompt = f"""**SYSTEM OVERRIDE**: You are a General Researcher with access to web search tools.

Your task: Find relevant sources for researching: {state.task}
{feedback_context}

Time period: {time_period}
Specific topics: {topics_str}

Use the 'firecrawl' tool in SEARCH mode to find 10-15 high-quality sources. Look for:
- Authoritative sources (official sites, academic papers, reputable news)
- Recent information (especially for current topics)
- Diverse perspectives
- Technical documentation and expert analyses

After searching, list the TOP 7-10 most relevant URLs you found that should be read in detail.
Prioritize sources that will provide:
1. Comprehensive coverage of the topic
2. Specific data, statistics, and technical details
3. Multiple viewpoints or approaches
4. Recent developments and current state"""

            search_result = circuit_breaker.call(
                gemini_service.generate_gemini_response,
                prompt=search_prompt,
                user_id=user_id,
                persona=self.persona,
            )

            logger.info(f"Phase 1 complete. Search results: {len(search_result)} chars")

            # PHASE 2: Scrape top sources for detailed content
            logger.info("Phase 2: Scraping 7-10 top sources for detailed content")
            scrape_prompt = f"""**SYSTEM OVERRIDE**: You are a General Researcher with access to web scraping tools.

Based on your previous search, you identified these sources:
{search_result}

Now use the 'firecrawl' tool in SCRAPE mode to read the full content of the TOP 7-10 most relevant URLs.

IMPORTANT: You MUST scrape AT LEAST 7 URLs to ensure comprehensive coverage. Scrape up to 10 if available.

For each URL:
1. Call firecrawl with mode="scrape" and url="<the url>"
2. Read the FULL content returned
3. Extract key information, facts, data points, quotes, and technical details
4. Note the source URL for citation purposes

After scraping all sources, provide a detailed summary of what you learned from EACH source, organized by URL."""

            scrape_result = circuit_breaker.call(
                gemini_service.generate_gemini_response,
                prompt=scrape_prompt,
                user_id=user_id,
                persona=self.persona,
            )

            logger.info(
                f"Phase 2 complete. Scraped content: {len(scrape_result)} chars"
            )

            # PHASE 3: Synthesize comprehensive research report
            logger.info("Phase 3: Synthesizing comprehensive research report")
            synthesis_prompt = f"""**SYSTEM OVERRIDE**: You are a General Researcher creating a comprehensive research report.

RESEARCH TOPIC: {state.task}
{feedback_context}
DEPTH REQUIRED: {depth}

YOU HAVE COMPLETED:
1. Initial search identifying 10-15 sources
2. Deep reading of 7-10 full articles/sources

SEARCH RESULTS:
{search_result}

DETAILED CONTENT FROM 7-10 SCRAPED SOURCES:
{scrape_result}

**CRITICAL OUTPUT REQUIREMENTS**:
1. **Format**: Use Markdown with clear headers (##, ###), bullet points, and bold text for key terms.
2. **Citations**: You MUST include inline citations for EVERY fact using [Source Name](url) format.
3. **Detail**: Provide specific numbers, dates, names, and quotes from the sources you read.
4. **Depth**: This should be a COMPREHENSIVE report leveraging ALL 7-10 sources. Include:
   - Specific data points and statistics from multiple sources
   - Direct quotes or paraphrases with proper attribution
   - Multiple perspectives and approaches
   - Technical details and expert insights
   - Comparative analysis where relevant

**CRITICAL OUTPUT FORMATTING REQUIREMENTS**:
- Use proper markdown that a frontend can parse
- Separate ALL major sections with double newlines (\\n\\n)
- Use horizontal rules (---) surrounded by blank lines to separate topics
- Break long text into short, readable paragraphs
- Ensure blank lines around lists and headers

5. **Structure**:
   - **Executive Summary**: High-level overview (2-3 paragraphs)
   - **Key Findings**: 8-12 detailed bullet points with citations from different sources
   - **Detailed Analysis**: In-depth explanation organized by subtopics (4-6 paragraphs minimum)
     * Use information from ALL scraped sources
     * Compare and contrast different perspectives
     * Include technical details and data
   - **Conclusion**: Synthesis of findings across all sources
   - **Sources**: Complete numbered list of all 7-10+ URLs cited

Create a comprehensive, well-researched report (2000+ words) that demonstrates deep understanding by synthesizing information from ALL sources you read."""

            final_result = circuit_breaker.call(
                gemini_service.generate_gemini_response,
                prompt=synthesis_prompt,
                user_id=user_id,
                persona=self.persona,
            )

            logger.info(f"Phase 3 complete. Final report: {len(final_result)} chars")
            return final_result

        except Exception as e:
            logger.error("Research agent execution failed: %s", e)
            raise
