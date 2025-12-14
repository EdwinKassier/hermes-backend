"""Council strategy implementation with dynamic agents."""

import logging
from typing import Any, Dict, List

from app.shared.utils.service_loader import get_async_llm_service

from ..agents.task_agent_planner import TaskAgentPlanner
from ..parallel.result_synthesizer import ResultSynthesizer
from ..utils.llm_utils import extract_json_from_llm_response
from .base import LegionStrategy

logger = logging.getLogger(__name__)


class CouncilStrategy(LegionStrategy):
    """
    Council strategy: Generates diverse expert personas to analyze a problem
    from multiple perspectives.
    """

    def __init__(self):
        """Initialize council strategy with dependencies."""
        # Council strategy uses predefined personas, so it doesn't need persona generation
        # But we keep the parameter for consistency
        pass

    async def generate_workers(
        self, query: str, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate diverse dynamic expert agents for council analysis."""
        try:
            # Generate diverse council perspectives
            llm_service = get_async_llm_service()
            prompt = f"""Generate 3 diverse expert perspectives to analyze: "{query}".

For each expert, specify:
- name: A descriptive expert title (e.g., "Economic Analyst", "Technical Architect")
- expertise: Their specific area of knowledge
- perspective: Their unique viewpoint or methodology
- agent_type: A custom agent type name for this expert

Return JSON: {{
  "experts": [
    {{
      "name": "expert_title",
      "expertise": "specific_knowledge_area",
      "perspective": "unique_viewpoint",
      "agent_type": "custom_agent_type_name"
    }}
  ]
}}"""

            response = await llm_service.generate_async(prompt, persona="hermes")
            data = extract_json_from_llm_response(response)
            experts = data.get("experts", [])

            if not experts:
                # Fallback expert generation
                experts = [
                    {
                        "name": "Critical Analyst",
                        "expertise": "systematic evaluation",
                        "perspective": "identify risks and limitations",
                        "agent_type": "critical_evaluation_specialist",
                    },
                    {
                        "name": "Creative Innovator",
                        "expertise": "novel solutions",
                        "perspective": "explore unconventional approaches",
                        "agent_type": "innovative_problem_solver",
                    },
                    {
                        "name": "Practical Implementer",
                        "expertise": "real-world application",
                        "perspective": "focus on feasibility and execution",
                        "agent_type": "implementation_practitioner",
                    },
                ]

            # Create dynamic agents for each expert
            workers = []
            for expert in experts:
                # Create dynamic agent configuration for this expert
                agent_config = self._create_council_agent_config(expert, query)

                workers.append(
                    {
                        "worker_id": f"council_{expert['name'].lower().replace(' ', '_')}",
                        "role": expert["agent_type"],
                        "task_description": f"Analyze '{query}' from the perspective of {expert['name']}. "
                        f"Expertise: {expert['expertise']}. "
                        f"Focus: {expert['perspective']}.",
                        "tools": [
                            "web_search",
                            "analysis",
                        ],  # Council members get analysis tools
                        "execution_level": 0,  # Council runs in parallel
                        "dependencies": [],
                        "dynamic_agent_config": agent_config,
                    }
                )

            logger.info(f"Generated {len(workers)} council members with dynamic agents")
            return workers

        except Exception as e:
            logger.error(f"Council strategy failed: {e}")
            # Fallback to single dynamic agent
            fallback_config = self._create_council_agent_config(
                {
                    "name": "General Analyst",
                    "expertise": "comprehensive analysis",
                    "perspective": "balanced evaluation",
                    "agent_type": "general_council_analyst",
                },
                query,
            )

            return [
                {
                    "worker_id": "council_fallback",
                    "role": "general_council_analyst",
                    "task_description": query,
                    "tools": [],
                    "execution_level": 0,
                    "dependencies": [],
                    "dynamic_agent_config": fallback_config,
                }
            ]

    def _create_council_agent_config(
        self, expert: Dict[str, Any], task: str
    ) -> Dict[str, Any]:
        """Create dynamic agent configuration for a council expert."""
        return {
            "agent_id": f"council_{expert['name'].lower().replace(' ', '_')}",
            "agent_type": expert["agent_type"],
            "task_types": ["analysis", "evaluation", "perspective"],
            "capabilities": {
                "primary_focus": f"providing {expert['expertise']} analysis from {expert['perspective']} perspective",
                "tools_needed": ["analysis", "reasoning", "evaluation"],
                "expertise_level": "expert",
                "specializations": [expert["expertise"], "critical_thinking"],
                "knowledge_domains": [expert["expertise"], "methodology"],
            },
            "prompts": {
                "identify_required_info": f"""As a {expert['name']} with expertise in {expert['expertise']}, analyze what information is needed.

Task: "{{task}}"
User Message: "{{user_message}}"

From your {expert['perspective']} perspective, determine what additional information would strengthen your analysis.

Response format (JSON):
{{
  "needs_info": true|false,
  "inferred_values": {{}},
  "required_fields": [],
  "reasoning": "why you need this information from your expert perspective"
}}""",
                "execute_task": f"""You are {expert['name']}, an expert in {expert['expertise']}.

Task: {{task}}
{{judge_feedback}}

Your perspective: {expert['perspective']}
Your capabilities: {{capabilities}}
Available tools: {{tool_context}}

Provide a comprehensive analysis from your expert viewpoint, focusing on {expert['perspective']}.""",
            },
            "persona": expert["name"].lower().replace(" ", "_"),
            "task_portion": f"Analysis from {expert['name']} perspective",
            "dependencies": [],
        }

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
