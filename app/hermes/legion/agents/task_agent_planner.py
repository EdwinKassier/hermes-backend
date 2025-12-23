"""Task-driven agent planner that analyzes tasks and creates appropriate dynamic agents."""

import logging
from typing import Any, Dict, List, Optional

from app.shared.utils.service_loader import get_async_llm_service, get_llm_service

from .dynamic_agent import DynamicAgent

logger = logging.getLogger(__name__)


class TaskAgentPlanner:
    """
    Analyzes tasks and dynamically creates the appropriate mix of agents needed.

    This is the core of the flexible agent system - it can analyze any task and
    determine what types of agents (research, coding, analysis, etc.) are needed,
    then creates them dynamically without any hardcoded logic.

    Supports both sync and async LLM calls:
    - Use analyze_task_and_plan_agents() for sync contexts
    - Use analyze_task_and_plan_agents_async() for async contexts (recommended)
    """

    def __init__(self):
        self._llm_service = None
        self._async_llm_service = None

    @property
    def llm_service(self):
        """Lazy load LLM service."""
        if self._llm_service is None:
            self._llm_service = get_llm_service()
        return self._llm_service

    @property
    def async_llm_service(self):
        """Lazy load async LLM service."""
        if self._async_llm_service is None:
            self._async_llm_service = get_async_llm_service()
        return self._async_llm_service

    def analyze_task_and_plan_agents(
        self,
        task_description: str,
        user_context: Optional[str] = None,
        complexity_estimate: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a task and determine what agents are needed to complete it.

        This is the key innovation: the system can analyze any task and dynamically
        create whatever combination of agents is needed, without hardcoded mappings.

        Args:
            task_description: The task to analyze
            user_context: Additional context about the user or requirements
            complexity_estimate: Estimated complexity (simple, moderate, complex)

        Returns:
            Dictionary containing agent plan with configurations
        """
        analysis_prompt = f"""Analyze this task and determine what types of AI agents are needed to complete it effectively.

Task: "{task_description}"
{f'User Context: {user_context}' if user_context else ''}
{f'Complexity Estimate: {complexity_estimate}' if complexity_estimate else ''}

Your goal is to create a dynamic team of AI agents that will work together to solve this task. Each agent should have:
- A specific role/purpose
- Required capabilities and expertise
- Appropriate tools and approaches

You have the ability to create ANY type of agent dynamically. Analyze the task and determine what specialized agents are needed, then define each agent's complete configuration from scratch.

Consider:
1. What specific skills/knowledge areas does this task require?
2. Should work be divided among different specialists?
3. What would be the most effective combination of agents?
4. Are there dependencies between different parts of the work?
5. What unique agent types need to be invented for this task?

For each agent, you must define:
- agent_id: A human-readable name like "Research Specialist" or "Technical Writer" (avoid camelCase)
- agent_type: A descriptive role name like "quantum_computing_researcher" or "technical_explanation_specialist"
- task_types: What types of tasks this agent handles
- capabilities: Specific skills, tools, and expertise areas
- prompts: Complete prompt templates for identify_required_info and execute_task
- persona: Appropriate AI persona for this agent

CRITICAL REQUIREMENT: In the 'execute_task' prompt for each agent, you MUST include instructions for a 'Knowledge Fallback Strategy'.
If tools fail, are unavailable, or return no results, the agent MUST use its own internal knowledge to answer the user's request as best as possible.
It should NOT refuse to answer or say 'I cannot do that because the tool failed'. It should say 'Tool [name] failed, but based on my knowledge...' and provide the answer.

Response Format (JSON):
{{
  "task_analysis": {{
    "primary_domain": "main area of expertise needed",
    "required_skills": ["skill1", "skill2"],
    "complexity_level": "simple|moderate|complex",
    "parallel_work_needed": true|false,
    "estimated_steps": 3-8
  }},
  "agent_plan": [
    {{
      "agent_id": "Research Specialist",
      "agent_type": "quantum_computing_research_specialist",
      "task_types": ["task_type1", "task_type2"],
      "capabilities": {{
        "primary_focus": "main responsibility area",
        "tools_needed": ["tool1", "tool2", "tool3"],
        "expertise_level": "beginner|intermediate|expert",
        "specializations": ["specific_area1", "specific_area2"],
        "knowledge_domains": ["domain1", "domain2"]
      }},
      "prompts": {{
        "identify_required_info": "Complete prompt template for gathering requirements...",
        "execute_task": "Complete prompt template for task execution..."
      }},
      "persona": "appropriate_persona_name",
      "task_portion": "what part of the overall task this agent handles",
      "dependencies": ["other_agent_ids_this_depends_on"]
    }}
  ],
  "execution_strategy": {{
    "parallel_execution": true|false,
    "sequential_dependencies": true|false,
    "coordination_needed": true|false
  }},
  "rationale": "why this combination of agents was chosen"
}}

Analyze the task and create completely custom agent types as needed."""

        try:
            response = self.llm_service.generate_response(
                prompt=analysis_prompt, persona="legion"
            )

            # Extract JSON from response
            import json
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group(0))

                # Validate the plan structure
                if self._validate_agent_plan(plan):
                    logger.info(
                        f"Generated agent plan with {len(plan.get('agent_plan', []))} agents"
                    )
                    return plan
                else:
                    logger.warning("Invalid agent plan structure, using fallback")
                    return self._create_fallback_plan(task_description)

        except Exception as e:
            logger.error(f"Failed to analyze task and plan agents: {e}")
            return self._create_fallback_plan(task_description)

    async def analyze_task_and_plan_agents_async(
        self,
        task_description: str,
        user_context: Optional[str] = None,
        complexity_estimate: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Async version of analyze_task_and_plan_agents for use in async contexts.

        This uses AsyncLLMService to prevent blocking the event loop.

        Args:
            task_description: The task to analyze
            user_context: Additional context about the user or requirements
            complexity_estimate: Estimated complexity (simple, moderate, complex)

        Returns:
            Dictionary containing agent plan with configurations
        """
        analysis_prompt = f"""Analyze this task and determine what types of AI agents are needed to complete it effectively.

Task: "{task_description}"
{f'User Context: {user_context}' if user_context else ''}
{f'Complexity Estimate: {complexity_estimate}' if complexity_estimate else ''}

Your goal is to create a dynamic team of AI agents that will work together to solve this task. Each agent should have:
- A specific role/purpose
- Required capabilities and expertise
- Appropriate tools and approaches

You have the ability to create ANY type of agent dynamically. Analyze the task and determine what specialized agents are needed, then define each agent's complete configuration from scratch.

Consider:
1. What specific skills/knowledge areas does this task require?
2. Should work be divided among different specialists?
3. What would be the most effective combination of agents?
4. Are there dependencies between different parts of the work?
5. What unique agent types need to be invented for this task?

For each agent, you must define:
- agent_id: A human-readable name like "Research Specialist" or "Technical Writer" (avoid camelCase)
- agent_type: A descriptive role name like "quantum_computing_researcher" or "technical_explanation_specialist"
- task_types: What types of tasks this agent handles
- capabilities: Specific skills, tools, and expertise areas
- prompts: Complete prompt templates for identify_required_info and execute_task
- persona: Appropriate AI persona for this agent

CRITICAL REQUIREMENT: In the 'execute_task' prompt for each agent, you MUST include instructions for a 'Knowledge Fallback Strategy'.
If tools fail, are unavailable, or return no results, the agent MUST use its own internal knowledge to answer the user's request as best as possible.
It should NOT refuse to answer or say 'I cannot do that because the tool failed'. It should say 'Tool [name] failed, but based on my knowledge...' and provide the answer.

Response Format (JSON):
{{
  "task_analysis": {{
    "primary_domain": "main area of expertise needed",
    "required_skills": ["skill1", "skill2"],
    "complexity_level": "simple|moderate|complex",
    "parallel_work_needed": true|false,
    "estimated_steps": 3-8
  }},
  "agent_plan": [
    {{
      "agent_id": "Research Specialist",
      "agent_type": "quantum_computing_research_specialist",
      "task_types": ["task_type1", "task_type2"],
      "capabilities": {{
        "primary_focus": "main responsibility area",
        "tools_needed": ["tool1", "tool2", "tool3"],
        "expertise_level": "beginner|intermediate|expert",
        "specializations": ["specific_area1", "specific_area2"],
        "knowledge_domains": ["domain1", "domain2"]
      }},
      "prompts": {{
        "identify_required_info": "Complete prompt template for gathering requirements...",
        "execute_task": "Complete prompt template for task execution..."
      }},
      "persona": "appropriate_persona_name",
      "task_portion": "what part of the overall task this agent handles",
      "dependencies": ["other_agent_ids_this_depends_on"]
    }}
  ],
  "execution_strategy": {{
    "parallel_execution": true|false,
    "sequential_dependencies": true|false,
    "coordination_needed": true|false
  }},
  "rationale": "why this combination of agents was chosen"
}}

Analyze the task and create completely custom agent types as needed."""

        try:
            # Use async LLM service for non-blocking call
            response = await self.async_llm_service.generate_async(
                prompt=analysis_prompt, persona="hermes"
            )

            # Extract JSON from response
            import json
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group(0))

                # Validate the plan structure
                if self._validate_agent_plan(plan):
                    logger.info(
                        f"Generated agent plan (async) with {len(plan.get('agent_plan', []))} agents"
                    )
                    return plan
                else:
                    logger.warning(
                        "Invalid agent plan structure (async), using fallback"
                    )
                    return self._create_fallback_plan(task_description)

        except Exception as e:
            logger.error(f"Failed to analyze task and plan agents (async): {e}")
            return self._create_fallback_plan(task_description)

    def create_worker_plan_from_analysis(
        self, analysis_result: Dict[str, Any], task_description: str
    ) -> List[Dict[str, Any]]:
        """
        Convert the analysis result into a worker plan with dynamic agents.

        Args:
            analysis_result: Result from analyze_task_and_plan_agents
            task_description: Original task description

        Returns:
            Worker plan suitable for Legion orchestration
        """
        from .dynamic_agent_utils import create_worker_plan_with_dynamic_agents

        agent_configs = []
        execution_levels = []

        for i, agent_config in enumerate(analysis_result.get("agent_plan", [])):
            # Create complete agent configuration from scratch
            config = {
                "agent_id": agent_config["agent_id"],
                "task_types": agent_config["task_types"],
                "capabilities": agent_config["capabilities"],
                "prompts": agent_config["prompts"],
                "persona": agent_config["persona"],
                "agent_type": agent_config["agent_type"],
                "task_portion": agent_config.get("task_portion", ""),
                "dependencies": agent_config.get("dependencies", []),
            }

            agent_configs.append(config)

            # Determine execution level based on dependencies
            dependencies = agent_config.get("dependencies", [])
            if dependencies:
                # This agent depends on others, so it should run later
                execution_levels.append(1)
            else:
                # No dependencies, can run in parallel with others
                execution_levels.append(0)

        # Create the worker plan
        worker_plan = create_worker_plan_with_dynamic_agents(
            task_description=task_description, agent_configs=agent_configs
        )

        # Add execution levels
        for i, worker in enumerate(worker_plan):
            if i < len(execution_levels):
                worker["execution_level"] = execution_levels[i]

        logger.info(
            f"Created worker plan with {len(worker_plan)} dynamic agents from task analysis"
        )
        return worker_plan

    def _validate_agent_plan(self, plan: Dict[str, Any]) -> bool:
        """Validate that the agent plan has the required structure."""
        required_keys = ["task_analysis", "agent_plan", "execution_strategy"]

        if not all(key in plan for key in required_keys):
            return False

        agent_plan = plan.get("agent_plan", [])
        if not isinstance(agent_plan, list) or len(agent_plan) == 0:
            return False

        # Validate each agent specification
        for agent in agent_plan:
            required_agent_keys = [
                "agent_id",
                "agent_type",
                "task_types",
                "capabilities",
                "prompts",
                "persona",
            ]
            if not all(key in agent for key in required_agent_keys):
                return False

        return True

    def _create_fallback_plan(self, task_description: str) -> Dict[str, Any]:
        """Create a basic fallback agent plan when analysis fails."""
        logger.info("Using fallback agent plan")

        # Simple heuristic-based fallback
        task_lower = task_description.lower()

        if "code" in task_lower or "implement" in task_lower or "build" in task_lower:
            # Coding task - use code_generator
            return {
                "task_analysis": {
                    "primary_domain": "coding",
                    "required_skills": ["programming"],
                    "complexity_level": "moderate",
                    "parallel_work_needed": False,
                    "estimated_steps": 3,
                },
                "agent_plan": [
                    {
                        "agent_id": "primary_coder",
                        "template": "code_generator",
                        "role": "Implement the requested code",
                        "capabilities": {
                            "primary_focus": "code_implementation",
                            "tools_needed": ["coding"],
                            "expertise_level": "intermediate",
                        },
                        "persona": "hermes",
                        "task_portion": "Complete implementation",
                        "dependencies": [],
                    }
                ],
                "execution_strategy": {
                    "parallel_execution": False,
                    "sequential_dependencies": False,
                    "coordination_needed": False,
                },
                "rationale": "Fallback plan: coding task detected, using code generator agent",
            }

        elif (
            "research" in task_lower
            or "analyze" in task_lower
            or "investigate" in task_lower
        ):
            # Research task - use research_specialist
            return {
                "task_analysis": {
                    "primary_domain": "research",
                    "required_skills": ["research", "analysis"],
                    "complexity_level": "moderate",
                    "parallel_work_needed": False,
                    "estimated_steps": 4,
                },
                "agent_plan": [
                    {
                        "agent_id": "researcher",
                        "template": "research_specialist",
                        "role": "Conduct research and analysis",
                        "capabilities": {
                            "primary_focus": "information_gathering",
                            "tools_needed": ["research", "analysis"],
                            "expertise_level": "intermediate",
                        },
                        "persona": "hermes",
                        "task_portion": "Complete research task",
                        "dependencies": [],
                    }
                ],
                "execution_strategy": {
                    "parallel_execution": False,
                    "sequential_dependencies": False,
                    "coordination_needed": False,
                },
                "rationale": "Fallback plan: research task detected, using research specialist agent",
            }

        else:
            # General task - use analytical expert
            return {
                "task_analysis": {
                    "primary_domain": "general",
                    "required_skills": ["analysis", "problem_solving"],
                    "complexity_level": "simple",
                    "parallel_work_needed": False,
                    "estimated_steps": 2,
                },
                "agent_plan": [
                    {
                        "agent_id": "general_assistant",
                        "template": "analytical_expert",
                        "role": "Analyze and address the request",
                        "capabilities": {
                            "primary_focus": "problem_solving",
                            "tools_needed": ["analysis"],
                            "expertise_level": "intermediate",
                        },
                        "persona": "hermes",
                        "task_portion": "Complete the general task",
                        "dependencies": [],
                    }
                ],
                "execution_strategy": {
                    "parallel_execution": False,
                    "sequential_dependencies": False,
                    "coordination_needed": False,
                },
                "rationale": "Fallback plan: general task, using analytical expert agent",
            }


def demonstrate_task_driven_agents():
    """Demonstrate how the system creates agents based on task analysis."""

    planner = TaskAgentPlanner()

    print("🎯 Task-Driven Dynamic Agent Creation")
    print("=" * 50)

    # Example 1: Complex web application
    task1 = "Build a full-stack web application for managing personal finances with data visualization and expense tracking"
    print(f"\n📋 Task: {task1}")

    analysis1 = planner.analyze_task_and_plan_agents(
        task1, complexity_estimate="complex"
    )
    plan1 = planner.create_worker_plan_from_analysis(analysis1, task1)

    print(f"Analysis: {analysis1['task_analysis']['primary_domain']} domain")
    print(f"Agents created: {len(plan1)}")
    for worker in plan1:
        config = worker.get("dynamic_agent_config", {})
        print(
            f"  • {config.get('agent_id', 'unknown')}: {config.get('task_types', [])}"
        )

    # Example 2: Research project
    task2 = "Research the latest developments in quantum computing and their potential impact on cryptography"
    print(f"\n📋 Task: {task2}")

    analysis2 = planner.analyze_task_and_plan_agents(
        task2, complexity_estimate="complex"
    )
    plan2 = planner.create_worker_plan_from_analysis(analysis2, task2)

    print(f"Analysis: {analysis2['task_analysis']['primary_domain']} domain")
    print(f"Agents created: {len(plan2)}")
    for worker in plan2:
        config = worker.get("dynamic_agent_config", {})
        print(
            f"  • {config.get('agent_id', 'unknown')}: {config.get('task_types', [])}"
        )

    # Example 3: Data analysis project
    task3 = "Analyze sales data from the past year, identify trends, and create a dashboard with recommendations"
    print(f"\n📋 Task: {task3}")

    analysis3 = planner.analyze_task_and_plan_agents(
        task3, complexity_estimate="moderate"
    )
    plan3 = planner.create_worker_plan_from_analysis(analysis3, task3)

    print(f"Analysis: {analysis3['task_analysis']['primary_domain']} domain")
    print(f"Agents created: {len(plan3)}")
    for worker in plan3:
        config = worker.get("dynamic_agent_config", {})
        print(
            f"  • {config.get('agent_id', 'unknown')}: {config.get('task_types', [])}"
        )

    print(
        "\n✨ The system dynamically creates whatever agents are needed for each task!"
    )
    print("   No hardcoded agent types - just task analysis and dynamic creation.")


if __name__ == "__main__":
    demonstrate_task_driven_agents()
