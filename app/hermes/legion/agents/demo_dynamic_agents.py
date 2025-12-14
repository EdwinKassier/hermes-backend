#!/usr/bin/env python3
"""
Demonstration of the new dynamic agent system.

This script shows how to create multiple different implementations of agents
dynamically, without hardcoded classes.
"""

import logging
from typing import List

from .factory import AgentFactory
from .templates import AGENT_TEMPLATES

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_dynamic_agents():
    """Demonstrate creating different types of dynamic agents."""

    print("🔧 Dynamic Agent System Demonstration")
    print("=" * 50)

    # 1. Show available templates
    print("\n📋 Available Agent Templates:")
    templates = AgentFactory.get_available_templates()
    for template in templates:
        print(f"  • {template}")

    # 2. Create different coding agents from the same template with variations
    print("\n👨‍💻 Creating Multiple Coding Agents:")

    coding_agents = AgentFactory.create_multiple_from_template(
        template_name="code_generator",
        count=3,
        agent_ids=["python_expert", "web_developer", "algorithm_specialist"],
        persona_variations=["hermes", "creative", "analyst"],
        # Add custom capabilities for each
        languages=["python", "javascript", "cpp"],
        style=["clean", "modern", "optimized"],
    )

    for agent in coding_agents:
        print(
            f"  ✓ Created {agent.agent_id} ({agent.persona}) - handles: {agent.task_types}"
        )

    # 3. Create a custom dynamic agent from scratch
    print("\n🎨 Creating Custom Dynamic Agent:")

    custom_agent = AgentFactory.create_dynamic_agent(
        agent_id="game_developer",
        task_types=["code", "game", "creative"],
        capabilities={
            "languages": ["javascript", "python"],
            "expertise": ["game_logic", "graphics", "physics", "ui"],
            "engines": ["phaser", "pygame", "three_js"],
            "style": "interactive and engaging",
        },
        prompts={
            "identify_required_info": """Analyze this game development request...

User Request: "{user_message}"
Task: "{task}"

**Game Development Capabilities**:
{languages}
{expertise}
{engines}

Focus on creating fun, interactive games with good user experience.

**Response Format (JSON)**:
{{
  "needs_info": false,
  "inferred_values": {{
    "game_type": "<what you'll build>",
    "platform": "<web or desktop>",
    "complexity": "<simple|moderate|complex>"
  }},
  "reasoning": "<why you can proceed>"
}}

Or with required_fields if needed.""",
            "execute_task": """Create an awesome game!

Task: {task}
{judge_feedback}

Game Requirements: {collected_info}

**Capabilities**: {capabilities}

{tool_context}

Create:
1. Game concept and design
2. Complete working code
3. Instructions for running
4. Ideas for enhancements

Make it fun and engaging!""",
        },
        persona="creative",
        focus="gaming",
        platform="web_first",
    )

    print(f"  ✓ Created {custom_agent.agent_id} - handles: {custom_agent.task_types}")

    # 4. Create specialized data analysis agents
    print("\n📊 Creating Specialized Data Analysis Agents:")

    analysis_agents = AgentFactory.create_multiple_from_template(
        template_name="data_analyst",
        count=2,
        agent_ids=["statistical_analyst", "visualization_expert"],
        persona="analyst",
        # Different specializations
        focus_areas=["statistics", "visualization"],
        tools=["pandas", "matplotlib"],
    )

    for agent in analysis_agents:
        print(
            f"  ✓ Created {agent.agent_id} ({agent.persona}) - handles: {agent.task_types}"
        )

    # 5. Show how agents can be created from templates with overrides
    print("\n🔄 Creating Agent with Template Overrides:")

    research_agent = AgentFactory.create_agent_from_template(
        template_name="research_specialist",
        agent_id="ai_researcher",
        persona="hermes",
        # Override capabilities
        capabilities_override={
            "domains": [
                "artificial_intelligence",
                "machine_learning",
                "neural_networks",
            ],
            "specialization": "AI/ML research",
        },
        # Override prompts to be AI-specific
        prompts_override={
            "identify_required_info": """Analyze this AI research request...

User Request: "{user_message}"
Task: "{task}"

**AI Research Focus**: Specialized in artificial intelligence, machine learning, and neural networks.

Determine if information is needed for AI research..."""
        },
    )

    print(f"  ✓ Created {research_agent.agent_id} - specialized in AI research")

    print("\n✅ Dynamic Agent System Demo Complete!")
    print("\nKey Benefits:")
    print("• No hardcoded agent classes needed")
    print("• Create multiple variants of same agent type")
    print("• Easy customization via templates and overrides")
    print("• Flexible configuration for different use cases")
    print("• Maintains backward compatibility with existing agents")

    return {
        "coding_agents": coding_agents,
        "custom_agent": custom_agent,
        "analysis_agents": analysis_agents,
        "research_agent": research_agent,
    }


if __name__ == "__main__":
    # Run the demonstration
    agents = demonstrate_dynamic_agents()

    # Show that agents are ready to use
    print("\n🚀 Agents Ready for Use:")
    for name, agent_list in agents.items():
        if isinstance(agent_list, list):
            for agent in agent_list:
                print(f"  • {agent.agent_id}: {type(agent).__name__}")
        else:
            print(f"  • {agent_list.agent_id}: {type(agent_list).__name__}")
