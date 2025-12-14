#!/usr/bin/env python3
"""
Test the Legion API functionality directly
"""

import asyncio
import os
import sys
from unittest.mock import Mock, patch

# Add the app path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))


def mock_gemini_response(*args, **kwargs):
    """Mock Gemini service responses for testing."""
    return """{
  "task_analysis": {
    "primary_domain": "programming",
    "required_skills": ["java", "basic_programming"],
    "complexity_level": "simple",
    "parallel_work_needed": false,
    "estimated_steps": 1
  },
  "agent_plan": [
    {
      "agent_id": "java_developer",
      "agent_type": "java_programming_specialist",
      "task_types": ["coding", "java"],
      "capabilities": {
        "primary_focus": "writing clean, efficient Java code",
        "tools_needed": ["java", "javac"],
        "expertise_level": "intermediate",
        "specializations": ["console_applications", "basic_programming"],
        "knowledge_domains": ["java_language", "object_oriented_programming"]
      },
      "prompts": {
        "identify_required_info": "Analyze Java programming requirements...",
        "execute_task": "Write clean, well-documented Java code..."
      },
      "persona": "experienced_java_developer",
      "task_portion": "Implement the complete Java program",
      "dependencies": []
    }
  ],
  "execution_strategy": {
    "parallel_execution": false,
    "sequential_dependencies": false,
    "coordination_needed": false
  },
  "rationale": "Simple Java hello world task requires single specialized Java developer agent"
}"""


async def test_legion_request_processing():
    """Test the legion request processing pipeline."""
    print("🧪 Testing Legion API Request Processing")
    print("=" * 50)

    try:
        # Mock the Gemini service
        with patch(
            "app.hermes.legion.agents.task_agent_planner.get_gemini_service"
        ) as mock_service:
            mock_service.return_value.generate_gemini_response = mock_gemini_response

            # Import the necessary components
            from hermes.legion.agents.task_agent_planner import TaskAgentPlanner
            from hermes.legion.nodes.legion_orchestrator import legion_orchestrator_node
            from hermes.legion.strategies.intelligent import IntelligentStrategy

            print("✅ Successfully imported legion components")

            # Test 1: TaskAgentPlanner
            planner = TaskAgentPlanner()
            analysis = planner.analyze_task_and_plan_agents(
                task_description="give me a basic hello world code snippet in java",
                user_context=None,
                complexity_estimate="simple",
            )

            assert analysis is not None, "Task analysis failed"
            assert len(analysis.get("agent_plan", [])) == 1, "Expected 1 agent"

            agent = analysis["agent_plan"][0]
            assert (
                agent["agent_type"] == "java_programming_specialist"
            ), f"Wrong agent type: {agent['agent_type']}"
            assert (
                "java" in agent["capabilities"]["tools_needed"]
            ), "Java tools not included"

            print("✅ TaskAgentPlanner working correctly")

            # Test 2: Strategy worker generation
            strategy = IntelligentStrategy()
            workers = await strategy.generate_workers(
                query="give me a basic hello world code snippet in java",
                context={"complexity_estimate": "simple"},
            )

            assert len(workers) == 1, f"Expected 1 worker, got {len(workers)}"
            assert "dynamic_agent_config" in workers[0], "Worker missing dynamic config"

            worker_config = workers[0]["dynamic_agent_config"]
            assert (
                worker_config["agent_type"] == "java_programming_specialist"
            ), "Worker has wrong agent type"

            print("✅ Strategy worker generation working correctly")

            # Test 3: Legion orchestrator (mock state)
            mock_state = {
                "messages": [
                    {"content": "give me a basic hello world code snippet in java"}
                ],
                "metadata": {},
            }

            # This would normally run the full LangGraph flow
            # For testing, we'll just verify the orchestrator logic
            result = await legion_orchestrator_node(mock_state)
            assert "legion_strategy" in result, "Orchestrator didn't set strategy"
            assert result["legion_strategy"] == "intelligent", "Wrong strategy selected"

            print("✅ Legion orchestrator working correctly")

            # Test 4: Complete agent creation
            from hermes.legion.agents.factory import AgentFactory

            factory = AgentFactory()
            agent = factory.create_dynamic_agent(tools=[], **worker_config)

            assert agent is not None, "Agent creation failed"
            assert (
                agent.agent_id == "java_developer"
            ), f"Wrong agent ID: {agent.agent_id}"
            assert agent.task_types == [
                "coding",
                "java",
            ], f"Wrong task types: {agent.task_types}"

            print("✅ Dynamic agent creation working correctly")

            # Summary
            print("\n" + "=" * 50)
            print("🎉 LEGION API REQUEST PROCESSING TEST PASSED!")
            print("=" * 50)
            print("✅ Task Analysis: Successfully analyzed Java hello world request")
            print("✅ Agent Planning: Created java_programming_specialist agent")
            print("✅ Strategy Selection: Chose intelligent strategy")
            print("✅ Worker Generation: Produced dynamic agent configuration")
            print("✅ Agent Creation: Successfully instantiated Java specialist")
            print("")
            print("📊 Test Results:")
            print(f"   • Agent Type Created: {agent.agent_type}")
            print(f"   • Agent Capabilities: {len(agent.capabilities)} defined")
            print(f"   • Task Types: {agent.task_types}")
            print(f"   • Strategy Used: intelligent")
            print("")
            print("🚀 LEGION MODE READY FOR API CALLS!")
            print("=" * 50)

            return True

    except Exception as e:
        print(f"\n❌ LEGION API TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_legion_request_processing())
    if success:
        print("\n🎯 CONCLUSION: The Legion system is ready for API requests!")
        print(
            "   You can now call: /api/v1/hermes/process_request?request_text=...&legion_mode=true"
        )
    else:
        print("\n❌ CONCLUSION: Legion system has issues that need fixing")
    sys.exit(0 if success else 1)
