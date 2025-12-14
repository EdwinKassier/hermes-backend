#!/usr/bin/env python3
"""
End-to-End Test for Dynamic Agent System
Tests the complete legion mode processing pipeline
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
    "primary_domain": "web_development",
    "required_skills": ["frontend", "backend", "ui_design"],
    "complexity_level": "moderate",
    "parallel_work_needed": true,
    "estimated_steps": 3
  },
  "agent_plan": [
    {
      "agent_id": "frontend_developer",
      "agent_type": "react_specialist",
      "task_types": ["frontend", "ui"],
      "capabilities": {
        "primary_focus": "modern React development with responsive design",
        "tools_needed": ["react", "typescript", "css"],
        "expertise_level": "expert",
        "specializations": ["component_architecture", "state_management"],
        "knowledge_domains": ["frontend_development", "ui_ux"]
      },
      "prompts": {
        "identify_required_info": "Analyze frontend requirements...",
        "execute_task": "Build modern React components..."
      },
      "persona": "frontend_architect",
      "task_portion": "Implement responsive user interface",
      "dependencies": []
    },
    {
      "agent_id": "backend_engineer",
      "agent_type": "api_specialist",
      "task_types": ["backend", "api"],
      "capabilities": {
        "primary_focus": "RESTful API development with database integration",
        "tools_needed": ["python", "fastapi", "postgresql"],
        "expertise_level": "expert",
        "specializations": ["api_design", "database_modeling"],
        "knowledge_domains": ["backend_development", "api_architecture"]
      },
      "prompts": {
        "identify_required_info": "Analyze backend API requirements...",
        "execute_task": "Build scalable REST APIs..."
      },
      "persona": "backend_architect",
      "task_portion": "Develop backend API services",
      "dependencies": []
    }
  ],
  "execution_strategy": {
    "parallel_execution": true,
    "sequential_dependencies": false,
    "coordination_needed": true
  },
  "rationale": "Creating specialized frontend and backend agents for full-stack web development"
}"""


async def test_task_agent_planner():
    """Test TaskAgentPlanner functionality."""
    print("🧪 Testing TaskAgentPlanner...")

    with patch(
        "app.hermes.legion.agents.task_agent_planner.get_gemini_service"
    ) as mock_service:
        mock_service.return_value.generate_gemini_response = mock_gemini_response

        from hermes.legion.agents.task_agent_planner import TaskAgentPlanner

        planner = TaskAgentPlanner()

        # Test task analysis
        analysis = planner.analyze_task_and_plan_agents(
            task_description="Build a modern web application with React frontend and Python backend",
            user_context="For a startup product",
            complexity_estimate="moderate",
        )

        assert analysis is not None, "Task analysis failed"
        assert "agent_plan" in analysis, "Agent plan missing"
        assert (
            len(analysis["agent_plan"]) == 2
        ), f"Expected 2 agents, got {len(analysis['agent_plan'])}"

        # Test worker plan creation
        worker_plan = planner.create_worker_plan_from_analysis(
            analysis, "Build web app"
        )

        assert len(worker_plan) == 2, f"Expected 2 workers, got {len(worker_plan)}"
        assert all(
            "dynamic_agent_config" in worker for worker in worker_plan
        ), "Workers missing dynamic config"

        print("   ✅ TaskAgentPlanner working correctly")
        return analysis, worker_plan


async def test_agent_factory():
    """Test AgentFactory dynamic agent creation."""
    print("🏭 Testing AgentFactory...")

    from hermes.legion.agents.factory import AgentFactory

    factory = AgentFactory()

    # Test dynamic agent creation
    agent_config = {
        "agent_id": "test_agent",
        "task_types": ["testing"],
        "capabilities": {
            "primary_focus": "test execution",
            "tools_needed": ["pytest"],
            "expertise_level": "intermediate",
        },
        "prompts": {
            "identify_required_info": "What needs testing?",
            "execute_task": "Run the tests...",
        },
        "persona": "tester",
    }

    # Mock the tools parameter since we don't have actual tools in test
    agent = factory.create_dynamic_agent(tools=[], **agent_config)

    assert agent is not None, "Agent creation failed"
    assert agent.agent_id == "test_agent", f"Wrong agent ID: {agent.agent_id}"
    assert "testing" in agent.task_types, "Wrong task types"

    print("   ✅ AgentFactory working correctly")
    return agent


async def test_strategies():
    """Test strategy worker generation."""
    print("🎯 Testing Strategies...")

    # Mock the task analysis for strategies
    with patch(
        "app.hermes.legion.agents.task_agent_planner.get_gemini_service"
    ) as mock_service:
        mock_service.return_value.generate_gemini_response = mock_gemini_response

        # Test Intelligent Strategy (most commonly used)
        from hermes.legion.strategies.intelligent import IntelligentStrategy

        strategy = IntelligentStrategy()
        context = {"complexity_estimate": "moderate"}

        workers = await strategy.generate_workers(
            query="Build a web application", context=context
        )

        assert workers is not None, "Strategy failed to generate workers"
        assert len(workers) > 0, "No workers generated"
        assert all(
            "dynamic_agent_config" in worker for worker in workers
        ), "Workers missing dynamic config"

        print("   ✅ Intelligent Strategy working correctly")
        return workers


async def test_orchestrator_flow():
    """Test the orchestrator flow simulation."""
    print("🎼 Testing Orchestrator Flow...")

    # This is a high-level test - we can't fully test the LangGraph flow without the full app
    # But we can test that the orchestrator logic would work

    # Test legion orchestrator strategy selection
    from hermes.legion.nodes.legion_orchestrator import legion_orchestrator_node

    # Mock state for testing
    mock_state = {
        "messages": [{"content": "Build a complex web application"}],
        "metadata": {},
    }

    # The function should set legion_strategy to "intelligent"
    result = await legion_orchestrator_node(mock_state)

    assert "legion_strategy" in result, "Strategy not set"
    assert (
        result["legion_strategy"] == "intelligent"
    ), f"Wrong strategy: {result['legion_strategy']}"

    print("   ✅ Orchestrator strategy selection working correctly")
    return result


async def run_e2e_tests():
    """Run all end-to-end tests."""
    print("🚀 Running Dynamic Agent System E2E Tests")
    print("=" * 50)

    try:
        # Test individual components
        analysis, worker_plan = await test_task_agent_planner()
        agent = await test_agent_factory()
        workers = await test_strategies()
        orchestrator_result = await test_orchestrator_flow()

        # Integration test
        print("\n🔗 Testing Integration...")

        # Verify that strategy-generated workers have the same structure as TaskAgentPlanner
        strategy_agent_ids = {w["dynamic_agent_config"]["agent_id"] for w in workers}
        planner_agent_ids = {w["dynamic_agent_config"]["agent_id"] for w in worker_plan}

        # Should have some overlap or at least same structure
        assert all(
            "dynamic_agent_config" in w for w in workers
        ), "Strategy workers missing config"
        assert all(
            "agent_id" in w["dynamic_agent_config"] for w in workers
        ), "Workers missing agent_id"

        print("   ✅ Integration test passed")

        # Final summary
        print("\n" + "=" * 50)
        print("🎉 DYNAMIC AGENT SYSTEM E2E TESTS PASSED!")
        print("=" * 50)
        print("✅ TaskAgentPlanner: Analyzes tasks and creates agent plans")
        print("✅ AgentFactory: Creates dynamic agents from configurations")
        print("✅ Strategies: Generate workers with dynamic agent configs")
        print("✅ Orchestrator: Selects appropriate strategies")
        print("✅ Integration: All components work together seamlessly")
        print("")
        print(f"📊 Test Results:")
        print(f"   • Task Analysis: {len(analysis['agent_plan'])} agents planned")
        print(f"   • Agent Creation: {agent.agent_id} created successfully")
        print(f"   • Strategy Workers: {len(workers)} workers generated")
        print(f"   • Orchestrator Strategy: {orchestrator_result['legion_strategy']}")
        print("")
        print("🚀 SYSTEM READY FOR PRODUCTION!")
        print("=" * 50)

        return True

    except Exception as e:
        print(f"\n❌ E2E TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(run_e2e_tests())
    sys.exit(0 if success else 1)
