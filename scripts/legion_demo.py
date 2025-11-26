"""
Demo script showing how to use the new LangGraph-based Legion system.

This demonstrates the Magentic Orchestrator pattern implementation with:
- Dynamic agent creation
- Task-based tool allocation
- Graph-based workflow orchestration
- State persistence via checkpointing
"""

from app.hermes.legion import LegionGraphService, get_legion_graph_service
from app.hermes.legion.agents.factory import AgentFactory
from app.hermes.legion.utils import ToolAllocator
from app.hermes.models import ResponseMode, UserIdentity


def demo_basic_usage():
    """Demonstrate basic usage of the LangGraph-based Legion service."""
    print("=" * 60)
    print("LangGraph Legion Service Demo")
    print("=" * 60)

    # Get the service instance
    legion_service = get_legion_graph_service()

    print("\n✅ LegionGraphService initialized")
    print(f"   - Using LangGraph StateGraph for orchestration")
    print(f"   - Checkpointing enabled for state persistence")
    print(f"   - Dynamic agent creation enabled")
    print(f"   - Task-based tool allocation enabled")

    # Show available agent types
    print("\n📋 Available Agent Types:")
    agent_types = AgentFactory.get_available_agent_types()
    for agent_type in agent_types:
        print(f"   - {agent_type}")

    # Show tool allocator capabilities
    tool_allocator = ToolAllocator()
    capabilities = tool_allocator.get_tool_capabilities()
    print(f"\n🔧 Tool Allocation Capabilities ({len(capabilities)} tools configured):")
    for tool_name, task_types in list(capabilities.items())[:5]:  # Show first 5
        print(f"   - {tool_name}: {', '.join(task_types)}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


def demo_agent_creation():
    """Demonstrate dynamic agent creation."""
    print("\n🤖 Dynamic Agent Creation Demo")
    print("-" * 60)

    from app.hermes.legion.state import AgentConfig

    # Create agent configuration
    config = AgentConfig(
        agent_type="research",
        required_tools=["search", "summarize"],
        max_iterations=5,
    )

    print(f"\n1. Created AgentConfig:")
    print(f"   - Type: {config.agent_type}")
    print(f"   - Required tools: {config.required_tools}")

    # Create agent from task
    agent, agent_info = AgentFactory.create_agent_from_task(
        task_description="Research quantum computing trends",
        task_type="research",
        tools=[],
    )

    print(f"\n2. Dynamically created agent:")
    print(f"   - Agent ID: {agent.agent_id}")
    print(f"   - Agent type: {agent_info.agent_type}")
    print(f"   - Task types handled: {agent.task_types}")


def demo_tool_allocation():
    """Demonstrate task-based tool allocation."""
    print("\n🔧 Dynamic Tool Allocation Demo")
    print("-" * 60)

    tool_allocator = ToolAllocator()

    # Simulate tool allocation for different task types
    task_types = ["research", "code", "analysis"]

    for task_type in task_types:
        # Note: This would normally get actual tools from GeminiService
        # For demo, we're just showing the concept
        print(f"\n  Task type: {task_type}")

        capabilities = tool_allocator.get_tool_capabilities()
        relevant_tools = [
            tool_name for tool_name, types in capabilities.items() if task_type in types
        ]

        print(f"  Allocated tools: {', '.join(relevant_tools[:3])}")


def main():
    """Run all demos."""
    try:
        demo_basic_usage()
        demo_agent_creation()
        demo_tool_allocation()

        print("\n\n✨ All demos completed successfully!")
        print("\nNext steps:")
        print("  1. Integrate with your API endpoints")
        print("  2. Test with real user requests")
        print("  3. Add more agent types as needed")
        print("  4. Configure persistent checkpointing (PostgreSQL/SQLite)")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
