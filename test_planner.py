from app.hermes.legion.agents.task_agent_planner import TaskAgentPlanner


def test():
    planner = TaskAgentPlanner()
    result = planner.analyze_task_and_plan_agents(
        "explain the basics of quantum computing"
    )
    print("Agent plan:")
    for agent in result.get("agent_plan", []):
        print(f'  - {agent["agent_id"]}: {agent.get("task_portion", "")}')
    print(f"Full result keys: {list(result.keys())}")


if __name__ == "__main__":
    test()
