"""
Example of the new decision rationale metadata structure.

This shows what will be included in the metadata when using LangGraph service.
"""

# Example response metadata with explainability
example_metadata = {
    "legion_mode": True,
    "langgraph_enabled": True,
    "model": "gemini-pro",
    # NEW: Decision rationale - history of all orchestrator decisions
    "decision_rationale": [
        {
            "timestamp": "2025-11-18T18:30:00",
            "node": "orchestrator",
            "analysis": {
                "user_message": "Research quantum computing and analyze the findings...",
                "identified_task_type": "research",
                "tool_allocation": {
                    "task_type": "research",
                    "tools_allocated": ["search", "web_search", "document_retrieval"],
                    "allocation_strategy": "task-based (not persona-based)",
                },
                "required_information": [],  # Empty = no info needed from user
            },
            "decisions": {
                "agent_needed": True,
                "selected_task_type": "research",
                "agent_created": True,
                "agent_id": "research_agent",
                "agent_type": "research",
                "task_created": True,
                "task_id": "task_1234567890",
                "action": "execute_agent",
            },
            "reasoning": {
                "task_analysis": "No specific task keywords detected (research, code, analysis, data)",
                "agent_selection": "Created 'research' agent (ID: research_agent) because task type 'research' matches this agent's capabilities: ['research', 'investigation', 'analysis']",
                "tool_selection": "Selected tools relevant to 'research' tasks based on tool capability mappings",
                "action": "Agent has all information needed, proceeding to execution",
            },
        }
    ],
    # NEW: Current reasoning context (most recent decision)
    "orchestration_reasoning": {
        "timestamp": "2025-11-18T18:30:00",
        "node": "orchestrator",
        "analysis": {
            "user_message": "Research quantum computing...",
            "identified_task_type": "research",
            "tool_allocation": {
                "task_type": "research",
                "tools_allocated": ["search", "web_search"],
                "allocation_strategy": "task-based (not persona-based)",
            },
        },
        "decisions": {
            "agent_needed": True,
            "agent_type": "research",
            "action": "execute_agent",
        },
        "reasoning": {
            "agent_selection": "Created 'research' agent because task type 'research' matches capabilities",
            "tool_selection": "Selected search tools relevant to research tasks",
            "action": "Agent ready with all info, executing",
        },
    },
    # Existing metadata
    "agents_used": ["research_agent"],
    "task_ledger": {
        "task_1234567890": {
            "task_id": "task_1234567890",
            "agent_id": "research_agent",
            "description": "Research quantum computing",
            "status": "completed",
        }
    },
}

# Example for "no agent needed" scenario
example_general_conversation = {
    "decision_rationale": [
        {
            "timestamp": "2025-11-18T18:30:00",
            "node": "orchestrator",
            "analysis": {
                "user_message": "Hello, how are you?",
                "identified_task_type": None,
            },
            "decisions": {"action": "complete", "agent_needed": False},
            "reasoning": {
                "action": "User message appears to be general conversation, not requiring specialized agent",
                "task_analysis": "No specific task keywords detected (research, code, analysis, data)",
            },
        }
    ],
    "orchestration_reasoning": {
        "decisions": {"agent_needed": False},
        "reasoning": {"action": "General conversation - no specialized agent required"},
    },
}

# Example for "gathering information" scenario
example_info_gathering = {
    "decision_rationale": [
        {
            "timestamp": "2025-11-18T18:30:00",
            "node": "orchestrator",
            "analysis": {
                "user_message": "Write code for a REST API",
                "identified_task_type": "code",
                "tool_allocation": {
                    "task_type": "code",
                    "tools_allocated": ["python_repl", "code_interpreter"],
                    "allocation_strategy": "task-based (not persona-based)",
                },
                "required_information": ["language", "requirements", "constraints"],
            },
            "decisions": {
                "agent_needed": True,
                "selected_task_type": "code",
                "agent_created": True,
                "agent_id": "code_agent",
                "task_created": True,
                "action": "gather_info",
            },
            "reasoning": {
                "agent_selection": "Created 'code' agent (ID: code_agent) because task type 'code' matches this agent's capabilities: ['code', 'programming', 'implementation', 'debugging']",
                "tool_selection": "Selected tools relevant to 'code' tasks based on tool capability mappings",
                "action": "Agent requires additional information before execution: ['language', 'requirements', 'constraints']",
            },
        }
    ]
}

print("✅ Decision Rationale Structure:")
print("=" * 60)
print("\n1. TASK ANALYSIS - What the user needs")
print("2. AGENT SELECTION - Why this agent was chosen")
print("3. TOOL ALLOCATION - Which tools and why")
print("4. ROUTING DECISION - What happens next and why")
print("\n" + "=" * 60)
