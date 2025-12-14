# Dynamic Agent System

The Legion system now supports **task-driven dynamic agent creation** without hardcoded agent types. The system analyzes any task and creates whatever combination of agents is needed to complete it effectively.

## Key Benefits

- ✅ **Task-Driven Creation**: System analyzes tasks and creates optimal agent combinations
- ✅ **No Hardcoded Types**: Any agent type can be created dynamically
- ✅ **Flexible Combinations**: Mix research, coding, analysis, creative agents as needed
- ✅ **Multiple Implementations**: Create different approaches for the same agent type
- ✅ **Template-Based**: Use reusable templates with custom configurations
- ✅ **Backward Compatible**: Existing hardcoded agents continue to work

## Architecture Overview

### Components

1. **`DynamicAgent`**: Flexible agent class that accepts any configuration
2. **`TaskAgentPlanner`**: Analyzes tasks and invents optimal agent combinations
3. **`AgentFactory`**: Factory methods for creating dynamic agents
4. **Dynamic Agent Utils**: Helper functions for creating agent plans

### True Dynamic Creation Process

```
1. Task Analysis → 2. Agent Invention → 3. Complete Configuration → 4. Execution
     ↓                    ↓                    ↓                    ↓
"Build web app" → "Invent: quantum_architect + data_flow_engineer + viz_craftsman" → Configure from scratch → Legion orchestration
```

The system analyzes any task and **completely invents** new agent types with full capabilities, prompts, and personas from scratch - no predefined templates or hardcoded agent types.

## True Dynamic Agent Creation

### Task-Driven Agent Invention

The system can analyze any task and invent completely new agent types:

```python
from app.hermes.legion.agents import TaskAgentPlanner

# The system analyzes ANY task and creates whatever agents are needed
planner = TaskAgentPlanner()

# Example: Complex ML dashboard task
analysis = planner.analyze_task_and_plan_agents(
    "Build an end-to-end ML dashboard with real-time data processing and interactive visualizations"
)

# System invents and creates agents like:
# - "neural_network_architect" (deep learning specialist)
# - "data_pipeline_engineer" (data engineering specialist)
# - "frontend_visualization_specialist" (data visualization expert)

worker_plan = planner.create_worker_plan_from_analysis(analysis, task)
```

### Manual Dynamic Agent Creation

When you want to manually specify agent configurations:

```python
from app.hermes.legion.agents import AgentFactory

# Create a completely custom agent from scratch
agent = AgentFactory.create_dynamic_agent(
    agent_id="quantum_algorithm_designer",
    task_types=["quantum_computing", "algorithm_design", "optimization"],
    capabilities={
        "primary_focus": "designing quantum algorithms and circuits",
        "tools_needed": ["qiskit", "cirq", "quantum_simulators", "complexity_analysis"],
        "expertise_level": "expert",
        "specializations": ["quantum_machine_learning", "quantum_optimization", "error_correction"],
        "knowledge_domains": ["quantum_computing", "quantum_information", "algorithm_design"]
    },
    prompts={
        "identify_required_info": """Analyze quantum algorithm requirements...

Task: "{task}"
User Message: "{user_message}"

Determine what quantum computing information is needed:
- Problem type (optimization, simulation, machine learning)
- Qubit requirements and constraints
- Available quantum hardware
- Classical preprocessing needs

Response format (JSON):
{{
  "needs_info": true|false,
  "inferred_values": {{"problem_type": "optimization", "qubits_needed": 50}},
  "required_fields": [...],
  "reasoning": "why you need this information"
}}""",
        "execute_task": """Design and implement quantum algorithms...

Task: {task}
{judge_feedback}

Design a complete quantum solution with circuits, classical preprocessing, and analysis."""
    },
    persona="quantum_physicist"
)
```

## Advanced Usage

### Creating Worker Plans

```python
from app.hermes.legion.agents import create_multiple_coding_agents_plan

# Create a plan for multiple coding agents
worker_plan = create_multiple_coding_agents_plan(
    task_description="Build a REST API with database integration",
    count=2,
    specializations=["backend_api", "database_layer"],
    languages=["python"],
    framework=["fastapi", "sqlalchemy"]
)

# Use in Legion orchestration
# The worker plan can be passed to legion_orchestrator_node
```

### Mixed Agent Teams

```python
from app.hermes.legion.agents import create_mixed_agent_plan

# Create a team with different agent types
worker_plan = create_mixed_agent_plan(
    task_description="Build a data visualization dashboard",
    agent_specs=[
        {
            "template": "code_generator",
            "agent_id": "backend_engineer",
            "config": {"languages": ["python"], "focus": "api"}
        },
        {
            "template": "data_analyst",
            "agent_id": "data_processor",
            "config": {"focus": "data_transformation"}
        },
        {
            "template": "creative_coder",
            "agent_id": "frontend_artist",
            "config": {"languages": ["javascript"], "focus": "ui_ux"}
        }
    ]
)
```

## Agent Invention Examples

The system can invent any agent type needed for any task:

### Research & Analysis Tasks
- **quantum_cryptography_researcher**: Expert in quantum-resistant cryptographic methods
- **blockchain_economics_analyst**: Analyzes economic implications of blockchain systems
- **neuroscience_data_interpreter**: Interprets brain imaging and neural data patterns

### Creative & Design Tasks
- **interactive_narrative_designer**: Creates branching story experiences and user journeys
- **sustainable_architecture_planner**: Designs eco-friendly building systems and materials
- **multimodal_art_synthesizer**: Combines visual, audio, and interactive art forms

### Technical & Engineering Tasks
- **distributed_systems_architect**: Designs scalable, fault-tolerant distributed systems
- **ai_safety_engineer**: Develops safe and aligned artificial intelligence systems
- **cyber_physical_system_integrator**: Integrates software with physical hardware systems

### Business & Strategy Tasks
- **market_prediction_modeler**: Builds economic forecasting and trend analysis models
- **organizational_change_facilitator**: Guides companies through digital transformation
- **ethical_ai_policy_developer**: Creates frameworks for responsible AI development

## Complete Agent Definition

Each agent is defined completely from scratch with:

```python
custom_agent_config = {
    "agent_id": "custom_agent_name",
    "agent_type": "completely_invented_type_name",
    "task_types": ["specific_task_types"],
    "capabilities": {
        "primary_focus": "core_responsibility",
        "tools_needed": ["required_tools"],
        "expertise_level": "expert_level",
        "specializations": ["specific_areas"],
        "knowledge_domains": ["knowledge_areas"]
    },
    "prompts": {
        "identify_required_info": "Complete prompt for gathering requirements...",
        "execute_task": "Complete prompt for task execution..."
    },
    "persona": "appropriate_ai_persona"
}
```

## Integration with Legion

The dynamic agent system integrates seamlessly with the existing Legion orchestration:

1. **Strategy Layer**: Strategies can create worker plans with dynamic agents
2. **Orchestrator**: Detects `dynamic_agent_config` in worker plans and creates dynamic agents
3. **Worker Execution**: Dynamic agents work identically to hardcoded agents

### Worker Plan Format

```python
worker = {
    "worker_id": "dynamic_worker_1",
    "role": "custom_coder",
    "task_description": "Build a feature",
    "tools": [],
    "dynamic_agent_config": {  # This triggers dynamic agent creation
        "agent_id": "feature_builder",
        "task_types": ["code"],
        "capabilities": {...},
        "prompts": {...},
        "persona": "creative"
    }
}
```

## Migration Guide

### Complete Agent Invention

**The system now creates agents from scratch for any task:**

```python
# Example: Complex quantum computing task
analysis = TaskAgentPlanner().analyze_task_and_plan_agents(
    "Design a quantum algorithm for portfolio optimization with classical preprocessing"
)

# System invents completely new agent types:
# - "quantum_finance_mathematician": Specializes in financial math + quantum algorithms
# - "classical_preprocessing_architect": Handles classical-quantum hybrid computation
# - "quantum_hardware_optimizer": Optimizes for specific quantum hardware constraints

worker_plan = TaskAgentPlanner().create_worker_plan_from_analysis(analysis, task)
```

### No Predefined Limitations

**Before:** Limited to `CodeAgent`, `DataAgent`, `ResearchAgent`, etc.

**After:** Complete freedom to invent any agent type needed:
- `quantum_cryptography_researcher`
- `sustainable_energy_optimizer`
- `neuroscience_data_interpreter`
- `blockchain_economics_analyst`
- `cyber_physical_system_integrator`
- Any custom agent type imaginable

## Task-Driven Agent Creation

The real power of the system is its ability to analyze any task and create whatever agents are needed.

### Task Analysis Examples

#### Complex Web Application Task
```
Task: "Build a full-stack web application for managing personal finances with data visualization and expense tracking"

Analysis Result:
├── Research Agent: Investigate finance APIs and best practices
├── Backend Code Agent: Build API with database integration
├── Data Analysis Agent: Process and analyze financial data
└── Frontend Creative Agent: Design interactive charts and UI
```

#### Research Project Task
```
Task: "Research quantum computing developments and their cryptography impact"

Analysis Result:
├── Research Specialist: Gather latest quantum computing papers
├── Analytical Expert: Analyze cryptography implications
└── Creative Coder: Build educational visualizations
```

#### Data Analysis Task
```
Task: "Analyze sales data, identify trends, create dashboard with recommendations"

Analysis Result:
├── Data Analyst: Process and clean sales data
├── Analytical Expert: Identify trends and patterns
└── Creative Coder: Build interactive dashboard
```

### Usage: Task-Driven Creation

```python
from app.hermes.legion.agents.task_agent_planner import TaskAgentPlanner

# The system analyzes ANY task and creates optimal agents
planner = TaskAgentPlanner()

# Complex task automatically gets appropriate agents
analysis = planner.analyze_task_and_plan_agents(
    "Build a machine learning model to predict customer churn with a web dashboard"
)

worker_plan = planner.create_worker_plan_from_analysis(analysis, task)

# Result: Creates ML Engineer + Data Analyst + Frontend Developer agents
```

### Examples

#### Example 1: Multiple Coding Approaches (When Specifically Needed)

```python
from app.hermes.legion.agents import create_multiple_coding_agents_plan

# Only when you specifically want multiple coding approaches
plan = create_multiple_coding_agents_plan(
    task_description="Implement a sorting algorithm with different approaches",
    count=3,
    specializations=["pythonic", "performant", "readable"],
    style=["idiomatic", "optimized", "well_documented"]
)
```

#### Example 2: Task-Driven Agent Invention

```python
from app.hermes.legion.agents.task_agent_planner import TaskAgentPlanner

# System analyzes ANY task and invents the needed agent types
planner = TaskAgentPlanner()

# Example: Complex quantum computing task
analysis = planner.analyze_task_and_plan_agents(
    "Design a quantum algorithm for portfolio optimization with classical preprocessing"
)

# System invents completely new agent types:
# - "quantum_finance_mathematician": Specializes in financial math + quantum algorithms
# - "classical_preprocessing_architect": Handles classical-quantum hybrid computation
# - "quantum_hardware_optimizer": Optimizes for specific quantum hardware constraints

worker_plan = planner.create_worker_plan_from_analysis(analysis, task)
```

## Backward Compatibility

The system maintains full backward compatibility:

- Existing hardcoded agents (`CodeAgent`, `DataAgent`, etc.) continue to work
- Legacy agent creation through `AgentFactory.create_agent()` is unchanged
- Worker plans without `dynamic_agent_config` use legacy agent mapping

## Best Practices

1. **Use Templates**: Start with templates and customize as needed
2. **Descriptive IDs**: Use clear, descriptive agent IDs
3. **Capability Definition**: Clearly define what each agent can do
4. **Prompt Customization**: Adapt prompts to specific use cases
5. **Persona Selection**: Choose personas that match agent roles
6. **Testing**: Test agent configurations before production use

## Future Extensions

- **Agent Evolution**: Dynamic agents can learn and improve over time
- **Capability Discovery**: Agents can dynamically discover new capabilities
- **Agent Composition**: Combine multiple agents into composite agents
- **Meta-Agent Creation**: Agents that can design and create other agents
- **Domain-Specific Libraries**: Reusable capability and prompt libraries for specific domains
