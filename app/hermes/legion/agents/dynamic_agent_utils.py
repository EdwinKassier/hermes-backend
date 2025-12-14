"""Utilities for creating and configuring dynamic agents."""

import logging
from typing import Any, Dict, List, Optional

from .factory import AgentFactory

logger = logging.getLogger(__name__)


def create_worker_plan_with_dynamic_agents(
    task_description: str,
    agent_configs: List[Dict[str, Any]],
    execution_levels: List[int] = None,
) -> List[Dict[str, Any]]:
    """
    Create a worker plan using dynamic agent configurations.

    Args:
        task_description: The task to be performed
        agent_configs: List of dynamic agent configurations
        execution_levels: Optional execution levels for each agent

    Returns:
        Worker plan suitable for Legion orchestration
    """
    workers = []

    for i, config in enumerate(agent_configs):
        worker = {
            "worker_id": f"dynamic_worker_{i+1}",
            "role": config.get("agent_id", f"dynamic_agent_{i+1}"),
            "task_description": task_description,
            "tools": [],  # Tools specified by dynamic agent configuration
            "dynamic_agent_config": config,  # This tells the worker to create a dynamic agent
            "metadata": {
                "agent_type": config.get("agent_type", "dynamic_agent"),
                "capabilities": config.get("capabilities", {}),
                "persona": config.get("persona", "dynamic_agent"),
                "task_types": config.get("task_types", []),
                "specialization": config.get("task_portion", ""),
                "created_by": "TaskAgentPlanner",
                "creation_timestamp": None,  # Will be set when worker is created
            },
        }

        if execution_levels and i < len(execution_levels):
            worker["execution_level"] = execution_levels[i]
        else:
            worker["execution_level"] = 0

        workers.append(worker)

    logger.info(f"Created worker plan with {len(workers)} dynamic agents")
    return workers


def create_mixed_agent_plan(
    task_description: str, agent_configs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Create a worker plan with mixed agent types from dynamic configurations.

    Args:
        task_description: The task to perform
        agent_configs: List of complete agent configurations, each with:
            - agent_id: Agent ID
            - agent_type: Custom agent type name
            - task_types: List of task types
            - capabilities: Agent capabilities
            - prompts: Complete prompt templates
            - persona: AI persona

    Returns:
        Worker plan with mixed agent types
    """
    workers = create_worker_plan_with_dynamic_agents(
        task_description=task_description, agent_configs=agent_configs
    )

    return workers


# Example usage functions
def example_dynamic_mixed_team():
    """Example of creating a completely custom mixed team for a complex task."""
    # This would typically come from TaskAgentPlanner, but here's a manual example
    custom_agent_configs = [
        {
            "agent_id": "neural_network_architect",
            "agent_type": "deep_learning_specialist",
            "task_types": ["model_design", "architecture", "training"],
            "capabilities": {
                "primary_focus": "neural network design and optimization",
                "tools_needed": ["tensorflow", "pytorch", "cuda", "model_analysis"],
                "expertise_level": "expert",
                "specializations": ["computer_vision", "natural_language_processing"],
                "knowledge_domains": [
                    "deep_learning",
                    "machine_learning",
                    "computer_vision",
                ],
            },
            "prompts": {
                "identify_required_info": """You are a deep learning architect analyzing neural network requirements.

Task: "{task}"
User Message: "{user_message}"

Analyze what neural network architecture information is needed:
- Model type (CNN, RNN, Transformer, etc.)
- Input/output specifications
- Performance requirements
- Dataset characteristics
- Hardware constraints

Response format (JSON):
{{
  "needs_info": true|false,
  "inferred_values": {{"model_type": "cnn", "input_shape": [224,224,3]}},
  "required_fields": [
    {{
      "field_name": "dataset_size",
      "field_type": "string",
      "question": "What's the size of your training dataset?",
      "description": "Number of training samples"
    }}
  ],
  "reasoning": "why you need this information"
}}""",
                "execute_task": """You are a neural network architect designing and implementing deep learning models.

Task: {task}
{judge_feedback}

Your capabilities: {capabilities}
Available tools: {tool_context}

Design and implement a neural network solution:

1. **Architecture Design**: Specify the complete model architecture
2. **Implementation**: Provide working code with proper structure
3. **Training Strategy**: Define training approach and hyperparameters
4. **Optimization**: Include performance optimizations and best practices
5. **Validation**: Provide evaluation metrics and testing approach

Use your expertise in {capabilities} to create a production-ready solution.""",
            },
            "persona": "technical_expert",
            "task_portion": "Design and implement the neural network architecture",
            "dependencies": [],
        },
        {
            "agent_id": "data_pipeline_engineer",
            "agent_type": "data_engineering_specialist",
            "task_types": ["data_processing", "pipeline_design", "etl"],
            "capabilities": {
                "primary_focus": "data pipeline design and optimization",
                "tools_needed": [
                    "apache_spark",
                    "kafka",
                    "airflow",
                    "postgresql",
                    "redis",
                ],
                "expertise_level": "expert",
                "specializations": [
                    "real_time_processing",
                    "batch_processing",
                    "data_quality",
                ],
                "knowledge_domains": [
                    "data_engineering",
                    "distributed_systems",
                    "database_design",
                ],
            },
            "prompts": {
                "identify_required_info": """You are a data pipeline engineer analyzing data processing requirements.

Task: "{task}"
User Message: "{user_message}"

Determine what data pipeline information is needed:
- Data volume and velocity
- Source and destination systems
- Real-time vs batch requirements
- Data quality and validation needs

Response format (JSON):
{{
  "needs_info": true|false,
  "inferred_values": {{"data_volume": "millions", "processing_type": "batch"}},
  "required_fields": [...],
  "reasoning": "why you need this information"
}}""",
                "execute_task": """You are a data pipeline engineer designing scalable data processing solutions.

Task: {task}
{judge_feedback}

Your capabilities: {capabilities}
Available tools: {tool_context}

Design and implement a complete data pipeline:

1. **Architecture**: Design the data flow and processing pipeline
2. **Implementation**: Provide working code and configurations
3. **Scalability**: Ensure the solution scales with data volume
4. **Reliability**: Include error handling and monitoring
5. **Performance**: Optimize for throughput and latency requirements

Create a production-ready data processing solution.""",
            },
            "persona": "system_architect",
            "task_portion": "Design and implement the data processing pipeline",
            "dependencies": [],
        },
        {
            "agent_id": "frontend_visualization_specialist",
            "agent_type": "data_visualization_expert",
            "task_types": ["visualization", "ui_design", "dashboard"],
            "capabilities": {
                "primary_focus": "creating interactive data visualizations and dashboards",
                "tools_needed": [
                    "d3.js",
                    "react",
                    "typescript",
                    "webgl",
                    "chart_libraries",
                ],
                "expertise_level": "expert",
                "specializations": [
                    "interactive_charts",
                    "real_time_dashboards",
                    "data_storytelling",
                ],
                "knowledge_domains": [
                    "data_visualization",
                    "ui_ux",
                    "frontend_architecture",
                ],
            },
            "prompts": {
                "identify_required_info": """You are a data visualization specialist analyzing dashboard requirements.

Task: "{task}"
User Message: "{user_message}"

Determine what visualization information is needed:
- Data types to visualize
- User audience and use cases
- Real-time vs static requirements
- Device/platform constraints

Response format (JSON):
{{
  "needs_info": true|false,
  "inferred_values": {{"chart_types": ["line", "bar"], "audience": "business_users"}},
  "required_fields": [...],
  "reasoning": "why you need this information"
}}""",
                "execute_task": """You are a data visualization specialist creating interactive dashboards.

Task: {task}
{judge_feedback}

Your capabilities: {capabilities}
Available tools: {tool_context}

Create a complete visualization solution:

1. **Design**: Plan the dashboard layout and chart types
2. **Implementation**: Provide working frontend code
3. **Interactivity**: Add interactive features and user controls
4. **Responsive**: Ensure works across devices and screen sizes
5. **Performance**: Optimize rendering and data loading

Create an engaging and informative data visualization experience.""",
            },
            "persona": "creative_designer",
            "task_portion": "Design and implement the data visualization dashboard",
            "dependencies": ["data_pipeline_engineer"],
        },
    ]

    return create_mixed_agent_plan(
        task_description="Build an end-to-end ML dashboard with real-time data processing and interactive visualizations",
        agent_configs=custom_agent_configs,
    )
