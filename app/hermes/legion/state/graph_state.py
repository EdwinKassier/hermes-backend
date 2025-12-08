"""State models for LangGraph-based orchestration."""

import operator
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field
from typing_extensions import Annotated, NotRequired

from ..models import RequiredInfoField


def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Reducer to merge dictionaries (shallow)."""
    return {**a, **b}


def deep_merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge reducer for nested dictionaries.

    Recursively merges nested dicts to prevent data loss when parallel
    workers update different parts of the same nested structure.
    """
    result = {**a}
    for key, value in b.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


class TaskStatus(str, Enum):
    """Status of a task in the ledger."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    AWAITING_INPUT = "awaiting_input"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskInfo(BaseModel):
    """Information about a task in the task ledger."""

    task_id: str = Field(..., description="Unique task identifier")
    agent_id: str = Field(..., description="Agent handling this task")
    description: str = Field(..., description="Task description")
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    dependencies: List[str] = Field(
        default_factory=list, description="Task IDs this task depends on"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[str] = None
    error: Optional[str] = None

    # Judge Feedback
    judge_feedback: Optional[str] = None
    judge_criteria: Optional[str] = None  # Criteria used for evaluation
    retry_count: int = 0
    max_retries: int = 3
    judgment_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="History of judge evaluations"
    )

    metadata: Dict[str, Any] = Field(default_factory=dict)

    def is_ready(self, ledger: Dict[str, "TaskInfo"]) -> bool:
        """Check if all dependencies are completed."""
        for dep_id in self.dependencies:
            dep = ledger.get(dep_id)
            if not dep or dep.status != TaskStatus.COMPLETED:
                return False
        return True

    def is_terminal(self) -> bool:
        """Check if task is in a terminal state."""
        return self.status in [
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        ]


class AgentInfo(BaseModel):
    """Information about a dynamically created agent."""

    agent_id: str = Field(..., description="Agent identifier")
    agent_type: str = Field(..., description="Type of agent (e.g., 'research', 'code')")
    tools: List[str] = Field(
        default_factory=list, description="Tool names allocated to this agent"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Message(TypedDict):
    """Message in conversation."""

    role: str
    content: str
    timestamp: str
    metadata: Dict[str, Any]


class OrchestratorState(TypedDict):
    """
    Central state for the orchestration graph.

    This is the shared state passed between all nodes in the StateGraph.
    """

    # Messages with accumulation (append-only)
    messages: Annotated[List[Dict[str, Any]], operator.add]

    # Task management
    task_ledger: NotRequired[
        Annotated[Dict[str, TaskInfo], merge_dicts]
    ]  # task_id -> TaskInfo

    # Agent management
    agents: NotRequired[Dict[str, AgentInfo]]  # agent_id -> AgentInfo
    tool_allocations: NotRequired[
        Dict[str, List[str]]
    ]  # agent_id -> tool names (not objects, for serialization)

    # User context
    user_id: str
    persona: str

    # Current execution context
    current_agent_id: NotRequired[Optional[str]]
    current_task_id: NotRequired[Optional[str]]
    next_action: NotRequired[str]  # GraphDecision or routing target

    # Information gathering state
    required_info: NotRequired[Dict[str, "RequiredInfoField"]]
    collected_info: NotRequired[Dict[str, Any]]
    pending_questions: NotRequired[List[str]]

    # Decision tracking and explainability
    decision_rationale: NotRequired[
        List[Dict[str, Any]]
    ]  # History of orchestrator decisions

    # Unified Legion Swarm State
    legion_strategy: NotRequired[Literal["council", "parallel", "intelligent"]]
    # Use deep_merge_dicts reducer to allow parallel workers to update results concurrently
    # Deep merge prevents data loss in nested structures
    legion_results: NotRequired[Annotated[Dict[str, Any], deep_merge_dicts]]

    # Conversation memory management
    conversation_summaries: NotRequired[
        List[Dict[str, Any]]
    ]  # Stored conversation summaries

    # Execution level tracking for dependency-aware sequential execution
    current_execution_level: NotRequired[
        int
    ]  # Which level is currently being processed (0-indexed)
    total_execution_levels: NotRequired[int]  # Total number of execution levels
    level_results: NotRequired[
        Dict[int, Dict[str, Any]]
    ]  # Results stored per execution level
    fail_on_level_error: NotRequired[
        bool
    ]  # If True, stop on any worker failure; if False, continue with partial results

    # Conversation continuation state
    awaiting_user_response: NotRequired[
        bool
    ]  # If True, pause and wait for next user message
    conversation_complete: NotRequired[bool]  # If True, conversation is finished

    # Metadata (execution metrics, etc.)
    metadata: NotRequired[Dict[str, Any]]
    execution_path: NotRequired[
        Annotated[List[Dict[str, Any]], operator.add]
    ]  # Track execution flow


class GraphDecision(str, Enum):
    """Decision made by the orchestrator."""

    ROUTE_TO_AGENT = "route_to_agent"
    GATHER_INFO = "gather_info"
    EXECUTE_AGENT = "execute_agent"
    REPLAN = "replan"  # NEW: Re-evaluate task when user changes topic
    COMPLETE = "complete"
    ERROR = "error"


class AgentConfig(BaseModel):
    """Configuration for creating an agent."""

    agent_type: str = Field(..., description="Type of agent (research, code, analysis)")
    required_tools: List[str] = Field(
        default_factory=list, description="Required tool names"
    )
    optional_tools: List[str] = Field(
        default_factory=list, description="Optional tool names"
    )
    max_iterations: int = Field(default=5, description="Maximum execution iterations")
    metadata: Dict[str, Any] = Field(default_factory=dict)
