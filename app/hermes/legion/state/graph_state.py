"""State models for LangGraph-based orchestration."""

import operator
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from ..models import RequiredInfoField


def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Reducer to merge dictionaries."""
    return {**a, **b}


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
    task_ledger: Dict[str, TaskInfo]  # task_id -> TaskInfo

    # Agent management
    agents: Dict[str, AgentInfo]  # agent_id -> AgentInfo
    tool_allocations: Dict[
        str, List[str]
    ]  # agent_id -> tool names (not objects, for serialization)

    # User context
    user_id: str
    persona: str

    # Current execution context
    current_agent_id: Optional[str]
    current_task_id: Optional[str]
    next_action: str  # GraphDecision or routing target

    # Information gathering state
    # Type hint uses forward reference to avoid circular import
    required_info: Dict[
        str, "RequiredInfoField"
    ]  # Proper type restored using TYPE_CHECKING
    collected_info: Dict[str, Any]
    pending_questions: List[str]

    # Decision tracking and explainability
    decision_rationale: List[Dict[str, Any]]  # History of orchestrator decisions

    # Unified Legion Swarm State
    legion_strategy: Literal["council", "parallel", "intelligent"]
    # Use merge_dicts reducer to allow parallel workers to update results concurrently
    legion_results: Annotated[Dict[str, Any], merge_dicts]

    # Conversation continuation state
    awaiting_user_response: bool  # NEW: If True, pause and wait for next user message
    conversation_complete: bool  # NEW: If True, conversation is finished

    # Metadata (execution metrics, etc.)
    metadata: Dict[str, Any]


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
