"""Legion sub-agent models - Following Pydantic patterns from hermes/models.py."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SubAgentStatus(str, Enum):
    """Status of a sub-agent task."""

    CREATED = "created"
    AWAITING_USER_INPUT = "awaiting_user_input"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


class RequiredInfoField(BaseModel):
    """Schema for a required information field."""

    field_name: str = Field(..., description="Field identifier")
    field_type: str = Field(
        ..., description="Field type: string, list, enum, number, boolean"
    )
    question: str = Field(..., description="Question to ask the user")
    description: Optional[str] = Field(default=None, description="Field description")
    options: Optional[List[str]] = Field(
        default=None, description="Options for enum types"
    )
    validation: Optional[Dict[str, Any]] = Field(
        default=None, description="Validation rules"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "field_name": "time_period",
                "field_type": "string",
                "question": "What time period are you interested in?",
                "description": "Time period for research",
            }
        }
    }


class SubAgentState(BaseModel):
    """Enhanced sub-agent state structure."""

    agent_id: str = Field(..., description="Sub-agent identifier")
    status: SubAgentStatus = Field(
        default=SubAgentStatus.CREATED, description="Current status"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    # Task information
    task: str = Field(..., description="Task description")
    task_type: str = Field(..., description="Task type: research, code, analysis, etc.")
    triggering_message: str = Field(..., description="Original user message")

    # Information gathering
    required_info: Dict[str, RequiredInfoField] = Field(default_factory=dict)
    collected_info: Dict[str, Any] = Field(default_factory=dict)
    pending_questions: List[str] = Field(default_factory=list)

    # Execution
    result: Optional[str] = Field(default=None, description="Task result")
    error_message: Optional[str] = Field(
        default=None, description="Error message if failed"
    )

    # Context linking (use conversation index, not IDs)
    conversation_message_index: Optional[int] = Field(
        default=None, description="Index of triggering message in conversation"
    )

    # Judge Feedback
    judge_feedback: Optional[str] = Field(
        default=None, description="Feedback from the LLM Judge"
    )
    retry_count: int = Field(default=0, description="Number of retries attempted")

    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = self.model_dump()
        # Convert enum to string for JSON serialization
        data["status"] = self.status.value
        # Convert RequiredInfoField objects to dicts
        if "required_info" in data:
            data["required_info"] = {
                k: v.model_dump() if isinstance(v, RequiredInfoField) else v
                for k, v in data["required_info"].items()
            }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubAgentState":
        """Create from dictionary."""
        # Handle enum conversion
        if "status" in data and isinstance(data["status"], str):
            try:
                data["status"] = SubAgentStatus(data["status"])
            except ValueError:
                # Default to CREATED if invalid status
                data["status"] = SubAgentStatus.CREATED

        # Handle RequiredInfoField conversion
        if "required_info" in data:
            required_info = {}
            for k, v in data["required_info"].items():
                if isinstance(v, dict):
                    try:
                        required_info[k] = RequiredInfoField(**v)
                    except Exception:
                        # Skip invalid fields
                        continue
                elif isinstance(v, RequiredInfoField):
                    required_info[k] = v
            data["required_info"] = required_info

        # Handle datetime strings
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "last_updated" in data and isinstance(data["last_updated"], str):
            data["last_updated"] = datetime.fromisoformat(data["last_updated"])

        return cls(**data)

    model_config = {
        "json_schema_extra": {
            "example": {
                "agent_id": "research_agent",
                "status": "awaiting_user_input",
                "task": "Research quantum computing",
                "task_type": "research",
                "triggering_message": "Research quantum computing",
            }
        }
    }


# --- Intelligence Models ---


class QueryComplexity(BaseModel):
    """Analysis of query complexity and requirements."""

    score: float = Field(..., description="Complexity score 0.0 to 1.0")
    dimensions: Dict[str, float] = Field(
        ..., description="Complexity dimensions e.g. {'technical': 0.8}"
    )
    suggested_workers: int = Field(..., description="Recommended number of workers")
    estimated_time_seconds: float = Field(..., description="Estimated execution time")


class Domain(str, Enum):
    """Knowledge domains for worker specialization."""

    RESEARCH = "research"
    CODING = "coding"
    DATA_ANALYSIS = "data_analysis"
    CREATIVE = "creative"
    PLANNING = "planning"
    GENERAL = "general"


class WorkerPlan(BaseModel):
    """Plan for a single worker in the swarm."""

    worker_id: str
    role: str
    specialization: str
    task_description: str
    tools: List[str]
    priority: int = Field(default=1, description="Execution priority")
    estimated_duration: float = Field(
        default=0.0, description="Estimated duration in seconds"
    )


class QualityMetrics(BaseModel):
    """Quality assessment of synthesis results."""

    completeness: float = Field(..., description="0.0 to 1.0")
    coherence: float
    relevance: float
    confidence: float
    agreement: float = Field(default=0.0, description="For council mode")
