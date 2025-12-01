from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from app.hermes.legion.nodes.judge_node import judge_node
from app.hermes.legion.state import (
    GraphDecision,
    OrchestratorState,
    TaskInfo,
    TaskStatus,
)


@pytest.fixture
def mock_gemini_service():
    with patch("app.hermes.legion.nodes.judge_node.get_gemini_service") as mock:
        yield mock


@pytest.mark.asyncio
async def test_judge_node_approval(mock_gemini_service):
    """Test that the judge approves valid output."""
    # Setup state
    task_id = "task_1"
    agent_id = "agent_1"

    task_info = TaskInfo(
        task_id=task_id,
        agent_id=agent_id,
        description="Write a poem",
        status=TaskStatus.COMPLETED,
        result="Roses are red...",
        retry_count=0,
    )

    state = {
        "current_task_id": task_id,
        "current_agent_id": agent_id,
        "task_ledger": {task_id: task_info},
        "decision_rationale": [],
    }

    # Mock Gemini response for approval
    mock_service_instance = mock_gemini_service.return_value
    mock_service_instance.generate_gemini_response.return_value = """
    {
        "is_clarification": false,
        "is_valid": true,
        "score": 0.9,
        "feedback": "Good job",
        "reasoning": "Met requirements"
    }
    """

    # Execute node
    result = await judge_node(state)

    # Verify
    assert result["next_action"] == "general_response"
    # Should update ledger with history even on success
    assert "task_ledger" in result
    updated_task = result["task_ledger"][task_id]
    assert len(updated_task.judgment_history) == 1
    assert updated_task.judgment_history[0]["score"] == 0.9
    assert updated_task.judgment_history[0]["is_valid"] is True


@pytest.mark.asyncio
async def test_judge_node_rejection_retry(mock_gemini_service):
    """Test that the judge rejects invalid output and triggers retry."""
    # Setup state
    task_id = "task_1"
    agent_id = "agent_1"

    task_info = TaskInfo(
        task_id=task_id,
        agent_id=agent_id,
        description="Write a python script",
        status=TaskStatus.COMPLETED,
        result="Here is some java code...",
        retry_count=0,
        max_retries=3,
    )

    state = {
        "current_task_id": task_id,
        "current_agent_id": agent_id,
        "task_ledger": {task_id: task_info},
        "decision_rationale": [],
    }

    # Mock Gemini response for rejection
    mock_service_instance = mock_gemini_service.return_value
    mock_service_instance.generate_gemini_response.return_value = """
    {
        "is_clarification": false,
        "is_valid": false,
        "score": 0.4,
        "feedback": "Requested Python but got Java",
        "reasoning": "Wrong language"
    }
    """

    # Execute node
    result = await judge_node(state)

    # Verify
    assert result["next_action"] == "agent_executor"
    assert "task_ledger" in result
    updated_task = result["task_ledger"][task_id]

    assert updated_task.status == TaskStatus.IN_PROGRESS
    assert updated_task.retry_count == 1
    assert updated_task.judge_feedback == "Requested Python but got Java"
    assert updated_task.result is None
    assert len(updated_task.judgment_history) == 1
    assert updated_task.judgment_history[0]["is_valid"] is False


@pytest.mark.asyncio
async def test_judge_node_max_retries(mock_gemini_service):
    """Test that the judge accepts best effort after max retries."""
    # Setup state
    task_id = "task_1"
    agent_id = "agent_1"

    task_info = TaskInfo(
        task_id=task_id,
        agent_id=agent_id,
        description="Write a python script",
        status=TaskStatus.COMPLETED,
        result="Still Java code...",
        retry_count=3,  # Max retries reached
        max_retries=3,
    )

    state = {
        "current_task_id": task_id,
        "current_agent_id": agent_id,
        "task_ledger": {task_id: task_info},
        "decision_rationale": [],
    }

    # Mock Gemini response for rejection
    mock_service_instance = mock_gemini_service.return_value
    mock_service_instance.generate_gemini_response.return_value = """
    {
        "is_clarification": false,
        "is_valid": false,
        "score": 0.4,
        "feedback": "Still wrong language",
        "reasoning": "Wrong language"
    }
    """

    # Execute node
    result = await judge_node(state)

    # Verify
    assert result["next_action"] == "general_response"
    # Should update ledger with history even on failure acceptance
    assert "task_ledger" in result
    updated_task = result["task_ledger"][task_id]
    assert len(updated_task.judgment_history) == 1
    assert updated_task.judgment_history[0]["is_valid"] is False


@pytest.mark.asyncio
async def test_judge_node_clarification(mock_gemini_service):
    """Test that the judge passes through clarification requests."""
    # Setup state
    task_id = "task_1"
    agent_id = "agent_1"

    task_info = TaskInfo(
        task_id=task_id,
        agent_id=agent_id,
        description="Research something",
        status=TaskStatus.COMPLETED,
        result="I need to know the time period.",
        retry_count=0,
    )

    state = {
        "current_task_id": task_id,
        "current_agent_id": agent_id,
        "task_ledger": {task_id: task_info},
        "decision_rationale": [],
    }

    # Mock Gemini response for clarification
    mock_service_instance = mock_gemini_service.return_value
    mock_service_instance.generate_gemini_response.return_value = """
    {
        "is_clarification": true,
        "is_valid": false,
        "score": 0.0,
        "feedback": "",
        "reasoning": "Asking for info"
    }
    """

    # Execute node
    result = await judge_node(state)

    # Verify
    assert result["next_action"] == "general_response"
    # Should NOT update ledger to retry
    assert "task_ledger" not in result


@pytest.mark.asyncio
async def test_judge_custom_strictness(mock_gemini_service):
    """Test that the judge respects custom strictness threshold."""
    # Setup state
    task_id = "task_1"
    agent_id = "agent_1"

    task_info = TaskInfo(
        task_id=task_id,
        agent_id=agent_id,
        description="Write a poem",
        status=TaskStatus.COMPLETED,
        result="Roses are red...",
        retry_count=0,
        metadata={"judge_strictness": 0.5},  # Lower threshold
    )

    state = {
        "current_task_id": task_id,
        "current_agent_id": agent_id,
        "task_ledger": {task_id: task_info},
        "decision_rationale": [],
    }

    # Mock Gemini response with score 0.6 (would fail default 0.7, but pass 0.5)
    mock_service_instance = mock_gemini_service.return_value
    mock_service_instance.generate_gemini_response.return_value = """
    {
        "is_clarification": false,
        "is_valid": true,
        "score": 0.6,
        "feedback": "Okay job",
        "reasoning": "Met minimum requirements"
    }
    """

    # Execute node
    result = await judge_node(state)

    # Verify
    assert result["next_action"] == "general_response"  # Should be accepted
    updated_task = result["task_ledger"][task_id]
    assert updated_task.judgment_history[0]["score"] == 0.6
    assert updated_task.judgment_history[0]["is_valid"] is True


@pytest.mark.asyncio
async def test_judge_custom_persona(mock_gemini_service):
    """Test that the judge uses the custom persona."""
    # Setup state
    task_id = "task_1"
    agent_id = "agent_1"

    task_info = TaskInfo(
        task_id=task_id,
        agent_id=agent_id,
        description="Write a poem",
        status=TaskStatus.COMPLETED,
        result="Roses are red...",
        retry_count=0,
        metadata={"judge_persona": "poet_laureate"},
    )

    state = {
        "current_task_id": task_id,
        "current_agent_id": agent_id,
        "task_ledger": {task_id: task_info},
        "decision_rationale": [],
    }

    # Mock Gemini response
    mock_service_instance = mock_gemini_service.return_value
    mock_service_instance.generate_gemini_response.return_value = """
    {
        "is_clarification": false,
        "is_valid": true,
        "score": 0.9,
        "feedback": "Beautiful",
        "reasoning": "Artistic"
    }
    """

    # Execute node
    await judge_node(state)

    # Verify call arguments
    call_args = mock_service_instance.generate_gemini_response.call_args
    assert call_args.kwargs["persona"] == "poet_laureate"
