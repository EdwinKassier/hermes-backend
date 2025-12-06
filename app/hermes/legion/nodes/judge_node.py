"""
LLM Judge Node for Legion System.

This node evaluates the output of sub-agents to ensure quality and correctness.
It supports a feedback loop where rejected outputs are sent back to the agent
for improvement, and clarification requests are allowed to pass through.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, Optional

from app.shared.utils.service_loader import get_gemini_service

from ..state import GraphDecision, OrchestratorState, TaskInfo, TaskStatus

logger = logging.getLogger(__name__)


async def judge_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Evaluates the result of an agent's execution.

    Decides whether to:
    1. Approve the result (complete task)
    2. Reject the result (retry with feedback)
    3. Pass through clarification requests (bypass judgment)
    """
    logger.info("Judge node evaluating result")

    task_id = state.get("current_task_id")
    agent_id = state.get("current_agent_id")

    if not task_id or not agent_id:
        logger.error("No current task or agent for judgment")
        return {"next_action": GraphDecision.ERROR.value}

    task_ledger = state.get("task_ledger", {})
    task_info = task_ledger.get(task_id)

    if not task_info or not task_info.result:
        logger.error(f"Task {task_id} has no result to judge")
        return {"next_action": GraphDecision.ERROR.value}

    result = task_info.result
    user_request = task_info.description

    # Check for skip_judge flag
    if task_info.metadata.get("skip_judge", False):
        logger.info(f"Skipping judge for task {task_id} (requested via metadata)")
        return {
            "next_action": "general_response",
            "execution_path": [
                {
                    "node": "judge",
                    "timestamp": datetime.now().isoformat(),
                    "action": "skipped",
                }
            ],
        }

    # Get Gemini Service
    gemini_service = get_gemini_service()

    # Get strictness and persona from metadata (default to 0.7 and 'critic')
    strictness = task_info.metadata.get("judge_strictness", 0.7)
    persona = task_info.metadata.get("judge_persona", "critic")

    # Construct Judge Prompt with Dynamic Criteria Generation
    judge_prompt = f"""You are the Quality Assurance Judge for an AI system.

Your goal is to evaluate the output of a sub-agent to ensure it meets the user's requirements and quality standards.

**User Request**: "{user_request}"

**Agent Output**:
{result}

**Instructions**:
1. **Analyze the Request**: First, determine what constitutes a high-quality, correct, and complete answer for this specific request. What are the implicit and explicit requirements?
2. **Evaluate**: Compare the Agent Output against these requirements.
3. **Check for Clarification**: If the agent is asking for clarification or more information, this is VALID and should be passed to the user.

**Response Format (JSON)**:
{{
  "is_clarification": boolean, // True if agent is asking the user for info
  "is_valid": boolean, // True if the answer is acceptable
  "score": float, // 0.0 to 1.0
  "criteria": "string", // The specific requirements/criteria you used for evaluation
  "feedback": "string", // Required if is_valid is false. Be specific about what needs to be fixed.
  "reasoning": "string" // Explain why you gave this score based on the criteria
}}

Analyze the output and provide your judgment in JSON format.
"""

    try:
        # Get user_id from state for proper Langfuse tracing linkage
        user_id = state.get("user_id")

        # Call LLM
        # Note: temperature parameter not supported in generate_gemini_response
        # Persona config should handle temperature settings
        response = gemini_service.generate_gemini_response(
            prompt=judge_prompt,
            persona=persona,  # Use configured persona
            user_id=user_id,  # Pass user_id for Langfuse tracing
        )

        # Parse JSON
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            logger.error("Failed to parse judge response")
            # Fail open: assume valid if we can't judge
            return {"next_action": "general_response"}

        judgment = json.loads(json_match.group(0))

        is_clarification = judgment.get("is_clarification", False)
        is_valid = judgment.get("is_valid", False)
        feedback = judgment.get("feedback", "")
        score = judgment.get("score", 0.0)
        criteria = judgment.get("criteria", "Criteria not specified")

        logger.info(
            f"Judge result: clarification={is_clarification}, valid={is_valid}, score={score}"
        )

        # Record judgment in history
        judgment_record = {
            "timestamp": datetime.now().isoformat(),
            "score": score,
            "is_valid": is_valid,
            "is_clarification": is_clarification,
            "feedback": feedback,
            "criteria": criteria,
            "reasoning": judgment.get("reasoning", ""),
        }

        # Case 1: Clarification Request -> Pass through
        if is_clarification:
            logger.info("Agent requesting clarification - passing through")
            return {
                "next_action": "general_response",
                "execution_path": [
                    {
                        "node": "judge",
                        "timestamp": datetime.now().isoformat(),
                        "action": "clarification_passthrough",
                    }
                ],
            }

        # Case 2: Valid Answer -> Complete
        if is_valid and score >= strictness:  # Threshold for acceptance
            logger.info(
                f"Agent output accepted by judge (score {score} >= {strictness})"
            )
            # We should still record the successful judgment in history if we want complete metadata
            # But the task is complete, so we might not need to update the ledger for retry
            # However, updating the ledger with the history is good practice

            # Create updated task info just to save history
            updated_history = task_info.judgment_history + [judgment_record]
            updated_task_info = TaskInfo(
                task_id=task_info.task_id,
                agent_id=task_info.agent_id,
                description=task_info.description,
                status=TaskStatus.COMPLETED,
                dependencies=task_info.dependencies,
                created_at=task_info.created_at,
                started_at=task_info.started_at,
                completed_at=task_info.completed_at or datetime.utcnow(),
                result=task_info.result,
                error=task_info.error,
                judge_feedback=None,  # Clear feedback on success
                judge_criteria=criteria,
                retry_count=task_info.retry_count,
                max_retries=task_info.max_retries,
                judgment_history=updated_history,
                metadata=task_info.metadata,
            )

            return {
                "task_ledger": {**task_ledger, task_id: updated_task_info},
                "next_action": "general_response",
                "execution_path": [
                    {
                        "node": "judge",
                        "timestamp": datetime.now().isoformat(),
                        "action": "approved",
                        "score": score,
                    }
                ],
            }

        # Case 3: Invalid/Low Quality -> Retry
        retry_count = task_info.retry_count
        max_retries = task_info.max_retries

        if retry_count < max_retries:
            logger.info(
                f"Agent output rejected. Retrying ({retry_count + 1}/{max_retries}). Feedback: {feedback}"
            )

            updated_history = task_info.judgment_history + [judgment_record]

            # Update task info with feedback and increment retry count
            updated_task_info = TaskInfo(
                task_id=task_info.task_id,
                agent_id=task_info.agent_id,
                description=task_info.description,
                status=TaskStatus.IN_PROGRESS,  # Reset to in progress
                dependencies=task_info.dependencies,
                created_at=task_info.created_at,
                started_at=task_info.started_at,
                completed_at=None,  # Reset completion
                result=None,  # Clear result
                error=None,
                judge_feedback=feedback,
                judge_criteria=criteria,
                retry_count=retry_count + 1,
                max_retries=task_info.max_retries,
                judgment_history=updated_history,
                metadata=task_info.metadata,
            )

            # Update decision rationale
            decision_rationale = state.get("decision_rationale", [])
            decision_rationale.append(
                {
                    "node": "judge",
                    "timestamp": datetime.now().isoformat(),
                    "decisions": {"action": "retry", "retry_count": retry_count + 1},
                    "reasoning": {"feedback": feedback, "score": score},
                }
            )

            return {
                "task_ledger": {**task_ledger, task_id: updated_task_info},
                "next_action": "agent_executor",  # Route back to executor
                "decision_rationale": decision_rationale,
                "execution_path": [
                    {
                        "node": "judge",
                        "timestamp": datetime.now().isoformat(),
                        "action": "rejected_retry",
                        "score": score,
                        "retry_count": retry_count + 1,
                    }
                ],
            }
        else:
            logger.warning(
                f"Max retries reached ({max_retries}). Accepting best effort."
            )

            # Record final failure in history
            updated_history = task_info.judgment_history + [judgment_record]

            updated_task_info = TaskInfo(
                task_id=task_info.task_id,
                agent_id=task_info.agent_id,
                description=task_info.description,
                status=TaskStatus.COMPLETED,  # Accept it
                dependencies=task_info.dependencies,
                created_at=task_info.created_at,
                started_at=task_info.started_at,
                completed_at=datetime.utcnow(),
                result=task_info.result,
                error=None,
                judge_feedback=feedback,
                judge_criteria=criteria,
                retry_count=task_info.retry_count,
                max_retries=task_info.max_retries,
                judgment_history=updated_history,
                metadata=task_info.metadata,
            )

            return {
                "task_ledger": {**task_ledger, task_id: updated_task_info},
                "next_action": "general_response",
                "execution_path": [
                    {
                        "node": "judge",
                        "timestamp": datetime.now().isoformat(),
                        "action": "accepted_max_retries",
                        "score": score,
                        "retry_count": task_info.retry_count,
                    }
                ],
            }

    except Exception as e:
        logger.error(f"Error in judge node: {e}")
        # Fail open
        return {"next_action": "general_response"}
