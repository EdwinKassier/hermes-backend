"""
Result Validation for Worker Outputs.

This module provides utilities for validating worker results before synthesis
to ensure quality and detect issues early.
"""

import logging
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class ResultValidator:
    """
    Validates worker results before synthesis.

    Checks for:
    - Empty or too short results
    - Error indicators in content
    - Placeholder/incomplete responses
    - Basic quality metrics
    """

    # Patterns that indicate incomplete or error results
    ERROR_PATTERNS = [
        r"(?i)^error:",
        r"(?i)^exception:",
        r"(?i)^failed to",
        r"(?i)^unable to",
        r"(?i)^i couldn't",
        r"(?i)^i can't",
        r"(?i)^sorry,? i",
    ]

    PLACEHOLDER_PATTERNS = [
        r"(?i)^todo:",
        r"(?i)^\[placeholder\]",
        r"(?i)^not implemented",
        r"(?i)^coming soon",
        r"(?i)^tbd",
    ]

    def __init__(
        self,
        min_length: int = 20,
        max_error_ratio: float = 0.5,
    ):
        """
        Initialize the validator.

        Args:
            min_length: Minimum result length to be considered valid
            max_error_ratio: Maximum ratio of error results before flagging
        """
        self.min_length = min_length
        self.max_error_ratio = max_error_ratio

    def validate(self, result: str, min_length: int = None) -> Tuple[bool, str]:
        """
        Validate a single result.

        Args:
            result: The result text to validate
            min_length: Optional override for minimum length

        Returns:
            Tuple of (is_valid, reason)
        """
        length_threshold = min_length or self.min_length

        # Check for None or empty
        if result is None:
            return False, "Result is None"

        if not isinstance(result, str):
            return False, f"Result is not a string: {type(result).__name__}"

        result_stripped = result.strip()

        # Check length
        if len(result_stripped) < length_threshold:
            return (
                False,
                f"Result too short ({len(result_stripped)} < {length_threshold})",
            )

        # Check for error patterns
        for pattern in self.ERROR_PATTERNS:
            if re.search(pattern, result_stripped):
                return False, f"Result contains error indicator: {pattern}"

        # Check for placeholder patterns
        for pattern in self.PLACEHOLDER_PATTERNS:
            if re.search(pattern, result_stripped):
                return False, f"Result contains placeholder: {pattern}"

        return True, "Valid"

    def validate_batch(
        self,
        results: Dict[str, Dict[str, Any]],
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, str]]]:
        """
        Validate a batch of worker results.

        Args:
            results: Dictionary of worker_id -> result_data

        Returns:
            Tuple of (valid_results, validation_issues)
        """
        valid_results = {}
        issues = []

        for worker_id, data in results.items():
            result_text = data.get("result", "")
            status = data.get("status", "unknown")

            # ENHANCEMENT (Issue 2): Include degraded results in valid_results
            # They have useful partial information for synthesis
            if status == "degraded" or data.get("is_partial"):
                valid_results[worker_id] = data
                issues.append(
                    {
                        "worker_id": worker_id,
                        "reason": "Worker returned degraded/partial result",
                        "severity": "low",  # Low severity - still usable
                    }
                )
                continue

            # Failed/timeout workers are excluded
            if status in ["failed", "timeout"]:
                issues.append(
                    {
                        "worker_id": worker_id,
                        "reason": f"Worker had status: {status}",
                        "severity": "high",
                    }
                )
                continue

            is_valid, reason = self.validate(result_text)

            if is_valid:
                valid_results[worker_id] = data
            else:
                issues.append(
                    {
                        "worker_id": worker_id,
                        "reason": reason,
                        "severity": "medium",
                    }
                )
                logger.warning(f"Worker {worker_id} result invalid: {reason}")

        # Log summary
        total = len(results)
        valid_count = len(valid_results)
        invalid_count = len([i for i in issues if i["severity"] == "high"])

        logger.info(
            f"Result validation: {valid_count}/{total} usable, {invalid_count} failures"
        )

        # Check error ratio
        if total > 0 and invalid_count / total > self.max_error_ratio:
            logger.warning(
                f"High error ratio: {invalid_count}/{total} = "
                f"{invalid_count/total:.1%} > {self.max_error_ratio:.1%}"
            )

        return valid_results, issues

    def validate_batch_with_partials(
        self,
        results: Dict[str, Dict[str, Any]],
    ) -> Tuple[
        Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], List[Dict[str, str]]
    ]:
        """
        Validate results separating full success, partial success, and failures.

        This provides more granular control for synthesis, allowing different
        treatment of partial vs complete results.

        Args:
            results: Dictionary of worker_id -> result_data

        Returns:
            Tuple of (valid_results, partial_results, complete_failures)
        """
        valid_results = {}
        partial_results = {}
        issues = []

        for worker_id, data in results.items():
            result_text = data.get("result", "")
            status = data.get("status", "unknown")

            # Check for degraded/partial status
            if status == "degraded" or data.get("is_partial"):
                partial_results[worker_id] = data
                continue

            # Complete failures
            if status in ["failed", "timeout"]:
                issues.append(
                    {
                        "worker_id": worker_id,
                        "reason": f"Worker had status: {status}",
                        "severity": "high",
                    }
                )
                continue

            # Validate the result content
            is_valid, reason = self.validate(result_text)

            if is_valid:
                valid_results[worker_id] = data
            else:
                issues.append(
                    {
                        "worker_id": worker_id,
                        "reason": reason,
                        "severity": "medium",
                    }
                )

        logger.info(
            f"Result validation (with partials): "
            f"{len(valid_results)} valid, {len(partial_results)} partial, "
            f"{len(issues)} issues"
        )

        return valid_results, partial_results, issues

    def get_quality_metrics(
        self,
        results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Calculate quality metrics for a batch of results.

        Args:
            results: Dictionary of worker_id -> result_data

        Returns:
            Dictionary of quality metrics
        """
        if not results:
            return {
                "total_workers": 0,
                "valid_count": 0,
                "invalid_count": 0,
                "success_rate": 0.0,
                "avg_result_length": 0,
                "issues": [],
            }

        valid_results, issues = self.validate_batch(results)

        # Calculate metrics
        total = len(results)
        valid_count = len(valid_results)

        result_lengths = [
            len(data.get("result", "")) for data in valid_results.values()
        ]
        avg_length = sum(result_lengths) / len(result_lengths) if result_lengths else 0

        return {
            "total_workers": total,
            "valid_count": valid_count,
            "invalid_count": len(issues),
            "success_rate": valid_count / total if total > 0 else 0.0,
            "avg_result_length": avg_length,
            "issues": issues,
        }


def get_result_validator() -> ResultValidator:
    """
    Get a ResultValidator instance with default settings.

    Returns:
        ResultValidator instance
    """
    return ResultValidator()
