"""Tests for ResultValidator."""

import pytest

from app.hermes.legion.utils.result_validator import (
    ResultValidator,
    get_result_validator,
)


class TestResultValidator:
    """Tests for ResultValidator class."""

    def test_validate_valid_result(self):
        """Valid result passes validation."""
        validator = ResultValidator(min_length=10)
        is_valid, reason = validator.validate(
            "This is a valid result that is long enough to pass validation."
        )
        assert is_valid is True
        assert reason == "Valid"

    def test_validate_too_short(self):
        """Short result fails validation."""
        validator = ResultValidator(min_length=50)
        is_valid, reason = validator.validate("Short")
        assert is_valid is False
        assert "too short" in reason.lower()

    def test_validate_none_result(self):
        """None result fails validation."""
        validator = ResultValidator()
        is_valid, reason = validator.validate(None)
        assert is_valid is False
        assert "None" in reason

    def test_validate_non_string_result(self):
        """Non-string result fails validation."""
        validator = ResultValidator()
        is_valid, reason = validator.validate(12345)
        assert is_valid is False
        assert "not a string" in reason.lower()

    def test_validate_empty_string(self):
        """Empty string fails validation."""
        validator = ResultValidator(min_length=1)
        is_valid, reason = validator.validate("")
        assert is_valid is False
        assert "too short" in reason.lower()

    def test_validate_whitespace_only(self):
        """Whitespace-only string fails validation."""
        validator = ResultValidator(min_length=1)
        is_valid, reason = validator.validate("   \n\t  ")
        assert is_valid is False
        assert "too short" in reason.lower()

    def test_validate_error_patterns(self):
        """Results with error patterns fail validation."""
        validator = ResultValidator(min_length=5)

        error_results = [
            "Error: Something went wrong",
            "Exception: Division by zero",
            "Failed to complete the task",
            "Unable to process request",
            "I couldn't find the information",
            "I can't help with that request",
            "Sorry, I encountered a problem",
        ]

        for result in error_results:
            is_valid, reason = validator.validate(result)
            assert is_valid is False, f"Expected '{result}' to fail validation"
            assert "error indicator" in reason.lower()

    def test_validate_placeholder_patterns(self):
        """Results with placeholder patterns fail validation."""
        validator = ResultValidator(min_length=5)

        placeholder_results = [
            "TODO: implement this",
            "[Placeholder] content here",
            "Not implemented yet",
            "Coming soon...",
            "TBD - to be determined",
        ]

        for result in placeholder_results:
            is_valid, reason = validator.validate(result)
            assert is_valid is False, f"Expected '{result}' to fail validation"
            assert "placeholder" in reason.lower()

    def test_validate_custom_min_length(self):
        """validate respects custom min_length parameter."""
        validator = ResultValidator(min_length=100)

        # Too short for default, but pass custom
        result = "This is a short result"
        is_valid, reason = validator.validate(result, min_length=10)
        assert is_valid is True

    def test_validate_batch_all_valid(self):
        """validate_batch with all valid results."""
        validator = ResultValidator(min_length=10)
        results = {
            "worker_1": {
                "result": "Valid result from worker one.",
                "status": "success",
            },
            "worker_2": {
                "result": "Another valid result from worker two.",
                "status": "success",
            },
        }

        valid_results, issues = validator.validate_batch(results)

        assert len(valid_results) == 2
        assert len(issues) == 0
        assert "worker_1" in valid_results
        assert "worker_2" in valid_results

    def test_validate_batch_some_invalid(self):
        """validate_batch filters out invalid results."""
        validator = ResultValidator(min_length=10)
        results = {
            "worker_1": {
                "result": "Valid result from worker one.",
                "status": "success",
            },
            "worker_2": {
                "result": "Short",
                "status": "success",
            },
            "worker_3": {
                "result": "Error: something failed",
                "status": "success",
            },
        }

        valid_results, issues = validator.validate_batch(results)

        assert len(valid_results) == 1
        assert len(issues) == 2
        assert "worker_1" in valid_results
        assert "worker_2" not in valid_results
        assert "worker_3" not in valid_results

    def test_validate_batch_failed_status_excluded(self):
        """Workers with failed status are excluded."""
        validator = ResultValidator(min_length=10)
        results = {
            "worker_1": {
                "result": "Valid result that should pass.",
                "status": "success",
            },
            "worker_2": {
                "result": "This would be valid but status is failed.",
                "status": "failed",
            },
            "worker_3": {
                "result": "This would be valid but timed out.",
                "status": "timeout",
            },
        }

        valid_results, issues = validator.validate_batch(results)

        assert len(valid_results) == 1
        assert len(issues) == 2
        assert "worker_1" in valid_results
        assert any(i["worker_id"] == "worker_2" for i in issues)
        assert any(i["worker_id"] == "worker_3" for i in issues)

    def test_validate_batch_empty(self):
        """validate_batch handles empty results."""
        validator = ResultValidator()
        valid_results, issues = validator.validate_batch({})
        assert len(valid_results) == 0
        assert len(issues) == 0

    def test_get_quality_metrics(self):
        """get_quality_metrics returns correct statistics."""
        validator = ResultValidator(min_length=10)
        results = {
            "worker_1": {
                "result": "Valid result from worker one which is long enough.",
                "status": "success",
            },
            "worker_2": {
                "result": "Another valid result from worker two.",
                "status": "success",
            },
            "worker_3": {
                "result": "Fail",
                "status": "success",
            },
        }

        metrics = validator.get_quality_metrics(results)

        assert metrics["total_workers"] == 3
        assert metrics["valid_count"] == 2
        assert metrics["invalid_count"] == 1
        assert metrics["success_rate"] == pytest.approx(2 / 3)
        assert metrics["avg_result_length"] > 0
        assert len(metrics["issues"]) == 1

    def test_get_quality_metrics_empty(self):
        """get_quality_metrics handles empty results."""
        validator = ResultValidator()
        metrics = validator.get_quality_metrics({})

        assert metrics["total_workers"] == 0
        assert metrics["valid_count"] == 0
        assert metrics["success_rate"] == 0.0

    def test_convenience_function(self):
        """get_result_validator convenience function works."""
        validator = get_result_validator()
        assert isinstance(validator, ResultValidator)

    def test_max_error_ratio_warning(self):
        """High error ratio triggers warning (logged)."""
        validator = ResultValidator(min_length=100, max_error_ratio=0.3)
        results = {
            f"worker_{i}": {
                "result": "Short",  # All will fail
                "status": "success",
            }
            for i in range(5)
        }

        # Should not raise, but will log warning
        valid_results, issues = validator.validate_batch(results)

        assert len(valid_results) == 0
        assert len(issues) == 5
