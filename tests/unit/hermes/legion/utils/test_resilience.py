"""Unit tests for Circuit Breaker resilience pattern."""

import time
from unittest.mock import MagicMock, Mock

import pytest

from app.hermes.legion.utils.resilience import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
    get_llm_circuit_breaker,
)


class TestCircuitBreaker:
    """Test suite for CircuitBreaker class."""

    def test_initial_state_is_closed(self):
        """Circuit breaker should start in CLOSED state."""
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=1)
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0

    def test_successful_call_in_closed_state(self):
        """Successful calls should work normally in CLOSED state."""
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=1)

        def successful_func():
            return "success"

        result = cb.call(successful_func)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0

    def test_failure_increments_count(self):
        """Failures should increment the failure count."""
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=1)

        def failing_func():
            raise ValueError("Test error")

        # First failure
        with pytest.raises(ValueError):
            cb.call(failing_func)

        assert cb._failure_count == 1
        assert cb.state == CircuitState.CLOSED

    def test_circuit_opens_after_threshold(self):
        """Circuit should open after reaching failure threshold."""
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=1)

        def failing_func():
            raise ValueError("Test error")

        # First failure
        with pytest.raises(ValueError):
            cb.call(failing_func)

        # Second failure - should open circuit
        with pytest.raises(ValueError):
            cb.call(failing_func)

        assert cb.state == CircuitState.OPEN
        assert cb._failure_count == 2

    def test_open_circuit_rejects_calls(self):
        """OPEN circuit should reject all calls immediately."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=1)

        def failing_func():
            raise ValueError("Test error")

        # Trigger circuit to open
        with pytest.raises(ValueError):
            cb.call(failing_func)

        assert cb.state == CircuitState.OPEN

        # Subsequent calls should be rejected without calling the function
        call_count = 0

        def tracked_func():
            nonlocal call_count
            call_count += 1
            return "success"

        with pytest.raises(CircuitBreakerError):
            cb.call(tracked_func)

        assert call_count == 0  # Function should not have been called
        assert cb.state == CircuitState.OPEN

    def test_half_open_after_timeout(self):
        """Circuit should transition to HALF_OPEN after reset timeout."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.1)

        def failing_func():
            raise ValueError("Test error")

        # Open the circuit
        with pytest.raises(ValueError):
            cb.call(failing_func)

        assert cb.state == CircuitState.OPEN

        # Wait for reset timeout
        time.sleep(0.15)

        # Next call should attempt in HALF_OPEN state
        def successful_func():
            return "success"

        result = cb.call(successful_func)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED  # Should close on success

    def test_half_open_success_closes_circuit(self):
        """Successful call in HALF_OPEN should close the circuit."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.1)

        def failing_func():
            raise ValueError("Test error")

        # Open the circuit
        with pytest.raises(ValueError):
            cb.call(failing_func)

        # Wait for reset timeout
        time.sleep(0.15)

        def successful_func():
            return "recovery"

        result = cb.call(successful_func)
        assert result == "recovery"
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0

    def test_half_open_failure_reopens_circuit(self):
        """Failed call in HALF_OPEN should re-open the circuit."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.1)

        def failing_func():
            raise ValueError("Test error")

        # Open the circuit
        with pytest.raises(ValueError):
            cb.call(failing_func)

        assert cb.state == CircuitState.OPEN

        # Wait for reset timeout
        time.sleep(0.15)

        # Fail again in HALF_OPEN
        with pytest.raises(ValueError):
            cb.call(failing_func)

        assert cb.state == CircuitState.OPEN

    def test_success_resets_failure_count(self):
        """Successful call should reset failure count."""
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=1)

        def failing_func():
            raise ValueError("Test error")

        def successful_func():
            return "success"

        # Two failures
        with pytest.raises(ValueError):
            cb.call(failing_func)
        with pytest.raises(ValueError):
            cb.call(failing_func)

        assert cb._failure_count == 2

        # One success should reset count
        cb.call(successful_func)
        assert cb._failure_count == 0
        assert cb.state == CircuitState.CLOSED

    def test_call_with_args_and_kwargs(self):
        """Circuit breaker should pass through args and kwargs."""
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=1)

        def func_with_params(a, b, c=None):
            return f"{a}-{b}-{c}"

        result = cb.call(func_with_params, 1, 2, c=3)
        assert result == "1-2-3"

    def test_different_exception_types(self):
        """Circuit breaker should handle different exception types."""
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=1)

        def runtime_error():
            raise RuntimeError("Runtime error")

        def type_error():
            raise TypeError("Type error")

        # Both should increment failure count
        with pytest.raises(RuntimeError):
            cb.call(runtime_error)

        assert cb._failure_count == 1

        with pytest.raises(TypeError):
            cb.call(type_error)

        assert cb.state == CircuitState.OPEN
        assert cb._failure_count == 2


class TestCircuitBreakerGlobalInstances:
    """Test global circuit breaker instances."""

    def test_get_llm_circuit_breaker_returns_same_instance(self):
        """get_llm_circuit_breaker should return the same instance."""
        cb1 = get_llm_circuit_breaker()
        cb2 = get_llm_circuit_breaker()

        assert cb1 is cb2


class TestCircuitBreakerThreadSafety:
    """Test thread safety of circuit breaker."""

    def test_concurrent_calls_dont_corrupt_state(self):
        """Concurrent calls should not corrupt circuit breaker state."""
        import threading

        cb = CircuitBreaker(failure_threshold=10, reset_timeout=1)
        results = []
        errors = []

        def successful_func():
            time.sleep(0.001)  # Simulate brief work
            return "success"

        def call_circuit_breaker():
            try:
                result = cb.call(successful_func)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Create 20 threads
        threads = [threading.Thread(target=call_circuit_breaker) for _ in range(20)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for all to complete
        for t in threads:
            t.join()

        # All should succeed
        assert len(results) == 20
        assert len(errors) == 0
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0


class TestCircuitBreakerEdgeCases:
    """Test edge cases and unusual scenarios."""

    def test_zero_threshold_immediately_opens(self):
        """Zero threshold should open immediately on first failure."""
        cb = CircuitBreaker(failure_threshold=0, reset_timeout=1)

        def failing_func():
            raise ValueError("Test error")

        # Should already be open or open on first failure
        with pytest.raises((ValueError, CircuitBreakerError)):
            cb.call(failing_func)

        # Definitely open now
        with pytest.raises(CircuitBreakerError):
            cb.call(lambda: "test")

    def test_very_long_reset_timeout(self):
        """Circuit with long reset timeout should stay OPEN."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=100)

        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            cb.call(failing_func)

        assert cb.state == CircuitState.OPEN

        # Even after a short wait, should still be OPEN
        time.sleep(0.1)

        with pytest.raises(CircuitBreakerError):
            cb.call(lambda: "test")

        assert cb.state == CircuitState.OPEN

    def test_function_returns_none(self):
        """Circuit breaker should handle functions that return None."""
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=1)

        def returns_none():
            return None

        result = cb.call(returns_none)
        assert result is None
        assert cb.state == CircuitState.CLOSED
