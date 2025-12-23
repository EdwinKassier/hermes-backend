"""Resilience utilities for the Legion system."""

import logging
import time
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Optional, Type

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"  # Failing, fast fail
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


class CircuitBreakerError(Exception):
    """Raised when circuit is open."""

    pass


class CircuitBreaker:
    """
    Circuit breaker implementation to prevent cascading failures.

    Wraps function calls to external services. If failures exceed a threshold,
    the circuit opens and subsequent calls fail immediately until a reset
    timeout passes.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: int = 60,
        allowed_exceptions: Optional[tuple] = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            reset_timeout: Seconds to wait before trying again (half-open)
            allowed_exceptions: Exceptions that count as failures (default: all Exception)
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.allowed_exceptions = allowed_exceptions or (Exception,)

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0
        self._half_open_success_count = 0

    @property
    def state(self) -> CircuitState:
        """Get current state, checking for timeout if OPEN."""
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time > self.reset_timeout:
                logger.info(
                    "Circuit breaker reset timeout passed, switching to HALF_OPEN"
                )
                self._state = CircuitState.HALF_OPEN
                self._half_open_success_count = 0
        return self._state

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute the function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: If function fails
        """
        state = self.state

        if state == CircuitState.OPEN:
            raise CircuitBreakerError(
                f"Circuit is OPEN. Failures: {self._failure_count}. "
                f"Next retry in {int(self.reset_timeout - (time.time() - self._last_failure_time))}s"
            )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.allowed_exceptions as e:
            self._on_failure()
            raise e

    def _on_success(self):
        """Handle successful execution."""
        if self._state == CircuitState.HALF_OPEN:
            # In half-open, we need a few successes to close fully?
            # For simplicity, one success closes it.
            logger.info("Circuit breaker recovering: HALF_OPEN -> CLOSED")
            self._state = CircuitState.CLOSED
            self._failure_count = 0
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success if we had some failures but didn't trip
            if self._failure_count > 0:
                self._failure_count = 0

    def _on_failure(self):
        """Handle failed execution."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            logger.warning("Circuit breaker probe failed: HALF_OPEN -> OPEN")
            self._state = CircuitState.OPEN
        elif self._state == CircuitState.CLOSED:
            if self._failure_count >= self.failure_threshold:
                logger.error(
                    f"Circuit breaker tripped: CLOSED -> OPEN ({self._failure_count} failures)"
                )
                self._state = CircuitState.OPEN


# Global circuit breakers for different services
_llm_circuit_breaker = CircuitBreaker(failure_threshold=5, reset_timeout=60)
_db_circuit_breaker = CircuitBreaker(failure_threshold=10, reset_timeout=30)


def with_circuit_breaker(circuit_breaker: CircuitBreaker):
    """Decorator to apply circuit breaker to a function."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return circuit_breaker.call(func, *args, **kwargs)

        return wrapper

    return decorator


def get_llm_circuit_breaker() -> CircuitBreaker:
    """Get the global LLM circuit breaker."""
    return _llm_circuit_breaker


def get_db_circuit_breaker() -> CircuitBreaker:
    """Get the global database circuit breaker."""
    return _db_circuit_breaker
