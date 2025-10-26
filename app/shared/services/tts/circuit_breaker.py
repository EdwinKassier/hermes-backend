"""Circuit breaker pattern for TTS provider fault tolerance."""

import time
import logging
import threading
from enum import Enum
from typing import Callable, Any

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failures detected, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker for TTS providers.
    
    Prevents cascading failures by detecting repeated errors and
    temporarily blocking requests to failing services.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exception: type = Exception
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before trying again (OPEN -> HALF_OPEN)
            expected_exception: Exception type to count as failure
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args, **kwargs: Function arguments
            
        Returns:
            Result from function
            
        Raises:
            RuntimeError: If circuit is OPEN
            Exception: Original exception from func
        """
        with self.lock:
            if self.state == CircuitState.OPEN:
                # Check if timeout expired
                if time.time() - self.last_failure_time >= self.timeout:
                    self.state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker: OPEN -> HALF_OPEN (testing recovery)")
                else:
                    raise RuntimeError(
                        f"Circuit breaker is OPEN. "
                        f"Service temporarily unavailable. "
                        f"Retry in {int(self.timeout - (time.time() - self.last_failure_time))}s"
                    )
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset if in HALF_OPEN
            with self.lock:
                if self.state == CircuitState.HALF_OPEN:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    logger.info("Circuit breaker: HALF_OPEN -> CLOSED (service recovered)")
            
            return result
            
        except self.expected_exception:
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    if self.state != CircuitState.OPEN:
                        self.state = CircuitState.OPEN
                        logger.error(
                            f"Circuit breaker: -> OPEN "
                            f"(failures: {self.failure_count}/{self.failure_threshold})"
                        )
            
            raise
    
    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        with self.lock:
            return self.state
    
    def reset(self) -> None:
        """Manually reset circuit breaker."""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_failure_time = None
            logger.info("Circuit breaker manually reset")

