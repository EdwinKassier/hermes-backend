"""In-memory rate limiter using token bucket algorithm."""

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class TokenBucket:
    """
    In-memory rate limiter using token bucket algorithm.

    Note: This is per-worker. With multiple workers, effective rate
    is: tokens_per_second * num_workers
    """

    def __init__(self, tokens_per_second: float, burst_size: int):
        """
        Initialize token bucket.

        Args:
            tokens_per_second: Rate of token refill (e.g., 10 = 10 requests/sec)
            burst_size: Maximum burst capacity (e.g., 20 = can burst 20 requests)
        """
        self.capacity = burst_size
        self.tokens = burst_size
        self.fill_rate = tokens_per_second
        self.last_update = time.time()
        self.lock = threading.Lock()

    def consume(
        self, tokens: int = 1, block: bool = True, timeout: Optional[float] = None
    ) -> bool:
        """
        Consume tokens from bucket.

        Args:
            tokens: Number of tokens to consume
            block: If True, wait for tokens to be available
            timeout: Maximum time to wait (None = infinite)

        Returns:
            True if tokens consumed, False if rate limited
        """
        start_time = time.time()

        while True:
            with self.lock:
                # Refill tokens based on time passed
                now = time.time()
                time_passed = now - self.last_update
                self.tokens = min(
                    self.capacity, self.tokens + time_passed * self.fill_rate
                )
                self.last_update = now

                # Try to consume tokens
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

                # Can't consume - check if we should wait
                if not block:
                    return False

                # Check timeout
                if timeout and (time.time() - start_time) >= timeout:
                    return False

            # Wait a bit before retrying
            time.sleep(0.01)


class RateLimiter:
    """Rate limiter for TTS providers."""

    # Rate limits per provider (conservative estimates)
    LIMITS = {
        "elevenlabs": {"rps": 10, "burst": 20},  # 10 req/sec, burst 20
        "google": {"rps": 100, "burst": 200},  # Higher limits
    }

    def __init__(self, provider: str):
        """
        Initialize rate limiter for provider.

        Args:
            provider: Provider name ('elevenlabs', 'google')
        """
        limits = self.LIMITS.get(provider, {"rps": 10, "burst": 20})
        self.bucket = TokenBucket(
            tokens_per_second=limits["rps"], burst_size=limits["burst"]
        )
        self.provider = provider
        logger.info(
            f"Rate limiter initialized for {provider}: "
            f"{limits['rps']} req/s, burst {limits['burst']}"
        )

    def acquire(self, timeout: float = 10.0) -> bool:
        """
        Acquire permission to make a request.

        Args:
            timeout: Maximum time to wait for rate limit

        Returns:
            True if acquired

        Raises:
            TimeoutError: If timeout exceeded
        """
        success = self.bucket.consume(block=True, timeout=timeout)

        if not success:
            logger.warning(
                f"Rate limit exceeded for {self.provider} (timeout after {timeout}s)"
            )
            raise TimeoutError(
                f"Rate limit exceeded for {self.provider}. "
                f"Service temporarily overloaded. Try again in a moment."
            )

        return True
