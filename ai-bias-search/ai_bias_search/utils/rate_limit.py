"""Simple synchronous rate limiting utilities."""

from __future__ import annotations

import threading
import time
from functools import wraps
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class RateLimiter:
    """Token bucket rate limiter suitable for low-throughput API usage."""

    def __init__(self, *, rate: float, burst: int) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        if burst <= 0:
            raise ValueError("burst must be positive")
        self.rate = rate
        self.burst = burst
        self._tokens = float(burst)
        self._lock = threading.Lock()
        self._updated = time.monotonic()

    def acquire(self) -> None:
        """Block until a token is available."""

        with self._lock:
            now = time.monotonic()
            elapsed = now - self._updated
            self._updated = now
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
            if self._tokens < 1.0:
                to_sleep = (1.0 - self._tokens) / self.rate
                time.sleep(to_sleep)
                self._updated = time.monotonic()
                self._tokens = min(self.burst, self._tokens + to_sleep * self.rate)
            self._tokens -= 1.0


def rate_limited(limiter: RateLimiter) -> Callable[[F], F]:
    """Decorator that applies rate limiting to a function call."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            limiter.acquire()
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
