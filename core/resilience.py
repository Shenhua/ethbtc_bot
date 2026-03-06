"""
Resilience Module - API Call Protection

Provides exponential backoff, retry logic, and circuit breaker patterns
for Binance API calls.

Usage:
    from core.resilience import with_retry, CircuitBreaker

    # Decorator usage
    @with_retry(max_attempts=3)
    def fetch_data():
        return api.get_data()

    # Circuit breaker usage
    breaker = CircuitBreaker(max_failures=5, reset_timeout=60)
    result = breaker.call(api.get_data)
"""
from __future__ import annotations
import time
import logging
from typing import Callable, TypeVar, Any, Optional
from functools import wraps
from dataclasses import dataclass, field

try:
    from tenacity import (
        retry, stop_after_attempt, wait_exponential, 
        retry_if_exception_type, before_sleep_log
    )
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

log = logging.getLogger("resilience")

T = TypeVar('T')


@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures.
    
    States:
        CLOSED: Normal operation, requests pass through
        OPEN: Too many failures, requests are blocked
        HALF_OPEN: Testing if service recovered
    
    Example:
        >>> breaker = CircuitBreaker(max_failures=5, reset_timeout=60)
        >>> result = breaker.call(lambda: api.get_data())
    """
    max_failures: int = 5
    reset_timeout: float = 60.0  # seconds
    
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _state: str = field(default="CLOSED", init=False)
    
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        if self._state == "OPEN":
            # Check if reset timeout has passed
            if time.time() - self._last_failure_time >= self.reset_timeout:
                log.info("Circuit breaker transitioning to HALF_OPEN")
                self._state = "HALF_OPEN"
                return False
            return True
        return False
    
    def record_success(self) -> None:
        """Record a successful call."""
        self._failure_count = 0
        if self._state == "HALF_OPEN":
            log.info("Circuit breaker CLOSED after successful call")
        self._state = "CLOSED"
    
    def record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._failure_count >= self.max_failures:
            log.warning(
                "Circuit breaker OPEN after %d consecutive failures",
                self._failure_count
            )
            self._state = "OPEN"
    
    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute function with circuit breaker protection.
        
        Raises:
            CircuitBreakerOpen: If circuit is open
        """
        if self.is_open():
            raise CircuitBreakerOpen(
                f"Circuit breaker is OPEN. Will reset in "
                f"{self.reset_timeout - (time.time() - self._last_failure_time):.1f}s"
            )
        
        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open and blocking requests."""
    pass


def with_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 30.0,
    exponential_base: float = 2.0,
) -> Callable:
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        exponential_base: Base for exponential backoff calculation
    
    Example:
        @with_retry(max_attempts=3, min_wait=1.0, max_wait=30.0)
        def fetch_klines():
            return adapter.get_klines(symbol, interval)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if TENACITY_AVAILABLE:
            # Use tenacity for sophisticated retry logic
            @retry(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(
                    multiplier=min_wait, 
                    min=min_wait, 
                    max=max_wait, 
                    exp_base=exponential_base
                ),
                before_sleep=before_sleep_log(log, logging.WARNING),
                reraise=True
            )
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                return func(*args, **kwargs)
            return wrapper
        else:
            # Fallback: simple retry with exponential backoff
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                last_exception: Optional[Exception] = None
                
                for attempt in range(1, max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_attempts:
                            wait_time = min(
                                max_wait, 
                                min_wait * (exponential_base ** (attempt - 1))
                            )
                            log.warning(
                                "Attempt %d/%d failed: %s. Retrying in %.1fs...",
                                attempt, max_attempts, e, wait_time
                            )
                            time.sleep(wait_time)
                        else:
                            log.error(
                                "All %d attempts failed. Last error: %s",
                                max_attempts, e
                            )
                
                raise last_exception  # type: ignore
            return wrapper
    return decorator


def retry_api_call(
    func: Callable[..., T],
    *args: Any,
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 30.0,
    circuit_breaker: Optional[CircuitBreaker] = None,
    **kwargs: Any
) -> T:
    """
    Execute an API call with retry logic and optional circuit breaker.
    
    This is a functional alternative to the decorator for one-off calls.
    
    Args:
        func: The function to call
        *args: Positional arguments for func
        max_attempts: Maximum retry attempts
        min_wait: Minimum backoff wait (seconds)
        max_wait: Maximum backoff wait (seconds)
        circuit_breaker: Optional CircuitBreaker instance
        **kwargs: Keyword arguments for func
    
    Returns:
        Result from func
    
    Raises:
        CircuitBreakerOpen: If circuit breaker is open
        Exception: The last exception if all retries fail
    
    Example:
        result = retry_api_call(
            adapter.get_klines, symbol, interval, limit=600,
            max_attempts=3, circuit_breaker=api_breaker
        )
    """
    last_exception: Optional[Exception] = None
    
    for attempt in range(1, max_attempts + 1):
        try:
            if circuit_breaker:
                return circuit_breaker.call(func, *args, **kwargs)
            else:
                return func(*args, **kwargs)
        except CircuitBreakerOpen:
            # Don't retry on circuit breaker open
            raise
        except Exception as e:
            last_exception = e
            if attempt < max_attempts:
                wait_time = min(
                    max_wait, 
                    min_wait * (2 ** (attempt - 1))
                )
                log.warning(
                    "[Retry %d/%d] %s failed: %s. Backing off %.1fs...",
                    attempt, max_attempts, func.__name__, e, wait_time
                )
                time.sleep(wait_time)
            else:
                log.error(
                    "[Retry %d/%d] %s: All attempts exhausted. Error: %s",
                    attempt, max_attempts, func.__name__, e
                )
    
    raise last_exception  # type: ignore
