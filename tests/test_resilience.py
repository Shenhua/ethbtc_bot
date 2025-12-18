"""
Tests for the resilience module (core/resilience.py)
"""
import pytest
import time
from core.resilience import CircuitBreaker, retry_api_call, CircuitBreakerOpen


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(max_failures=3)
        assert not cb.is_open()
        assert cb._state == "CLOSED"

    def test_opens_after_max_failures(self):
        cb = CircuitBreaker(max_failures=2, reset_timeout=60.0)
        
        def failing():
            raise ValueError("test error")
        
        # First failure
        with pytest.raises(ValueError):
            cb.call(failing)
        assert not cb.is_open()
        
        # Second failure - should trigger open
        with pytest.raises(ValueError):
            cb.call(failing)
        assert cb.is_open()
        assert cb._state == "OPEN"

    def test_blocks_when_open(self):
        cb = CircuitBreaker(max_failures=1, reset_timeout=60.0)
        
        def failing():
            raise ValueError("test error")
        
        # Trip the breaker
        with pytest.raises(ValueError):
            cb.call(failing)
        
        # Should now throw CircuitBreakerOpen
        with pytest.raises(CircuitBreakerOpen):
            cb.call(failing)

    def test_resets_after_timeout(self):
        cb = CircuitBreaker(max_failures=1, reset_timeout=0.1)
        
        def failing():
            raise ValueError("test error")
        
        # Trip the breaker
        with pytest.raises(ValueError):
            cb.call(failing)
        assert cb.is_open()
        
        # Wait for reset timeout
        time.sleep(0.15)
        
        # Should be half-open now
        assert not cb.is_open()
        assert cb._state == "HALF_OPEN"

    def test_closes_after_successful_call(self):
        cb = CircuitBreaker(max_failures=1, reset_timeout=0.1)
        
        call_count = 0
        def sometimes_fails():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("first call fails")
            return "success"
        
        # Trip the breaker
        with pytest.raises(ValueError):
            cb.call(sometimes_fails)
        
        # Wait for reset
        time.sleep(0.15)
        
        # Successful call should close circuit
        result = cb.call(sometimes_fails)
        assert result == "success"
        assert cb._state == "CLOSED"


class TestRetryApiCall:
    def test_succeeds_on_first_try(self):
        def succeeds():
            return "ok"
        
        result = retry_api_call(succeeds, max_attempts=3)
        assert result == "ok"

    def test_retries_on_failure(self):
        attempts = 0
        def fails_twice():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError(f"attempt {attempts}")
            return "success"
        
        result = retry_api_call(fails_twice, max_attempts=3, min_wait=0.01, max_wait=0.1)
        assert result == "success"
        assert attempts == 3

    def test_raises_after_max_attempts(self):
        def always_fails():
            raise ValueError("always fails")
        
        with pytest.raises(ValueError, match="always fails"):
            retry_api_call(always_fails, max_attempts=2, min_wait=0.01, max_wait=0.1)

    def test_respects_circuit_breaker(self):
        cb = CircuitBreaker(max_failures=1, reset_timeout=60.0)
        
        def always_fails():
            raise ValueError("fail")
        
        # First call with retry - will trip breaker
        with pytest.raises(ValueError):
            retry_api_call(always_fails, max_attempts=1, circuit_breaker=cb)
        
        # Second call should immediately raise CircuitBreakerOpen
        with pytest.raises(CircuitBreakerOpen):
            retry_api_call(always_fails, max_attempts=3, circuit_breaker=cb)
