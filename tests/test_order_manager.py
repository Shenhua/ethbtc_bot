"""
Tests for OrderManager helper functions.
"""
import pytest
import time
from core.order_manager import (
    ExecutionFilters, OrderTracker,
    round_to_step, calculate_slippage_bps, wait_for_fill, validate_order_params
)


class TestRoundToStep:
    def test_basic_rounding(self):
        assert round_to_step(0.12345, 0.001) == 0.123
        assert round_to_step(0.12355, 0.001) == 0.124
    
    def test_step_zero_uses_precision(self):
        assert round_to_step(0.123456789, 0, precision=4) == 0.1235
    
    def test_large_step(self):
        assert round_to_step(15.0, 10.0) == 20.0
        assert round_to_step(14.0, 10.0) == 10.0


class TestCalculateSlippageBps:
    def test_no_slippage(self):
        assert calculate_slippage_bps(100.0, 100.0) == 0.0
    
    def test_positive_slippage(self):
        # Price moved against us by 0.1%
        assert calculate_slippage_bps(100.0, 100.1) == pytest.approx(10.0)
    
    def test_negative_slippage(self):
        # Price moved in our favor (still reported as positive)
        assert calculate_slippage_bps(100.0, 99.9) == pytest.approx(10.0)
    
    def test_zero_expected_price(self):
        assert calculate_slippage_bps(0.0, 100.0) == 0.0


class TestValidateOrderParams:
    @pytest.fixture
    def filters(self):
        return ExecutionFilters(
            min_notional=10.0,
            min_qty=0.001
        )
    
    def test_valid_order(self, filters):
        valid, msg = validate_order_params("BUY", 0.1, 100.0, filters)
        assert valid == True
        assert msg == ""
    
    def test_invalid_side(self, filters):
        valid, msg = validate_order_params("HOLD", 0.1, 100.0, filters)
        assert valid == False
        assert "side" in msg.lower()
    
    def test_zero_quantity(self, filters):
        valid, msg = validate_order_params("BUY", 0, 100.0, filters)
        assert valid == False
        assert "quantity" in msg.lower()
    
    def test_below_min_notional(self, filters):
        valid, msg = validate_order_params("BUY", 0.001, 100.0, filters)
        assert valid == False
        assert "notional" in msg.lower()


class TestOrderTracker:
    def test_empty_tracker(self):
        tracker = OrderTracker()
        assert tracker.fill_rate == 0.0
        assert tracker.avg_slippage_bps == 0.0
    
    def test_record_fills(self):
        tracker = OrderTracker()
        tracker.record_submission()
        tracker.record_submission()
        tracker.record_fill(1.0, slippage_bps=5.0)
        
        assert tracker.orders_submitted == 2
        assert tracker.orders_filled == 1
        assert tracker.fill_rate == 50.0
        assert tracker.avg_slippage_bps == 5.0
    
    def test_summary(self):
        tracker = OrderTracker()
        tracker.record_submission()
        tracker.record_fill(1.0)
        
        summary = tracker.summary()
        assert "orders_submitted" in summary
        assert "fill_rate_pct" in summary


class TestWaitForFill:
    def test_immediate_fill(self):
        def check_fn(symbol, order_id):
            return True, 1.0

        filled, qty = wait_for_fill(check_fn, "TEST", "123", max_wait_seconds=1)
        assert filled == True
        assert qty == 1.0

    def test_timeout_partial_fill(self):
        def check_fn(symbol, order_id):
            return False, 0.5  # Partial fill

        start = time.time()
        filled, qty = wait_for_fill(check_fn, "TEST", "123", max_wait_seconds=0.3, poll_interval=0.1)
        elapsed = time.time() - start

        assert filled == False
        assert qty == 0.5
        assert elapsed >= 0.3

    def test_timeout_calls_cancel_fn(self):
        """When the wait times out and cancel_fn is provided, it is called once."""
        cancel_calls = []

        def check_fn(symbol, order_id):
            return False, 0.0

        def cancel_fn(symbol, order_id):
            cancel_calls.append((symbol, order_id))

        filled, qty = wait_for_fill(
            check_fn, "ETHBTC", "ORD-1",
            max_wait_seconds=0.15, poll_interval=0.1,
            cancel_fn=cancel_fn,
        )

        assert filled == False
        assert cancel_calls == [("ETHBTC", "ORD-1")]

    def test_timeout_without_cancel_fn_does_not_raise(self):
        """Timeout with no cancel_fn still returns cleanly (no crash)."""
        def check_fn(symbol, order_id):
            return False, 0.0

        filled, qty = wait_for_fill(
            check_fn, "ETHBTC", "ORD-2",
            max_wait_seconds=0.15, poll_interval=0.1,
        )
        assert filled == False

    def test_cancel_failure_logs_error_and_returns(self):
        """If cancel_fn raises, the error is logged but wait_for_fill still returns."""

        def check_fn(symbol, order_id):
            return False, 0.0

        def cancel_fn(symbol, order_id):
            raise RuntimeError("network error")

        # Should not propagate the exception
        filled, qty = wait_for_fill(
            check_fn, "ETHBTC", "ORD-3",
            max_wait_seconds=0.15, poll_interval=0.1,
            cancel_fn=cancel_fn,
        )
        assert filled == False
