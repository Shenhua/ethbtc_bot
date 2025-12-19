"""
Order Manager Module

Extracted helper functions from live_executor.py to improve modularity.
The main order execution flow remains in live_executor.py to preserve
existing behavior and avoid breaking changes.

IMPORTANT: This module provides helper functions and data structures.
The main order execution logic is intentionally kept in live_executor.py
to ensure backward compatibility and preserve all features.
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Callable, Any

log = logging.getLogger("order_manager")


@dataclass
class OrderResult:
    """Result of an order execution attempt."""
    success: bool
    order_id: Optional[str] = None
    filled_qty: float = 0.0
    fill_price: float = 0.0
    slippage_bps: float = 0.0
    error: Optional[str] = None


@dataclass
class ExecutionFilters:
    """Symbol trading filters from exchange."""
    min_notional: float = 10.0
    tick_size: float = 0.00000001
    step_size: float = 0.00001
    min_qty: float = 0.0001
    price_precision: int = 8
    qty_precision: int = 4


def round_to_step(value: float, step: float, precision: int = 8) -> float:
    """
    Round a value to the nearest step size.
    
    Args:
        value: Value to round
        step: Step size (e.g., 0.001 for 3 decimal places)
        precision: Maximum decimal precision
    
    Returns:
        Rounded value
    """
    if step <= 0:
        return round(value, precision)
    return round(round(value / step) * step, precision)


def calculate_slippage_bps(expected_price: float, fill_price: float) -> float:
    """
    Calculate slippage in basis points.
    
    Args:
        expected_price: Price expected at order submission
        fill_price: Actual fill price
    
    Returns:
        Slippage in basis points (positive = worse than expected)
    """
    if expected_price <= 0:
        return 0.0
    return (abs(fill_price - expected_price) / expected_price) * 10000


def wait_for_fill(
    check_fn: Callable[[str, str], Tuple[bool, float]],
    symbol: str,
    order_id: str,
    max_wait_seconds: float = 10.0,
    poll_interval: float = 0.5
) -> Tuple[bool, float]:
    """
    Wait for an order to fill with polling.
    
    Args:
        check_fn: Function to check order status (symbol, order_id) -> (is_filled, filled_qty)
        symbol: Trading symbol
        order_id: Order ID to check
        max_wait_seconds: Maximum time to wait
        poll_interval: Time between checks
    
    Returns:
        Tuple of (is_filled, filled_quantity)
    """
    start = time.time()
    last_filled = 0.0
    
    while time.time() - start < max_wait_seconds:
        try:
            is_filled, filled_qty = check_fn(symbol, order_id)
            last_filled = filled_qty
            
            if is_filled:
                return True, filled_qty
            
            # If partial fill and not making progress, stop waiting
            time.sleep(poll_interval)
        except Exception as e:
            log.warning("Fill check error: %s", e)
            time.sleep(poll_interval)
    
    return False, last_filled


def validate_order_params(
    side: str,
    quantity: float,
    price: float,
    filters: ExecutionFilters
) -> Tuple[bool, str]:
    """
    Validate order parameters against exchange filters.
    
    Args:
        side: "BUY" or "SELL"
        quantity: Order quantity
        price: Order price
        filters: Exchange trading filters
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if side not in ("BUY", "SELL"):
        return False, f"Invalid side: {side}"
    
    if quantity <= 0:
        return False, f"Invalid quantity: {quantity}"
    
    if price <= 0:
        return False, f"Invalid price: {price}"
    
    notional = quantity * price
    if notional < filters.min_notional:
        return False, f"Notional {notional} below minimum {filters.min_notional}"
    
    if quantity < filters.min_qty:
        return False, f"Quantity {quantity} below minimum {filters.min_qty}"
    
    return True, ""


class OrderTracker:
    """
    Tracks order execution for a session.
    
    Useful for debugging and metrics collection.
    """
    
    def __init__(self) -> None:
        self.orders_submitted: int = 0
        self.orders_filled: int = 0
        self.orders_rejected: int = 0
        self.total_volume: float = 0.0
        self.total_slippage_bps: float = 0.0
    
    def record_submission(self) -> None:
        """Record an order submission."""
        self.orders_submitted += 1
    
    def record_fill(self, quantity: float, slippage_bps: float = 0.0) -> None:
        """Record a successful fill."""
        self.orders_filled += 1
        self.total_volume += quantity
        self.total_slippage_bps += slippage_bps
    
    def record_rejection(self) -> None:
        """Record an order rejection."""
        self.orders_rejected += 1
    
    @property
    def fill_rate(self) -> float:
        """Get fill rate as percentage."""
        if self.orders_submitted == 0:
            return 0.0
        return (self.orders_filled / self.orders_submitted) * 100
    
    @property
    def avg_slippage_bps(self) -> float:
        """Get average slippage in basis points."""
        if self.orders_filled == 0:
            return 0.0
        return self.total_slippage_bps / self.orders_filled
    
    def summary(self) -> dict[str, float | int]:
        """Get summary statistics."""
        return {
            "orders_submitted": self.orders_submitted,
            "orders_filled": self.orders_filled,
            "orders_rejected": self.orders_rejected,
            "fill_rate_pct": self.fill_rate,
            "total_volume": self.total_volume,
            "avg_slippage_bps": self.avg_slippage_bps,
        }
