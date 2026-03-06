"""
Precision Module for High-Accuracy Financial Calculations

Provides Decimal-based helpers for live trading where numerical precision
is critical. Backtesting continues to use float for performance.

Key Features:
- Exchange-precision quantization (8 decimal places for crypto)
- Safe rounding (always round down for sells, up for min sizes)
- PreciseBalance class for accumulating small amounts without drift
"""

from decimal import Decimal, ROUND_DOWN, InvalidOperation
from typing import Union
import logging

log = logging.getLogger(__name__)

# Standard crypto precision (8 decimal places, e.g., BTC satoshi level)
CRYPTO_PRECISION = Decimal('0.00000001')

# USDT precision (typically 2-4 decimal places depending on exchange)
USDT_PRECISION = Decimal('0.0001')


def to_decimal(value: Union[float, str, Decimal], precision: Decimal = CRYPTO_PRECISION) -> Decimal:
    """
    Convert a value to Decimal with exchange precision.
    
    Args:
        value: Float, string, or Decimal to convert
        precision: Decimal precision level (default: 8 decimal places)
    
    Returns:
        Quantized Decimal value
    
    Examples:
        >>> to_decimal(0.123456789)
        Decimal('0.12345678')
        >>> to_decimal('0.1')
        Decimal('0.10000000')
    """
    try:
        if isinstance(value, Decimal):
            return value.quantize(precision, rounding=ROUND_DOWN)
        return Decimal(str(value)).quantize(precision, rounding=ROUND_DOWN)
    except (InvalidOperation, ValueError) as e:
        log.warning("[precision] Failed to convert %s to Decimal: %s", value, e)
        return Decimal('0')


def from_decimal(value: Decimal) -> float:
    """
    Convert Decimal back to float for external APIs.
    
    Args:
        value: Decimal to convert
    
    Returns:
        Float representation
    """
    return float(value)


def quantize_order_size(size: float, min_size: float = 0.0001, precision: Decimal = CRYPTO_PRECISION) -> Decimal:
    """
    Quantize an order size for exchange submission.
    
    Ensures:
    - Size is rounded DOWN (never overspend)
    - Size meets minimum requirements (or returns 0)
    - Size has correct precision for exchange
    
    Args:
        size: Raw order size
        min_size: Minimum order size for exchange
        precision: Decimal precision
    
    Returns:
        Quantized Decimal size (or 0 if below minimum)
    """
    d_size = to_decimal(abs(size), precision)
    d_min = to_decimal(min_size, precision)
    
    if d_size < d_min:
        return Decimal('0')
    
    return d_size


def quantize_price(price: float, tick_size: float = 0.01) -> Decimal:
    """
    Quantize a price to the exchange's tick size.
    
    Args:
        price: Raw price
        tick_size: Minimum price increment
    
    Returns:
        Quantized Decimal price
    """
    tick = Decimal(str(tick_size))
    d_price = Decimal(str(price))
    return (d_price / tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * tick


class PreciseBalance:
    """
    High-precision balance tracker for live trading.
    
    Prevents floating-point drift when accumulating many small amounts.
    Useful for tracking fees, fractional fills, and long-running balances.
    
    Example:
        >>> balance = PreciseBalance(1.0)
        >>> balance.add(0.00000001)  # Add 1 satoshi
        >>> balance.add(0.00000001)  # Add another
        >>> balance.value
        1.00000002
    """
    
    def __init__(self, initial: float = 0.0, precision: Decimal = CRYPTO_PRECISION):
        """
        Initialize balance tracker.
        
        Args:
            initial: Starting balance
            precision: Decimal precision level
        """
        self._precision = precision
        self._value = to_decimal(initial, precision)
    
    def add(self, amount: float) -> None:
        """Add to balance (can be negative for deductions)."""
        self._value += to_decimal(amount, self._precision)
    
    def subtract(self, amount: float) -> None:
        """Subtract from balance."""
        self._value -= to_decimal(amount, self._precision)
    
    def set(self, value: float) -> None:
        """Set balance to a specific value."""
        self._value = to_decimal(value, self._precision)
    
    @property
    def value(self) -> float:
        """Get balance as float for external use."""
        return from_decimal(self._value)
    
    @property
    def decimal_value(self) -> Decimal:
        """Get balance as Decimal for internal calculations."""
        return self._value
    
    def __repr__(self) -> str:
        return f"PreciseBalance({self._value})"
    
    def __float__(self) -> float:
        return self.value


class PreciseAccumulator:
    """
    Accumulator for tracking running totals (fees, PnL, etc.).
    
    Provides both running total and count of operations.
    """
    
    def __init__(self, precision: Decimal = CRYPTO_PRECISION):
        self._precision = precision
        self._total = Decimal('0')
        self._count = 0
    
    def add(self, amount: float) -> None:
        """Add a value to the accumulator."""
        self._total += to_decimal(amount, self._precision)
        self._count += 1
    
    @property
    def total(self) -> float:
        """Get total as float."""
        return from_decimal(self._total)
    
    @property
    def count(self) -> int:
        """Get number of additions."""
        return self._count
    
    @property
    def average(self) -> float:
        """Get average value."""
        if self._count == 0:
            return 0.0
        return from_decimal(self._total / self._count)
    
    def reset(self) -> None:
        """Reset accumulator to zero."""
        self._total = Decimal('0')
        self._count = 0
