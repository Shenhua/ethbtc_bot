from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

@dataclass
class Book:
    """
    Represents the current state of the order book (best bid/ask).
    """
    best_bid: float
    best_ask: float

@dataclass
class Filters:
    """
    Exchange trading filters for a symbol.
    """
    step_size: float
    tick_size: float
    min_notional: float

class ExchangeAdapter:
    """
    Abstract base class for exchange adapters (Spot, Futures, etc.).
    Defines the standard interface for interacting with an exchange.
    """
    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[Dict[str, Any]]:
        """
        Fetches historical klines (candlesticks).

        Args:
            symbol: Trading pair symbol.
            interval: Bar interval (e.g., '1m', '1h').
            limit: Number of bars to fetch.

        Returns:
            List[Dict[str, Any]]: List of kline data.
        """
        raise NotImplementedError

    def get_book(self, symbol: str) -> Book:
        """
        Fetches current best bid/ask.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Book: Current book state.
        """
        raise NotImplementedError

    def get_filters(self, symbol: str) -> Filters:
        """
        Fetches trading rules/filters.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Filters: Symbol filters.
        """
        raise NotImplementedError

    # Renamed & Simplified: Just places the order, returns ID immediately. No sleeping.
    def place_limit_maker(self, symbol: str, side: str, quantity: float, price: float) -> str:
        """
        Places a Maker-only limit order (Post-Only).

        Args:
            symbol: Trading pair symbol.
            side: 'BUY' or 'SELL'.
            quantity: Order amount.
            price: Limit price.

        Returns:
            str: Order ID.
        """
        raise NotImplementedError

    def cancel(self, symbol: str, order_id: str) -> None:
        """
        Cancels an order.

        Args:
            symbol: Trading pair symbol.
            order_id: Order ID to cancel.
        """
        raise NotImplementedError

    def check_order(self, symbol: str, order_id: str) -> Tuple[bool, float]:
        """
        Checks the status of an order.

        Args:
            symbol: Trading pair symbol.
            order_id: Order ID.

        Returns:
            Tuple[bool, float]: (is_filled, executed_qty).
        """
        raise NotImplementedError

    def market_order(self, symbol: str, side: str, quantity: float) -> str:
        """
        Places a Market order.

        Args:
            symbol: Trading pair symbol.
            side: 'BUY' or 'SELL'.
            quantity: Order amount.

        Returns:
            str: Order ID.
        """
        raise NotImplementedError

    def get_usd_price(self, symbol: str) -> float:
        """
        Gets the current price in USD.

        Args:
            symbol: Trading pair symbol.

        Returns:
            float: Price.
        """
        raise NotImplementedError
    
    def get_funding_rate(self, symbol: str) -> float:
        """
        Gets the current funding rate (Futures only).

        Args:
            symbol: Trading pair symbol.

        Returns:
            float: Funding rate percentage.
        """
        raise NotImplementedError

    def round_qty(self, qty: float, step: float) -> float:
        """
        Rounds a quantity to the nearest step size (floored).

        Args:
            qty: Quantity to round.
            step: Step size.

        Returns:
            float: Rounded quantity.
        """
        from math import floor
        if step <= 0: return qty
        return floor(qty / step) * step

    def round_price(self, price: float, tick: float) -> float:
        """
        Rounds a price to the nearest tick size (floored).

        Args:
            price: Price to round.
            tick: Tick size.

        Returns:
            float: Rounded price.
        """
        from math import floor
        if tick <= 0: return price
        return floor(price / tick) * tick