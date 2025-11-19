from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

@dataclass
class Book:
    best_bid: float
    best_ask: float

@dataclass
class Filters:
    step_size: float
    tick_size: float
    min_notional: float

class ExchangeAdapter:
    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_book(self, symbol: str) -> Book:
        raise NotImplementedError

    def get_filters(self, symbol: str) -> Filters:
        raise NotImplementedError

    def place_post_only(self, symbol: str, side: str, quantity: float, price: float, ttl_sec: int) -> str:
        raise NotImplementedError

    def cancel(self, symbol: str, order_id: str) -> None:
        raise NotImplementedError

    def check_order(self, symbol: str, order_id: str) -> Tuple[bool, float]:
        raise NotImplementedError

    def market_order(self, symbol: str, side: str, quantity: float) -> str:
        raise NotImplementedError

    def round_qty(self, qty: float, step: float) -> float:
        from math import floor
        if step <= 0: return qty
        return floor(qty / step) * step

    def round_price(self, price: float, tick: float) -> float:
        from math import floor
        if tick <= 0: return price
        return floor(price / tick) * tick
    
