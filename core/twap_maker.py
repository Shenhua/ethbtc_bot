from __future__ import annotations
import time
from .exchange_adapter import ExchangeAdapter

def maker_chase(adapter: ExchangeAdapter, symbol: str, side: str, qty: float, tick: float,
                max_reprices: int = 3, step_sec: int = 7) -> float:
    filled_total = 0.0
    for _ in range(max_reprices):
        book = adapter.get_book(symbol)
        price = book.best_bid if side == "BUY" else book.best_ask
        price = adapter.round_price(price, tick)
        oid = adapter.place_post_only(symbol, side, qty, price, ttl_sec=step_sec)
        is_filled, filled = adapter.check_order(symbol, oid)
        filled_total += filled
        qty = max(qty - filled, 0.0)
        if qty <= 0.0:
            break
    return filled_total

def twap_rebalance(adapter: ExchangeAdapter, symbol: str, side: str, qty: float, tick: float,
                   bar_seconds: int, parts: int = 3, per_part_ttl: int = 10) -> float:
    filled_total = 0.0
    if parts <= 1:
        return maker_chase(adapter, symbol, side, qty, tick, max_reprices=3, step_sec=per_part_ttl)
    slice_qty = qty / parts
    sleep_between = max(bar_seconds // parts, 1)
    for _ in range(parts):
        filled = maker_chase(adapter, symbol, side, slice_qty, tick, max_reprices=2, step_sec=per_part_ttl)
        filled_total += filled
        time.sleep(sleep_between)
    return filled_total
