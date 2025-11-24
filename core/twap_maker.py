from __future__ import annotations
import time
import logging
from core.exchange_adapter import ExchangeAdapter

log = logging.getLogger("maker_chase")

def maker_chase(adapter: ExchangeAdapter, symbol: str, side: str, qty: float, tick: float,
                max_reprices: int = 3, step_sec: int = 10) -> float:
    """
    Aggressive Maker Strategy:
    1. Place LIMIT_MAKER at best Bid/Ask.
    2. Wait `step_sec`.
    3. If not filled, Cancel.
    4. Repeat `max_reprices` times.
    5. Return amount filled. (Caller decides if Taker fallback is needed for remainder).
    """
    filled_total = 0.0
    remaining_qty = qty

    for i in range(max_reprices):
        if remaining_qty <= 0:
            break

        # 1. Get Price
        try:
            book = adapter.get_book(symbol)
            # If buying, join the bid. If selling, join the ask.
            raw_price = book.best_bid if side == "BUY" else book.best_ask
            price = adapter.round_price(raw_price, tick)
        except Exception as e:
            log.warning(f"Chase {i}: Failed to get book: {e}")
            time.sleep(1)
            continue

        # 2. Place Order
        oid = None
        try:
            # This call is now non-blocking
            oid = adapter.place_limit_maker(symbol, side, remaining_qty, price)
        except Exception as e:
            # LIMIT_MAKER fails if price crosses spread (would match immediately).
            # In that case, we might be too aggressive. Sleep and retry.
            log.debug(f"Chase {i}: Post-only failed (crossed spread?): {e}")
            time.sleep(2)
            continue

        # 3. Wait
        # In a single-threaded bot, this blocks the main loop.
        # For 5-10s, this is acceptable.
        time.sleep(step_sec)

        # 4. Check & Cancel
        try:
            is_filled, this_fill = adapter.check_order(symbol, oid)
            filled_total += this_fill
            remaining_qty = max(0.0, remaining_qty - this_fill)

            if not is_filled and remaining_qty > 1e-8:
                adapter.cancel(symbol, oid)
                # Double check after cancel if we got filled in the split second before cancel
                _, post_cancel_fill = adapter.check_order(symbol, oid)
                # If fill increased after cancel, account for it
                diff = max(0.0, post_cancel_fill - this_fill)
                filled_total += diff
                remaining_qty = max(0.0, remaining_qty - diff)
        except Exception as e:
            log.error(f"Chase {i}: Error checking/canceling order {oid}: {e}")
            # If we can't verify, we stop chasing to avoid double spending
            break
            
        if remaining_qty <= 1e-8:
            break

    return filled_total