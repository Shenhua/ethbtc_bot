from __future__ import annotations
import time
import logging
from core.exchange_adapter import ExchangeAdapter

log = logging.getLogger("maker_chase")

def maker_chase(adapter: ExchangeAdapter, symbol: str, side: str, qty: float, tick: float,
                max_reprices: int = 3, step_sec: int = 10, stop_event=None) -> float:
    """
    Executes a 'Maker Chase' strategy to fill an order by placing Post-Only limit orders
    at the best bid/ask and updating them if the price moves.

    This logic attempts to capture the spread (or at least avoid taker fees) by
    aggressively re-pricing limit orders.

    Args:
        adapter: Exchange adapter instance.
        symbol: Trading pair symbol.
        side: "BUY" or "SELL".
        qty: Total quantity to fill.
        tick: Price tick size (for rounding).
        max_reprices: Maximum number of times to cancel and replace the order.
        step_sec: Seconds to wait before checking/re-pricing.
        stop_event: Optional threading.Event to signal immediate abort (e.g., shutdown).

    Returns:
        float: Total quantity filled.
    """
    filled_total = 0.0
    remaining_qty = qty

    for i in range(max_reprices):
        # 0. Check external abort signal (Fix Item 2)
        if stop_event and stop_event.is_set():
            log.info("Maker chase aborted by stop signal.")
            break

        if remaining_qty <= 0:
            break

        # 1. Get Price
        try:
            log.debug(f"Chase {i}: Fetching order book for {symbol}")
            book = adapter.get_book(symbol)
            raw_price = book.best_bid if side == "BUY" else book.best_ask
            price = adapter.round_price(raw_price, tick)
            log.debug(f"Chase {i}: Book price: bid={book.best_bid:.8f} ask={book.best_ask:.8f}, using {price:.8f}")
        except Exception as e:
            log.warning(f"Chase {i}: Failed to get book: {e}")
            time.sleep(1)
            continue

        # 2. Place Order
        oid = None
        try:
            log.debug(f"Chase {i}: Placing POST_ONLY {side} {remaining_qty:.8f} @ {price:.8f}")
            oid = adapter.place_limit_maker(symbol, side, remaining_qty, price)
            log.debug(f"Chase {i}: Order placed successfully (oid={oid})")
        except Exception as e:
            log.debug(f"Chase {i}: Post-only order rejected: {e}", exc_info=True)
            log.info(f"Maker order rejected (attempt {i+1}/{max_reprices}): {type(e).__name__}")
            time.sleep(2)
            continue

        # 3. Wait (Interruptible Sleep - Fix Item 2)
        # We sleep in small chunks to check for stop signals
        chunk = 0.5
        waited = 0.0
        while waited < step_sec:
            if stop_event and stop_event.is_set():
                log.info("Maker chase aborted during wait. Cancelling...")
                adapter.cancel(symbol, oid)
                return filled_total # Return what we have
            time.sleep(chunk)
            waited += chunk

        # 4. Check & Cancel
        try:
            is_filled, this_fill = adapter.check_order(symbol, oid)
            filled_total += this_fill
            remaining_qty = max(0.0, remaining_qty - this_fill)

            if not is_filled and remaining_qty > 1e-8:
                adapter.cancel(symbol, oid)
                # Double check fill after cancel
                _, post_cancel_fill = adapter.check_order(symbol, oid)
                diff = max(0.0, post_cancel_fill - this_fill)
                filled_total += diff
                remaining_qty = max(0.0, remaining_qty - diff)
        except Exception as e:
            log.error(f"Chase {i}: Error checking/canceling order {oid}: {e}")
            break
            
        if remaining_qty <= 1e-8:
            break

    return filled_total