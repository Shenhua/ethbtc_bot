from __future__ import annotations
import logging
from typing import List, Dict, Any, Tuple
from binance.um_futures import UMFutures
from binance.error import ClientError
from .exchange_adapter import ExchangeAdapter, Book, Filters

log = logging.getLogger("futures_adapter")

class BinanceFuturesAdapter(ExchangeAdapter):
    def __init__(self, client: UMFutures):
        self.client = client

    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[Dict[str, Any]]:
        # Futures API uses same format as Spot for Klines
        ks = self.client.klines(symbol, interval, limit=limit)
        out = []
        for k in ks:
            out.append({
                "open_time": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": int(k[6]),
            })
        return out

    def get_book(self, symbol: str) -> Book:
        t = self.client.book_ticker(symbol)
        return Book(best_bid=float(t["bidPrice"]), best_ask=float(t["askPrice"]))

    def get_filters(self, symbol: str) -> Filters:
        info = self.client.exchange_info()
        s_info = next((s for s in info["symbols"] if s["symbol"] == symbol), None)
        if not s_info:
            raise ValueError(f"Symbol {symbol} not found in Futures exchange info")

        # Futures filters are slightly different than Spot
        price_filter = next((f for f in s_info["filters"] if f["filterType"] == "PRICE_FILTER"), {})
        lot_filter = next((f for f in s_info["filters"] if f["filterType"] == "LOT_SIZE"), {})
        
        # Futures usually ~5 USDT min notional, but sometimes defined in MIN_NOTIONAL filter
        notional_filter = next((f for f in s_info["filters"] if f["filterType"] == "MIN_NOTIONAL"), {})
        min_notional = float(notional_filter.get("notional", "5.0"))

        return Filters(
            step_size=float(lot_filter.get("stepSize", "0")),
            tick_size=float(price_filter.get("tickSize", "0")),
            min_notional=min_notional
        )

    def get_usd_price(self, symbol: str) -> float:
        if "USDT" in symbol or "USDC" in symbol:
            t = self.client.ticker_price(symbol)
            return float(t["price"])
        return 0.0

    def get_funding_rate(self, symbol: str) -> float:
        f = self.client.mark_price(symbol)
        return float(f["lastFundingRate"]) * 100.0

    # --- Execution & State ---

    def set_leverage(self, symbol: str, leverage: int):
        log.debug(f"[FUTURES] Attempting to set leverage for {symbol} to {leverage}")
        try:
            self.client.change_leverage(symbol, leverage)
            log.debug(f"[FUTURES] Leverage set for {symbol} to {leverage}")
        except ClientError as e:
            log.warning(f"Could not set leverage: {e}")

    def get_position(self, symbol: str) -> float:
        """Get current position size (can be negative for short)."""
        log.debug(f"[FUTURES] Fetching position for {symbol}")
        try:
            positions = self.client.get_position_risk(symbol=symbol)
            for pos in positions:
                if pos["symbol"] == symbol:
                    position_amt = float(pos["positionAmt"])
                    log.debug(f"[FUTURES] Position: {position_amt} {symbol} @ {pos['entryPrice']}")
                    return position_amt
            log.debug(f"[FUTURES] No position found for {symbol}")
            return 0.0
        except Exception as e:
            log.error(f"Error fetching position for {symbol}: {e}")
            return 0.0

    def get_account_balance(self, asset: str) -> float:
        """Get Margin Balance (Wallet + PnL) for the asset"""
        log.debug(f"[FUTURES] Fetching account balance for {asset}")
        try:
            acct = self.client.account()
            for a in acct["assets"]:
                if a["asset"] == asset:
                    balance = float(a["marginBalance"])
                    log.debug(f"[FUTURES] Account balance for {asset}: {balance}")
                    return balance
            log.debug(f"[FUTURES] No balance found for asset {asset}")
            return 0.0
        except Exception as e:
            log.error(f"Error fetching account balance for {asset}: {e}")
            return 0.0

    def cancel_open_orders(self, symbol: str) -> List[str]:
        """Cancel all open orders for the symbol. Returns list of cancelled IDs."""
        log.debug(f"[FUTURES] Attempting to cancel all open orders for {symbol}")
        try:
            open_orders = self.client.get_open_orders(symbol)
            cancelled_ids = []
            for o in open_orders:
                oid = o.get("orderId")
                try:
                    self.client.cancel_order(symbol=symbol, orderId=oid)
                    cancelled_ids.append(str(oid))
                    log.debug(f"[FUTURES] Cancelled order {oid} for {symbol}")
                except Exception as e:
                    log.warning(f"Could not cancel order {oid} for {symbol}: {e}")
            log.debug(f"[FUTURES] Cancelled {len(cancelled_ids)} orders for {symbol}")
            return cancelled_ids
        except Exception as e:
            log.error(f"Error cancelling open orders for {symbol}: {e}")
            return []

    # --- Order Placement (Required for Maker Chase) ---

    def place_limit_maker(self, symbol: str, side: str, quantity: float, price: float) -> str:
        """
        Futures Post-Only Order.
        Uses timeInForce='GTX' which is Binance Futures equivalent of LIMIT_MAKER.
        """
        log.debug(f"[FUTURES] Placing POST_ONLY {side} order: {quantity:.8f} {symbol} @ {price:.8f}")
        resp = self.client.new_order(
            symbol=symbol,
            side=side,
            type="LIMIT",
            timeInForce="GTX",  # GTX = Post Only
            quantity=f"{quantity:.8f}",
            price=f"{price:.8f}"
        )
        log.debug(f"[FUTURES] Order response: {resp}")
        return str(resp["orderId"])

    def market_order(self, symbol: str, side: str, quantity: float) -> str:
        """Execute market order on Futures."""
        log.debug(f"[FUTURES] Placing MARKET {side} order: {quantity:.8f} {symbol}")
        resp = self.client.new_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=f"{quantity:.8f}"
        )
        log.debug(f"[FUTURES] Market order response: {resp}")
        return str(resp["orderId"])

    def cancel(self, symbol: str, order_id: str) -> None:
        log.debug(f"[FUTURES] Attempting to cancel order {order_id} for {symbol}")
        try:
            self.client.cancel_order(symbol=symbol, orderId=order_id)
        except Exception as e:
            log.warning(f"Cancel failed for order {order_id}: {e}")

    def check_order(self, symbol: str, order_id: str) -> Tuple[bool, float]:
        """
        Returns (is_filled, executed_qty).
        """
        try:
            od = self.client.get_order(symbol=symbol, orderId=order_id)
            status = od.get("status", "")
            filled = float(od.get("executedQty", "0"))
            return (status == "FILLED"), filled
        except Exception:
            # If order not found or error, assume not filled (caller handles retry/cancel)
            return False, 0.0