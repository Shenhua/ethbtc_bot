from __future__ import annotations
import time
import requests
from typing import Any, Dict, List, Tuple
from binance.spot import Spot
from .exchange_adapter import ExchangeAdapter, Book, Filters
import logging

log = logging.getLogger("binance_adapter")

class BinanceSpotAdapter(ExchangeAdapter):
    """
    Adapter for Binance Spot API.

    Inherits from ExchangeAdapter and implements methods to interact with
    Binance Spot markets, including fetching klines, order book, balance,
    and executing orders.
    """
    def __init__(self, client: Spot, public_client: Spot | None = None, timeout: int = 5000):
        """
        Initializes the Binance Spot adapter.

        Args:
            client: Authenticated Binance Spot client instance.
            public_client: Optional unauthenticated client for public endpoints.
            timeout: Timeout for requests in milliseconds.
        """
        self.client = client
        self.public_client = public_client or Spot(timeout=timeout)   

    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[Dict[str, Any]]:
        """
        Fetches kline (candlestick) data.

        Args:
            symbol: Trading pair symbol (e.g., 'ETHBTC').
            interval: Kline interval (e.g., '15m').
            limit: Number of klines to fetch.

        Returns:
            List[Dict[str, Any]]: List of kline dictionaries with standardized keys.
        """
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
    
    def get_usd_price(self, symbol: str) -> float:
        """
        Fetches current USD price for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., 'ETHUSDT').

        Returns:
            float: Current price.
        """
        try:
            data = self.public_client.ticker_price(symbol=symbol)
            return float(data["price"])
        except Exception:
            raise

    def get_mid(self, symbol: str) -> float:
        """
        Calculates the mid-price from the order book.

        Args:
            symbol: Trading pair symbol.

        Returns:
            float: Mid price (average of best bid and ask).
        """
        book = self.get_book(symbol)
        return 0.5 * (book.best_bid + book.best_ask)

    def get_book(self, symbol: str) -> Book:
        """
        Fetches the current best bid and ask prices.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Book: Object containing best_bid and best_ask.
        """
        try:
            t = self.client.ticker_book_ticker(symbol=symbol)
            return Book(best_bid=float(t["bidPrice"]), best_ask=float(t["askPrice"]))
        except AttributeError:
            try:
                t = self.client.book_ticker(symbol=symbol)
                return Book(best_bid=float(t["bidPrice"]), best_ask=float(t["askPrice"]))
            except AttributeError:
                d = self.client.depth(symbol=symbol, limit=5)
                bid = float(d["bids"][0][0]); ask = float(d["asks"][0][0])
                return Book(best_bid=bid, best_ask=ask)

    def _get_exchange_info_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Helper to fetch exchange info for a specific symbol.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Dict[str, Any]: Symbol configuration dictionary.
        """
        info = self.client.exchange_info(symbol=symbol)
        s = info["symbols"][0]
        return s

    def get_filters(self, symbol: str) -> Filters:
        """
        Fetches and parses trading filters (tick size, step size, min notional).

        Args:
            symbol: Trading pair symbol.

        Returns:
            Filters: Object containing parsed filters.
        """
        s = self._get_exchange_info_symbol(symbol)
        flist = s.get("filters", [])

        def pick(kind):
            for f in flist:
                if f.get("filterType") == kind:
                    return f
            return None

        lot = pick("LOT_SIZE") or pick("MARKET_LOT_SIZE")
        price = pick("PRICE_FILTER")
        notional = pick("MIN_NOTIONAL") or pick("NOTIONAL")

        step_size = float(lot.get("stepSize", "0")) if lot else 0.0
        tick_size = float(price.get("tickSize", "0")) if price else 0.0
        min_notional = float(notional.get("minNotional", "0")) if notional else 0.0

        filters = Filters(step_size=step_size, tick_size=tick_size, min_notional=min_notional)
        log.debug(f"[SPOT] Filters for {symbol}: {filters}")
        return filters

    # --- REFACTORED: Non-blocking Place Only ---
    def place_limit_maker(self, symbol: str, side: str, quantity: float, price: float) -> str:
        """
        Places a POST_ONLY limit order.

        Args:
            symbol: Trading pair symbol.
            side: 'BUY' or 'SELL'.
            quantity: Order quantity.
            price: Limit price.

        Returns:
            str: The order ID.
        """
        log.debug(f"[SPOT] Placing POST_ONLY {side} order: {quantity:.8f} {symbol} @ {price:.8f}")
        resp = self.client.new_order(
            symbol=symbol,
            side=side,
            type="LIMIT_MAKER",
            quantity=f"{quantity:.8f}",
            price=f"{price:.8f}"
        )
        log.debug(f"[SPOT] Order response: {resp}")
        return str(resp["orderId"])

    def cancel_open_orders(self, symbol: str) -> List[str]:
        """
        Cancels all open orders for a symbol.

        Args:
            symbol: Trading pair symbol.

        Returns:
            List[str]: List of cancelled order IDs.
        """
        log.debug(f"[SPOT] Attempting to cancel all open orders for {symbol}")
        try:
            open_orders = self.client.get_open_orders(symbol=symbol)
            cancelled_ids = []
            for o in open_orders:
                oid = o.get("orderId")
                try:
                    self.client.cancel_order(symbol=symbol, orderId=oid)
                    cancelled_ids.append(str(oid))
                    log.debug(f"[SPOT] Cancelled order {oid} for {symbol}")
                except Exception as e:
                    log.warning(f"[SPOT] Could not cancel order {oid} for {symbol}: {e}")
            log.debug(f"[SPOT] Cancelled {len(cancelled_ids)} orders for {symbol}")
            return cancelled_ids
        except Exception as e:
            log.error(f"[SPOT] Error cancelling open orders for {symbol}: {e}")
            return []
            
    def cancel(self, symbol: str, order_id: str) -> None:
        """
        Cancels a specific order.

        Args:
            symbol: Trading pair symbol.
            order_id: ID of the order to cancel.
        """
        log.debug(f"[SPOT] Attempting to cancel order {order_id} for {symbol}")
        try:
            self.client.cancel_order(symbol=symbol, orderId=order_id)
            log.debug(f"[SPOT] Order {order_id} cancelled successfully")
        except Exception as e:
            log.warning(f"[SPOT] Could not cancel order {order_id}: {e}")

    def check_order(self, symbol: str, order_id: str) -> Tuple[bool, float]:
        """
        Checks the status of an order.

        Args:
            symbol: Trading pair symbol.
            order_id: Order ID to check.

        Returns:
            Tuple[bool, float]: (is_filled, executed_qty).
        """
        log.debug(f"[SPOT] Checking order {order_id} for {symbol}")
        od = self.client.get_order(symbol=symbol, orderId=order_id)
        status = od.get("status","")
        filled = float(od.get("executedQty","0"))
        log.debug(f"[SPOT] Order {order_id} status: {status}, filled: {filled}")
        return (status == "FILLED"), filled

    def market_order(self, symbol: str, side: str, quantity: float) -> str:
        """
        Executes a market order.

        Args:
            symbol: Trading pair symbol.
            side: 'BUY' or 'SELL'.
            quantity: Order quantity.

        Returns:
            str: The order ID.
        """
        log.debug(f"[SPOT] Placing MARKET {side} order: {quantity:.8f} {symbol}")
        resp = self.client.new_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=f"{quantity:.8f}"
        )
        log.debug(f"[SPOT] Market order response: {resp}")
        return str(resp["orderId"])
    
    def get_funding_rate(self, symbol: str = "ETHUSDT") -> float:
        """
        Fetches the current funding rate for the equivalent Futures symbol.
        Note: Spot markets don't have funding, but we use this for signal generation.

        Args:
            symbol: Futures symbol (e.g. 'ETHUSDT').

        Returns:
            float: Funding rate as a percentage (e.g. 0.01 for 0.01%).
        """
        log.debug(f"[SPOT] Fetching funding rate for {symbol}")
        try:
            url = "https://fapi.binance.com/fapi/v1/premiumIndex"
            params = {"symbol": symbol}
            resp = requests.get(url, params=params, timeout=2)
            data = resp.json()
            raw_rate = float(data.get("lastFundingRate", 0.0))
            return raw_rate * 100.0
        except Exception:
            return 0.0