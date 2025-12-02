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
        
        # Futures often don't have MIN_NOTIONAL in the same way, but usually ~5 USDT
        # We can default to 5.0 if not found
        min_notional = 5.0 

        return Filters(
            step_size=float(lot_filter.get("stepSize", "0")),
            tick_size=float(price_filter.get("tickSize", "0")),
            min_notional=min_notional
        )

    def get_usd_price(self, symbol: str) -> float:
        # For USDS-M, the price IS the USD price usually
        if "USDT" in symbol or "USDC" in symbol:
            t = self.client.ticker_price(symbol)
            return float(t["price"])
        return 0.0

    def get_funding_rate(self, symbol: str) -> float:
        # Futures specific: Real-time funding rate
        f = self.client.mark_price(symbol)
        return float(f["lastFundingRate"]) * 100.0

    # --- Execution ---

    def set_leverage(self, symbol: str, leverage: int):
        try:
            self.client.change_leverage(symbol, leverage)
        except ClientError as e:
            log.warning(f"Could not set leverage: {e}")

    def get_position(self, symbol: str) -> float:
        """Returns current position size (Signed: +Long, -Short)"""
        try:
            # We must fetch specific position risk
            # Note: This assumes One-Way Mode (not Hedge Mode)
            positions = self.client.account()["positions"]
            target = next((p for p in positions if p["symbol"] == symbol), None)
            if target:
                return float(target["positionAmt"])
            return 0.0
        except Exception as e:
            log.error(f"Error fetching position: {e}")
            return 0.0

    def market_order(self, symbol: str, side: str, quantity: float) -> str:
        # Futures Market Order
        resp = self.client.new_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=f"{quantity:.8f}" # Futures API is strict on precision
        )
        return str(resp["orderId"])

    def get_account_balance(self, asset: str) -> float:
        """Get Margin Balance (Wallet + PnL) for the asset"""
        acct = self.client.account()
        for a in acct["assets"]:
            if a["asset"] == asset:
                return float(a["marginBalance"]) # Use marginBalance to account for unrealized PnL
        return 0.0