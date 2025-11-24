from __future__ import annotations
import time
import requests
from typing import Any, Dict, List, Tuple
from binance.spot import Spot
from .exchange_adapter import ExchangeAdapter, Book, Filters

class BinanceSpotAdapter(ExchangeAdapter):
    def __init__(self, client: Spot, public_client: Spot | None = None, timeout: int = 5000):
        self.client = client
        self.public_client = public_client or Spot(timeout=timeout)   

    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[Dict[str, Any]]:
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
        try:
            data = self.public_client.ticker_price(symbol=symbol)
            return float(data["price"])
        except Exception:
            raise

    def get_mid(self, symbol: str) -> float:
        book = self.get_book(symbol)
        return 0.5 * (book.best_bid + book.best_ask)

    def get_book(self, symbol: str) -> Book:
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
        info = self.client.exchange_info(symbol=symbol)
        s = info["symbols"][0]
        return s

    def get_filters(self, symbol: str) -> Filters:
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

        return Filters(step_size=step_size, tick_size=tick_size, min_notional=min_notional)

    # --- REFACTORED: Non-blocking Place Only ---
    def place_limit_maker(self, symbol: str, side: str, quantity: float, price: float) -> str:
        """
        Places a LIMIT_MAKER (Post-Only) order.
        Returns order_id immediately. Does NOT wait or sleep.
        """
        resp = self.client.new_order(
            symbol=symbol, 
            side=side, 
            type="LIMIT_MAKER",
            timeInForce="GTC", 
            quantity=f"{quantity:.8f}", 
            price=f"{price:.8f}",
            newOrderRespType="RESULT"
        )
        return str(resp.get("orderId") or resp.get("clientOrderId"))

    def cancel(self, symbol: str, order_id: str) -> None:
        try:
            self.client.cancel_order(symbol=symbol, orderId=order_id)
        except Exception:
            pass

    def check_order(self, symbol: str, order_id: str) -> Tuple[bool, float]:
        od = self.client.get_order(symbol=symbol, orderId=order_id)
        status = od.get("status","")
        filled = float(od.get("executedQty","0"))
        return (status == "FILLED"), filled

    def market_order(self, symbol: str, side: str, quantity: float) -> str:
        resp = self.client.new_order(symbol=symbol, side=side, type="MARKET",
                                     quantity=f"{quantity:.8f}", newOrderRespType="FULL")
        return str(resp.get("orderId") or resp.get("clientOrderId"))
    
    def get_funding_rate(self, symbol: str = "ETHUSDT") -> float:
        try:
            url = "https://fapi.binance.com/fapi/v1/premiumIndex"
            params = {"symbol": symbol}
            resp = requests.get(url, params=params, timeout=2)
            data = resp.json()
            raw_rate = float(data.get("lastFundingRate", 0.0))
            return raw_rate * 100.0
        except Exception:
            return 0.0