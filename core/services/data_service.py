from __future__ import annotations
import logging
import pandas as pd
import time
from typing import Optional, Dict, Any
from core.resilience import CircuitBreaker, CircuitBreakerOpen, retry_api_call
from core.exchange_adapter import ExchangeAdapter

log = logging.getLogger("data_service")

class DataService:
    """
    Service for fetching and cleaning market data.
    
    Handles:
    - Kline fetching with retry logic & circuit breakers
    - Data cleaning (numeric conversion)
    - Deduplication
    - Dropping incomplete (open) candles
    """
    
    def __init__(
        self, 
        adapter: ExchangeAdapter, 
        symbol: str, 
        interval: str,
        circuit_breaker: CircuitBreaker,
        alerter: Any
    ):
        self.adapter = adapter
        self.symbol = symbol
        self.interval = interval
        self.circuit_breaker = circuit_breaker
        self.alerter = alerter

    def get_closed_klines(self, limit: int = 600) -> Optional[pd.DataFrame]:
        """
        Fetch klines and return a cleaned DataFrame of CLOSED candles only.
        Returns None if fetching fails or no closed candles are available.
        """
        try:
            ks = retry_api_call(
                self.adapter.get_klines, 
                self.symbol, self.interval, limit=limit,
                max_attempts=3,
                min_wait=2.0,
                max_wait=30.0,
                circuit_breaker=self.circuit_breaker
            )
            df = pd.DataFrame(ks)
            
            if df.empty:
                log.warning("Received empty kline list from adapter")
                return None

            # 1. Ensure numeric types
            cols = ["open", "high", "low", "close", "volume"]
            for col in cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # 2. Set Index
            if "close_time" in df.columns:
                df.index = pd.to_datetime(df["close_time"], unit="ms", utc=True)
            
            # 3. Deduplication
            df = df[~df.index.duplicated(keep='last')]
            df = df.sort_index()

            # 4. Drop incomplete candle
            # Binance returns the currently OPEN candle as the last element.
            now_utc = pd.Timestamp.now(tz="UTC")
            if df.index[-1] > now_utc:
                log.debug("Dropping incomplete candle at %s (now=%s)", df.index[-1], now_utc)
                df = df.iloc[:-1]
            
            if df.empty:
                log.warning("No closed candles available after filtering.")
                return None
                
            return df

        except CircuitBreakerOpen as e:
            log.critical("🚨 CIRCUIT BREAKER OPEN: %s", e)
            self.alerter.send(
                f"🚨 CIRCUIT BREAKER [{self.symbol}]\nAPI failures exceeded limit. Bot in safe mode.", 
                level="CRITICAL"
            )
            return None
        except Exception as e:
            log.error("Failed to fetch klines after retries: %s", e)
            return None
