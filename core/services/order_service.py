from __future__ import annotations
import pandas as pd
import time
from typing import Optional, Dict, Any, Tuple
from core.resilience import CircuitBreaker, CircuitBreakerOpen, retry_api_call
from core.exchange_adapter import ExchangeAdapter
from core.metrics import (
    mark_asset_price_usd, mark_futures_risk,
    EXPOSURE_NOTIONAL, EXPOSURE_SIGNAL_WEIGHT,
    POSITION_STALE
)
from core.logging import get_logger

log = get_logger("order_service")

class OrderService:
    """
    Service for managing balances, positions, and order execution.
    
    Handles:
    - Balance fetching (Spot/Futures)
    - Position tracking with staleness checks
    - Risk metrics (Margin, Liquidation)
    - Order placement and tracking (Phase 2)
    """
    
    def __init__(
        self,
        adapter: Any,  # BinanceSpotAdapter or BinanceFuturesAdapter (duck typed)
        symbol: str,
        quote_asset: str,
        is_futures: bool,
        leverage: int,
        circuit_breaker: CircuitBreaker,
        alerter: Any,
        instance_name: str
    ):
        self.adapter = adapter
        self.symbol = symbol
        self.quote_asset = quote_asset
        self.is_futures = is_futures
        self.leverage = leverage
        self.circuit_breaker = circuit_breaker
        self.alerter = alerter
        self.instance_name = instance_name

    def get_account_state(self, state: Dict[str, Any], bar_dt: pd.Timestamp, price: float) -> Dict[str, Any]:
        """
        Fetch current balances, positions, and wealth.
        Updates weights and risk metrics.
        Returns a dictionary of current account state.
        """
        res = {
            "quote_bal": 0.0,
            "current_position": 0.0,
            "W": 0.0,
            "cur_w": 0.0,
            "position_stale": False,
            "margin_util_pct": 0.0,
            "liq_dist_pct": 0.0
        }
        
        try:
            if self.is_futures:
                # 1. Fetch Balance
                res["quote_bal"] = self.adapter.get_account_balance(self.quote_asset)
                
                # 2. Risk Metrics
                try:
                    acc_info = self.adapter.client.account()
                    total_maint = float(acc_info.get('totalMaintMargin', 0))
                    total_bal = float(acc_info.get('totalMarginBalance', 1e-9))
                    res["margin_util_pct"] = (total_maint / total_bal) * 100 if total_bal > 0 else 0.0

                    pos_risks = self.adapter.client.get_position_risk(symbol=self.symbol)
                    risk_entry = next((r for r in pos_risks if r['symbol'] == self.symbol), None)
                    if risk_entry:
                         liq_px = float(risk_entry.get('liquidationPrice', 0))
                         mark_px = float(risk_entry.get('markPrice', 0))
                         if liq_px > 0 and mark_px > 0:
                             res["liq_dist_pct"] = abs(mark_px - liq_px) / mark_px * 100
                    
                    mark_futures_risk(self.instance_name, res["margin_util_pct"], res["liq_dist_pct"], self.symbol)
                except Exception as e:
                    log.debug("Futures risk metric fetch failed: %s", e)

                # 3. Position Tracking
                try:
                    res["current_position"] = retry_api_call(
                        self.adapter.get_position, self.symbol,
                        max_attempts=2,
                        min_wait=1.0,
                        max_wait=5.0,
                        circuit_breaker=self.circuit_breaker
                    )
                    state["last_known_position"] = res["current_position"]
                    state["last_position_fetch_ts"] = bar_dt.isoformat()
                except CircuitBreakerOpen as e:
                    log.critical("🚨 Position fetch blocked by circuit breaker: %s", e)
                    res["current_position"] = state.get("last_known_position", 0.0)
                    res["position_stale"] = True
                except Exception as e:
                    log.error("Position fetch failed, using last known: %s", e)
                    res["current_position"] = state.get("last_known_position", 0.0)
                    last_fetch_ts = state.get("last_position_fetch_ts")
                    if last_fetch_ts:
                        position_age = bar_dt - pd.Timestamp(last_fetch_ts)
                        if position_age > pd.Timedelta("15min"):
                            log.error("⚠️ Position data too stale (%s ago).", position_age)
                            res["position_stale"] = True

                res["W"] = res["quote_bal"]
                
            else:
                # SPOT MODE (Simplified for now, can expand later)
                res["quote_bal"] = self.adapter.get_account_balance(self.quote_asset)
                res["current_position"] = self.adapter.get_account_balance(self.symbol.replace(self.quote_asset, ""))
                res["W"] = res["quote_bal"] + (res["current_position"] * price)

            # 4. Calculate Weights
            position_val = res["current_position"] * price
            res["cur_w"] = position_val / res["W"] if res["W"] > 1e-6 else 0.0
            
            signal_weight = res["cur_w"] / self.leverage if self.leverage > 0 else res["cur_w"]
            
            # 5. Metrics
            EXPOSURE_NOTIONAL.labels(instance=self.instance_name).set(res["cur_w"] * 100)
            EXPOSURE_SIGNAL_WEIGHT.labels(instance=self.instance_name).set(signal_weight * 100)
            POSITION_STALE.labels(instance=self.instance_name).set(1.0 if res["position_stale"] else 0.0)
            
            return res

        except Exception as e:
            log.error("Critical error in OrderService.get_account_state: %s", e)
            res["position_stale"] = True
            return res
