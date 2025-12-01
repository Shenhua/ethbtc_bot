from __future__ import annotations
from typing import Optional, Literal, Any, Dict
from pydantic import BaseModel, Field
import json

Interval = Literal["1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d"]

class Fees(BaseModel):
    maker_fee: float = Field(..., ge=0.0, le=0.01)
    taker_fee: float = Field(..., ge=0.0, le=0.02)
    slippage_bps: float = Field(0.0, ge=0.0, le=100.0)
    bnb_discount: float = Field(0.0, ge=0.0, le=1.0)
    pay_fees_in_bnb: bool = True

class Strategy(BaseModel):
    # --- Strategy Selector ---
    # "mean_reversion" (default), "trend", or "meta"
    strategy_type: Literal["mean_reversion", "trend", "meta"] = "mean_reversion"

    # --- Mean Reversion Params ---
    trend_kind: Literal["sma","roc"] = "roc"
    trend_lookback: int = Field(200, ge=1, le=10000)
    flip_band_entry: float = Field(0.025, ge=0.0, le=1.0)
    flip_band_exit: float = Field(0.015, ge=0.0, le=1.0)
    vol_window: int = Field(45, ge=1, le=10000)
    vol_adapt_k: float = Field(0.0, ge=0.0, le=1.0)
    target_vol: float = Field(0.0, ge=0.0, le=10.0)
    min_mult: float = Field(0.5, ge=0.0, le=10.0)
    max_mult: float = Field(1.5, ge=0.0, le=10.0)
    gate_window_days: int = Field(0, ge=0, le=3660)
    gate_roc_threshold: float = Field(0.0, ge=0.0, le=1.0)
    
    # --- Trend Strategy Params ---
    fast_period: int = Field(50, ge=1)
    slow_period: int = Field(200, ge=1)
    ma_type: Literal["sma", "ema"] = "ema"
    
    # --- Meta Strategy Params ---
    adx_threshold: float = Field(25.0, ge=0.0, le=100.0)

    # --- Shared / Global ---
    cooldown_minutes: int = Field(0, ge=0, le=100000)
    step_allocation: float = Field(0.33, ge=0.0, le=1.0)
    max_position: float = Field(1.0, ge=0.0, le=1.0)
    long_only: bool = True
    rebalance_threshold_w: float = Field(0.0, ge=0.0, le=1.0)
    profit_lock_dd: float = Field(0.0, ge=0.0, le=1.0)
    vol_scaled_step: bool = False
    
    funding_limit_long: float = Field(0.05, ge=0.0, le=1.0)
    funding_limit_short: float = Field(-0.05, ge=-1.0, le=0.0)
    
    # --- Overrides for Meta Strategy ---
    mean_reversion_overrides: Dict[str, Any] = {}
    trend_overrides: Dict[str, Any] = {}

class Execution(BaseModel):
    interval: Interval = "15m"
    poll_sec: int = Field(5, ge=1, le=300)
    ttl_sec: int = Field(30, ge=5, le=600)
    taker_fallback: bool = False
    max_taker_btc: float = Field(0.002, ge=0.0, le=1.0)
    max_spread_bps_for_taker: float = Field(2.0, ge=0.0, le=100.0)
    min_trade_frac: float = Field(0.0015, ge=0.0, le=1.0)
    min_trade_floor_btc: float = Field(0.0, ge=0.0, le=10.0)
    min_trade_cap_btc: float = Field(0.0, ge=0.0, le=10.0)
    min_trade_btc: Optional[float] = None

class Risk(BaseModel):
    basis_btc: float = Field(0.0, ge=0.0, le=100000.0)
    max_daily_loss_btc: float = Field(0.0, ge=0.0, le=100.0)
    max_dd_btc: float = Field(0.0, ge=0.0, le=100.0)
    max_daily_loss_frac: float = Field(0.0, ge=0.0, le=1.0)
    max_dd_frac: float = Field(0.0, ge=0.0, le=1.0)
    risk_mode: Literal["fixed_basis", "dynamic"] = "fixed_basis"

class AppConfig(BaseModel):
    fees: Fees
    strategy: Strategy
    execution: Execution
    risk: Risk
    
    # Legacy Coercion Logic
    @staticmethod
    def coerce_legacy(d: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        # Full legacy logic preserved for compatibility with old configs
        fees_keys = {"maker_fee","taker_fee","slippage_bps","bnb_discount","pay_fees_in_bnb"}
        strat_keys = {"trend_kind","trend_lookback","flip_band_entry","flip_band_exit","vol_window","vol_adapt_k",
                      "target_vol","min_mult","max_mult","cooldown_minutes","step_allocation","max_position",
                      "gate_window_days","gate_roc_threshold","rebalance_threshold_w","vol_scaled_step","profit_lock_dd","long_only",
                      "fast_period","slow_period","ma_type","adx_threshold","strategy_type","funding_limit_long","funding_limit_short"}
        exec_keys = {"interval","poll_sec","ttl_sec","taker_fallback","max_taker_btc","max_spread_bps_for_taker",
                     "min_trade_frac","min_trade_floor_btc","min_trade_cap_btc","min_trade_btc"}
        risk_keys = {"basis_btc","max_daily_loss_btc","max_dd_btc","max_daily_loss_frac","max_dd_frac","risk_mode"}
        
        out["fees"] = {k: d[k] for k in fees_keys if k in d}
        out["strategy"] = {k: d[k] for k in strat_keys if k in d}
        out["execution"] = {k: d[k] for k in exec_keys if k in d}
        out["risk"] = {k: d[k] for k in risk_keys if k in d}
        
        # Defaults
        if "maker_fee" not in out["fees"]: out["fees"]["maker_fee"] = d.get("maker_fee", 0.0002)
        if "taker_fee" not in out["fees"]: out["fees"]["taker_fee"] = d.get("taker_fee", 0.0004)
        if "basis_btc" not in out["risk"]: out["risk"]["basis_btc"] = d.get("basis_btc", 0.1)
        if "interval" not in out["execution"]: out["execution"]["interval"] = d.get("interval","15m")
        
        return out

def load_config(path: str) -> AppConfig:
    with open(path, "r") as f:
        data = json.load(f)
    
    # Check if it's legacy flat structure
    if "strategy" not in data or "execution" not in data:
        data = AppConfig.coerce_legacy(data)
        
    return AppConfig(**data)