from __future__ import annotations
from typing import Optional, Literal, Any, Dict
from pydantic import BaseModel, Field, root_validator
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
    max_position: float = Field(1.0, ge=0.0)
    long_only: bool = True
    rebalance_threshold_w: float = Field(0.0, ge=0.0)
    profit_lock_dd: float = Field(0.0, ge=0.0, le=1.0)
    # Volatility
    vol_scaled_step: bool = False
    # Dynamic Position Sizing
    position_sizing_mode: str = Field("static", pattern="^(static|volatility|kelly)$")
    position_sizing_target_vol: float = Field(0.5, ge=0.1, le=2.0)
    position_sizing_min_step: float = Field(0.1, ge=0.0, le=1.0)
    position_sizing_max_step: float = Field(1.0, ge=0.0, le=1.0)
    # Kelly Criterion params (optional)
    kelly_win_rate: float = Field(0.55, ge=0.0, le=1.0)
    kelly_avg_win: float = Field(0.02, ge=0.0)
    kelly_avg_loss: float = Field(0.01, ge=0.0)
    
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
    exchange_type: Literal["spot", "futures"] = "spot"  
    leverage: int = Field(1, ge=1, le=20)              

class Risk(BaseModel):
    basis_btc: float = Field(0.0, ge=0.0, le=100000.0)
    max_daily_loss_btc: float = Field(0.0, ge=0.0, le=100.0)
    max_dd_btc: float = Field(0.0, ge=0.0, le=100.0)
    max_daily_loss_frac: float = Field(0.0, ge=0.0, le=1.0)
    max_dd_frac: float = Field(0.0, ge=0.0, le=1.0)
    risk_mode: Literal["fixed_basis", "dynamic"] = "fixed_basis"
    drawdown_reset_days: float = Field(0.0, ge=0.0, le=365.0)
    drawdown_reset_score: float = Field(25.0, ge=0.0, le=100.0)

class AppConfig(BaseModel):
    fees: Fees
    strategy: Strategy
    execution: Execution
    risk: Risk
    
    # FIX ITEM 7: Robust Pre-Validation for Legacy Configs
    @root_validator(pre=True)
    def flatten_compatibility(cls, values):
        # If structure is already nested (has 'fees', 'strategy', etc), return as is
        if "fees" in values and "strategy" in values:
            return values
            
        # Otherwise, assume flat legacy config and map fields
        fees_data = {k: v for k, v in values.items() if k in Fees.__fields__}
        
        # Strategy fields might have overrides (e.g. trend_kind)
        strat_data = {k: v for k, v in values.items() if k in Strategy.__fields__}
        
        # Specific legacy defaults
        if "strategy_type" not in strat_data: 
            strat_data["strategy_type"] = "mean_reversion"
            
        exec_data = {k: v for k, v in values.items() if k in Execution.__fields__}
        risk_data = {k: v for k, v in values.items() if k in Risk.__fields__}
        
        # Handle Basis legacy
        if "basis_btc" in values and "basis_btc" not in risk_data:
            risk_data["basis_btc"] = values["basis_btc"]

        return {
            "fees": fees_data,
            "strategy": strat_data,
            "execution": exec_data,
            "risk": risk_data
        }

def load_config(path: str) -> AppConfig:
    with open(path, "r") as f:
        data = json.load(f)
    return AppConfig(**data)