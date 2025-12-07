from __future__ import annotations
from typing import Optional, Literal, Any, Dict
from pydantic import BaseModel, Field, root_validator
import json

Interval = Literal["1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d"]

class Fees(BaseModel):
    """
    Fee configuration settings.
    """
    maker_fee: float = Field(..., ge=0.0, le=0.01, description="Maker fee rate (e.g. 0.0002 for 0.02%)")
    taker_fee: float = Field(..., ge=0.0, le=0.02, description="Taker fee rate (e.g. 0.0004 for 0.04%)")
    slippage_bps: float = Field(0.0, ge=0.0, le=100.0, description="Simulated slippage in basis points")
    bnb_discount: float = Field(0.0, ge=0.0, le=1.0, description="Discount rate if paying fees in BNB")
    pay_fees_in_bnb: bool = Field(True, description="Whether to simulate paying fees in BNB")

class Strategy(BaseModel):
    """
    Strategy configuration settings.
    """
    # --- Strategy Selector ---
    strategy_type: Literal["mean_reversion", "trend", "meta"] = Field("mean_reversion", description="Strategy type to execute")

    # --- Mean Reversion Params ---
    trend_kind: Literal["sma","roc"] = Field("roc", description="Trend indicator for Mean Reversion (SMA or ROC)")
    trend_lookback: int = Field(200, ge=1, le=10000, description="Lookback period for trend baseline")
    flip_band_entry: float = Field(0.025, ge=0.0, le=1.0, description="Entry band deviation (e.g. 0.025 for 2.5%)")
    flip_band_exit: float = Field(0.015, ge=0.0, le=1.0, description="Exit band deviation (e.g. 0.015 for 1.5%)")
    vol_window: int = Field(45, ge=1, le=10000, description="Volatility calculation window")
    vol_adapt_k: float = Field(0.0, ge=0.0, le=1.0, description="Volatility adaptation factor (0.0 to disable)")
    target_vol: float = Field(0.0, ge=0.0, le=10.0, description="Target volatility scaling factor (0.0 to disable)")
    min_mult: float = Field(0.5, ge=0.0, le=10.0, description="Minimum leverage multiplier (if vol scaling active)")
    max_mult: float = Field(1.5, ge=0.0, le=10.0, description="Maximum leverage multiplier (if vol scaling active)")
    gate_window_days: int = Field(0, ge=0, le=3660, description="Trend gate window in days (0 to disable)")
    gate_roc_threshold: float = Field(0.0, ge=0.0, le=1.0, description="Trend gate ROC threshold")
    
    # --- Trend Strategy Params ---
    fast_period: int = Field(50, ge=1, description="Fast moving average period")
    slow_period: int = Field(200, ge=1, description="Slow moving average period")
    ma_type: Literal["sma", "ema"] = Field("ema", description="Moving average type")
    
    # --- Meta Strategy Params ---
    adx_threshold: float = Field(25.0, ge=0.0, le=100.0, description="ADX threshold for regime switching")

    # --- Shared / Global ---
    cooldown_minutes: int = Field(0, ge=0, le=100000, description="Minimum time between trade direction flips")
    step_allocation: float = Field(0.33, ge=0.0, le=1.0, description="Portion of target weight deviation to close per step")
    max_position: float = Field(1.0, ge=0.0, le=1.0, description="Maximum exposure (1.0 = 100%)")
    long_only: bool = Field(True, description="If True, only allow long positions (Spot mode)")
    rebalance_threshold_w: float = Field(0.0, ge=0.0, le=1.0, description="Minimum weight deviation to trigger a trade")
    profit_lock_dd: float = Field(0.0, ge=0.0, le=1.0, description="Profit locking drawdown threshold (unused)")
    vol_scaled_step: bool = Field(False, description="Scale step allocation by volatility")
    
    funding_limit_long: float = Field(0.05, ge=0.0, le=1.0, description="Max funding rate to allow long positions")
    funding_limit_short: float = Field(-0.05, ge=-1.0, le=0.0, description="Min funding rate to allow short positions")
    
    # --- Overrides for Meta Strategy ---
    mean_reversion_overrides: Dict[str, Any] = Field({}, description="Parameter overrides for Mean Reversion when in Meta mode")
    trend_overrides: Dict[str, Any] = Field({}, description="Parameter overrides for Trend Strategy when in Meta mode")

class Execution(BaseModel):
    """
    Execution configuration settings.
    """
    interval: Interval = Field("15m", description="Candle interval for strategy updates")
    poll_sec: int = Field(5, ge=1, le=300, description="Main loop poll interval in seconds")
    ttl_sec: int = Field(30, ge=5, le=600, description="Order time-to-live in seconds (unused)")
    taker_fallback: bool = Field(False, description="Allow Taker orders if Maker fails")
    max_taker_btc: float = Field(0.002, ge=0.0, le=1.0, description="Max size for taker orders")
    max_spread_bps_for_taker: float = Field(2.0, ge=0.0, le=100.0, description="Max spread allowed for taker execution")
    min_trade_frac: float = Field(0.0015, ge=0.0, le=1.0, description="Minimum trade size as fraction of portfolio")
    min_trade_floor_btc: float = Field(0.0, ge=0.0, le=10.0, description="Absolute minimum trade size in BTC")
    min_trade_cap_btc: float = Field(0.0, ge=0.0, le=10.0, description="Maximum trade size cap in BTC")
    min_trade_btc: Optional[float] = Field(None, description="Legacy override for min trade size")
    exchange_type: Literal["spot", "futures"] = Field("spot", description="Exchange mode: 'spot' or 'futures'")
    leverage: int = Field(1, ge=1, le=20, description="Leverage multiplier (Futures only)")

class Risk(BaseModel):
    """
    Risk management configuration settings.
    """
    basis_btc: float = Field(0.0, ge=0.0, le=100000.0, description="Initial capital basis in BTC")
    max_daily_loss_btc: float = Field(0.0, ge=0.0, le=100.0, description="Max allowed daily loss in BTC")
    max_dd_btc: float = Field(0.0, ge=0.0, le=100.0, description="Max allowed drawdown in BTC (from HWM)")
    max_daily_loss_frac: float = Field(0.0, ge=0.0, le=1.0, description="Max daily loss as fraction of equity")
    max_dd_frac: float = Field(0.0, ge=0.0, le=1.0, description="Max drawdown as fraction of equity")
    risk_mode: Literal["fixed_basis", "dynamic"] = Field("fixed_basis", description="Risk calculation mode")
    drawdown_reset_days: float = Field(0.0, ge=0.0, le=365.0, description="Days to wait for Phoenix Protocol reset")
    drawdown_reset_score: float = Field(25.0, ge=0.0, le=100.0, description="Trend score required for Phoenix Protocol reset")

class AppConfig(BaseModel):
    """
    Root application configuration.
    """
    fees: Fees
    strategy: Strategy
    execution: Execution
    risk: Risk
    
    # FIX ITEM 7: Robust Pre-Validation for Legacy Configs
    @root_validator(pre=True)
    def flatten_compatibility(cls, values):
        """
        Pre-validator to handle legacy flat configuration files.
        Maps flat keys to nested Pydantic models.
        """
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
    """
    Loads configuration from a JSON file.

    Args:
        path: Path to the JSON configuration file.

    Returns:
        AppConfig: Validated application configuration object.
    """
    with open(path, "r") as f:
        data = json.load(f)
    return AppConfig(**data)