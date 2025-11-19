from __future__ import annotations
from typing import Optional, Literal, Any, Dict
from pydantic import BaseModel, Field

Interval = Literal["1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d"]

class Fees(BaseModel):
    maker_fee: float = Field(..., ge=0.0, le=0.01)
    taker_fee: float = Field(..., ge=0.0, le=0.02)
    slippage_bps: float = Field(0.0, ge=0.0, le=100.0)
    bnb_discount: float = Field(0.0, ge=0.0, le=1.0)
    pay_fees_in_bnb: bool = True

class Strategy(BaseModel):
    trend_kind: Literal["sma","roc"] = "roc"
    trend_lookback: int = Field(..., ge=1, le=10000)
    flip_band_entry: float = Field(..., ge=0.0, le=1.0)
    flip_band_exit: float = Field(..., ge=0.0, le=1.0)
    vol_window: int = Field(45, ge=1, le=10000)
    vol_adapt_k: float = Field(0.0, ge=0.0, le=1.0)
    target_vol: float = Field(0.0, ge=0.0, le=10.0)
    min_mult: float = Field(0.5, ge=0.0, le=10.0)
    max_mult: float = Field(1.5, ge=0.0, le=10.0)
    cooldown_minutes: int = Field(0, ge=0, le=100000)
    step_allocation: float = Field(0.33, ge=0.0, le=1.0)
    max_position: float = Field(1.0, ge=0.0, le=1.0)
    gate_window_days: int = Field(0, ge=0, le=3660)
    gate_roc_threshold: float = Field(0.0, ge=0.0, le=1.0)
    rebalance_threshold_w: float = Field(0.0, ge=0.0, le=1.0)
    vol_scaled_step: bool = False
    profit_lock_dd: float = Field(0.0, ge=0.0, le=1.0)

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
    basis_btc: float = Field(0.0, ge=0.0, le=1000.0)

    # Absolute caps in BTC (0.0 = disabled)
    max_daily_loss_btc: float = Field(0.0, ge=0.0, le=100.0)
    max_dd_btc: float = Field(0.0, ge=0.0, le=100.0)

    # Fractional caps (0.02 = 2%; 0.0 = disabled)
    max_daily_loss_frac: float = Field(0.0, ge=0.0, le=1.0)
    max_dd_frac: float = Field(0.0, ge=0.0, le=1.0)

    # How to interpret the caps:
    # - "fixed_basis": BTC caps anchored to basis_btc
    # - "dynamic": fractions of current / peak equity
    risk_mode: Literal["fixed_basis", "dynamic"] = "fixed_basis"

class AppConfig(BaseModel):
    fees: Fees
    strategy: Strategy
    execution: Execution
    risk: Risk

    @staticmethod
    def coerce_legacy(d: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        fees_keys = {"maker_fee","taker_fee","slippage_bps","bnb_discount","pay_fees_in_bnb"}
        strat_keys = {"trend_kind","trend_lookback","flip_band_entry","flip_band_exit","vol_window","vol_adapt_k",
                      "target_vol","min_mult","max_mult","cooldown_minutes","step_allocation","max_position",
                      "gate_window_days","gate_roc_threshold","rebalance_threshold_w","vol_scaled_step","profit_lock_dd"}
        exec_keys = {"interval","poll_sec","ttl_sec","taker_fallback","max_taker_btc","max_spread_bps_for_taker",
                     "min_trade_frac","min_trade_floor_btc","min_trade_cap_btc","min_trade_btc"}
        risk_keys = {
            "basis_btc",
            "max_daily_loss_btc",
            "max_dd_btc",
            "max_daily_loss_frac",
            "max_dd_frac",
            "risk_mode",
        }
        out["fees"] = {k: d[k] for k in fees_keys if k in d}
        out["strategy"] = {k: d[k] for k in strat_keys if k in d}
        out["execution"] = {k: d[k] for k in exec_keys if k in d}
        out["risk"] = {k: d[k] for k in risk_keys if k in d}
        if "maker_fee" not in out["fees"]:
            out["fees"]["maker_fee"] = d.get("maker_fee", 0.0002)
        if "taker_fee" not in out["fees"]:
            out["fees"]["taker_fee"] = d.get("taker_fee", 0.0004)

        if "basis_btc" not in out["risk"]:
            out["risk"]["basis_btc"] = d.get("basis_btc", 0.1)
        if "max_daily_loss_frac" not in out["risk"]:
            out["risk"]["max_daily_loss_frac"] = d.get("max_daily_loss_frac", 0.0)
        if "max_dd_frac" not in out["risk"]:
            out["risk"]["max_dd_frac"] = d.get("max_dd_frac", 0.0)
        if "risk_mode" not in out["risk"]:
            out["risk"]["risk_mode"] = d.get("risk_mode", "fixed_basis")

        if "interval" not in out["execution"]:
            out["execution"]["interval"] = d.get("interval","15m")
        return out
INTERVAL_FROM_MINUTES = {
    1: "1m",
    3: "3m",
    5: "5m",
    15: "15m",
    30: "30m",
    60: "1h",
    120: "2h",
    240: "4h",
    360: "6h",
    480: "8h",
    720: "12h",
    1440: "1d",
}
def load_config(path: str) -> AppConfig:
    import json
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # --- Case 1: full new-style nested config (fees/strategy/execution/risk) ----
    if isinstance(raw, dict):
        # Already nested and has a proper strategy block
        if isinstance(raw.get("strategy"), dict):
            return AppConfig(**raw)

        # Backwards compatibility: explicit blocks already present
        if {"fees", "strategy", "execution", "risk"}.issubset(raw.keys()):
            return AppConfig(**raw)

        # --- Case 2: backtest-style config with "params" ------------------------
        # Shape like:
        # {
        #   "params": { trend_kind, trend_lookback, flip_band_entry, ... },
        #   "fees":   { ... },
        #   "risk":   { ... }
        # }
        if isinstance(raw.get("params"), dict):
            p = raw["params"]

            # Build a Strategy dict from params, ignoring fields that Strategy doesn't know
            strat_dict = {
                "trend_kind":          p.get("trend_kind", "roc"),
                "trend_lookback":      p["trend_lookback"],
                "flip_band_entry":     p["flip_band_entry"],
                "flip_band_exit":      p["flip_band_exit"],
                "vol_window":          p.get("vol_window", 45),
                "vol_adapt_k":         p.get("vol_adapt_k", 0.0),
                "target_vol":          p.get("target_vol", 0.0),
                "min_mult":            p.get("min_mult", 0.5),
                "max_mult":            p.get("max_mult", 1.5),
                "cooldown_minutes":    p.get("cooldown_minutes", 0),
                "step_allocation":     p.get("step_allocation", 0.33),
                "max_position":        p.get("max_position", 1.0),
                "gate_window_days":    p.get("gate_window_days", 0),
                "gate_roc_threshold":  p.get("gate_roc_threshold", 0.0),
                "rebalance_threshold_w": p.get("rebalance_threshold_w", 0.0),
                "vol_scaled_step":     p.get("vol_scaled_step", False),
                "profit_lock_dd":      p.get("profit_lock_dd", 0.0),
            }

            # Map bar_interval_minutes → Execution.interval
            bar_min = int(p.get("bar_interval_minutes", 15))
            interval = INTERVAL_FROM_MINUTES.get(bar_min, "15m")

            exec_dict = {
                "interval":                 interval,
                "poll_sec":                 5,
                "ttl_sec":                  30,
                "taker_fallback":           False,
                "max_taker_btc":            0.002,
                "max_spread_bps_for_taker": 2.0,
                "min_trade_frac":           0.0015,
                "min_trade_floor_btc":      0.0,
                "min_trade_cap_btc":        0.0,
                "min_trade_btc":            None,
            }

            app_raw = {
                "fees":     raw.get("fees", {}),
                "strategy": strat_dict,
                "execution": exec_dict,
                "risk":     raw.get("risk", {}),
            }
            return AppConfig(**app_raw)

    # --- Case 3: legacy flat config (top-level keys) ------------------------------
    return AppConfig(**AppConfig.coerce_legacy(raw))
