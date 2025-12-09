"""
Strategy Factory - Shared module for building strategies from config.

This is the SINGLE SOURCE OF TRUTH for strategy construction.
Used by both backtest (ethbtc_accum_bot.py) and live executor (live_executor.py).

DO NOT duplicate this logic elsewhere. Any changes to config handling
MUST be made here to maintain backtest-live parity.
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, Optional
import pandas as pd

from core.config_schema import AppConfig
from core.ethbtc_accum_bot import EthBtcStrategy, StratParams
from core.trend_strategy import TrendStrategy, TrendParams
from core.meta_strategy import MetaStrategy


def merge_strategy_params(cfg: AppConfig) -> Dict[str, Any]:
    """
    Merge base strategy config with overrides for meta strategy.
    
    Returns a dict with:
        - 'strategy_type': str ("mean_reversion", "trend", or "meta")
        - 'mr_params': dict with all MR params (merged with overrides if meta)
        - 'tr_params': dict with all Trend params (merged with overrides if meta)
        - 'base_params': dict with unmerged base config
    
    This function ensures that live executor uses the EXACT same 
    merged params as the backtest.
    """
    base = cfg.strategy.model_dump() if hasattr(cfg.strategy, 'model_dump') else dict(cfg.strategy)
    strategy_type = base.get("strategy_type", "mean_reversion")
    
    if strategy_type == "meta":
        mr_opts = base.get("mean_reversion_overrides", {})
        tr_opts = base.get("trend_overrides", {})
        
        # Merge: base + overrides (overrides win)
        mr_merged = {**base, **mr_opts}
        tr_merged = {**base, **tr_opts}
    else:
        # Non-meta: both use base params
        mr_merged = base.copy()
        tr_merged = base.copy()
    
    return {
        "strategy_type": strategy_type,
        "mr_params": mr_merged,
        "tr_params": tr_merged,
        "base_params": base,
    }


def build_mr_params(merged: Dict[str, Any]) -> StratParams:
    """
    Build StratParams (Mean Reversion) from merged dict.
    
    All params are extracted from the merged dict with explicit type conversion
    to ensure consistent behavior between backtest and live.
    """
    return StratParams(
        trend_kind=merged.get("trend_kind", "roc"),
        trend_lookback=int(merged.get("trend_lookback", 200)),
        flip_band_entry=float(merged.get("flip_band_entry", 0.025)),
        flip_band_exit=float(merged.get("flip_band_exit", 0.015)),
        vol_window=int(merged.get("vol_window", 60)),
        vol_adapt_k=float(merged.get("vol_adapt_k", 0.0)),
        bar_interval_minutes=int(merged.get("bar_interval_minutes", 15)),
        target_vol=float(merged.get("target_vol", 0.5)),
        min_mult=float(merged.get("min_mult", 0.5)),
        max_mult=float(merged.get("max_mult", 1.5)),
        cooldown_minutes=int(merged.get("cooldown_minutes", 60)),
        step_allocation=float(merged.get("step_allocation", 0.33)),
        max_position=float(merged.get("max_position", 1.0)),
        long_only=bool(int(merged.get("long_only", 1))),
        rebalance_threshold_w=float(merged.get("rebalance_threshold_w", 0.0)),
        min_trade_btc=float(merged.get("min_trade_btc", 0.0)),
        gate_window_days=int(merged.get("gate_window_days", 0)),
        gate_roc_threshold=float(merged.get("gate_roc_threshold", 0.0)),
        profit_lock_dd=float(merged.get("profit_lock_dd", 0.0)),
        vol_scaled_step=bool(merged.get("vol_scaled_step", False)),
        position_sizing_mode=merged.get("position_sizing_mode", "static"),
        position_sizing_target_vol=float(merged.get("position_sizing_target_vol", 0.5)),
        position_sizing_min_step=float(merged.get("position_sizing_min_step", 0.1)),
        position_sizing_max_step=float(merged.get("position_sizing_max_step", 1.0)),
        kelly_win_rate=float(merged.get("kelly_win_rate", 0.55)),
        kelly_avg_win=float(merged.get("kelly_avg_win", 0.02)),
        kelly_avg_loss=float(merged.get("kelly_avg_loss", 0.01)),
        funding_limit_long=float(merged.get("funding_limit_long", 0.05)),
        funding_limit_short=float(merged.get("funding_limit_short", -0.05)),
        fast_period=int(merged.get("fast_period", 50)),
        slow_period=int(merged.get("slow_period", 200)),
        ma_type=merged.get("ma_type", "ema"),
        adx_threshold=float(merged.get("adx_threshold", 25.0)),
        strategy_type=merged.get("strategy_type", "mean_reversion"),
    )


def build_tr_params(merged: Dict[str, Any]) -> TrendParams:
    """
    Build TrendParams from merged dict.
    
    All params are extracted from the merged dict with explicit type conversion
    to ensure consistent behavior between backtest and live.
    """
    return TrendParams(
        fast_period=int(merged.get("fast_period", 50)),
        slow_period=int(merged.get("slow_period", 200)),
        ma_type=merged.get("ma_type", "ema"),
        cooldown_minutes=int(merged.get("cooldown_minutes", 180)),
        step_allocation=float(merged.get("step_allocation", 1.0)),
        max_position=float(merged.get("max_position", 1.0)),
        long_only=bool(merged.get("long_only", True)),
        funding_limit_long=float(merged.get("funding_limit_long", 0.05)),
        funding_limit_short=float(merged.get("funding_limit_short", -0.05)),
        rebalance_threshold_w=float(merged.get("rebalance_threshold_w", 0.0)),
    )


def build_strategy(cfg: AppConfig, df: Optional[pd.DataFrame] = None) -> Tuple[Any, Dict[str, Any]]:
    """
    Build strategy from config - SHARED between backtest and live executor.
    
    Args:
        cfg: AppConfig instance
        df: Optional dataframe (for compatibility, not used in construction)
    
    Returns:
        Tuple of (strategy_instance, merged_cfg_dict)
    
    This is the ONLY place strategies should be constructed from config.
    """
    merged = merge_strategy_params(cfg)
    
    if merged["strategy_type"] == "trend":
        tr_p = build_tr_params(merged["tr_params"])
        return TrendStrategy(tr_p), merged
    
    if merged["strategy_type"] == "meta":
        mr_p = build_mr_params(merged["mr_params"])
        tr_p = build_tr_params(merged["tr_params"])
        adx_thresh = float(merged["mr_params"].get("adx_threshold", 25.0))
        # ML Regime Detection params
        use_ml = merged["mr_params"].get("use_ml_regime", False)
        ml_path = merged["mr_params"].get("ml_model_path", "models/regime_classifier_v1.pkl")
        return MetaStrategy(
            mr_p, tr_p, 
            adx_threshold=adx_thresh,
            use_ml_regime=use_ml,
            ml_model_path=ml_path,
        ), merged
    
    # Default: Mean Reversion
    mr_p = build_mr_params(merged["mr_params"])
    return EthBtcStrategy(mr_p), merged


def get_active_params(merged_cfg: Dict[str, Any], current_regime_score: float) -> Dict[str, Any]:
    """
    Get the active params based on current regime.
    
    For meta strategy, returns mr_params or tr_params based on 
    whether current_regime_score exceeds adx_threshold.
    
    For non-meta strategies, returns mr_params.
    """
    if merged_cfg["strategy_type"] != "meta":
        return merged_cfg["mr_params"]
    
    adx_thresh = float(merged_cfg["mr_params"].get("adx_threshold", 25.0))
    
    if current_regime_score > adx_thresh:
        return merged_cfg["tr_params"]
    else:
        return merged_cfg["mr_params"]
