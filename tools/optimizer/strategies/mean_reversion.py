"""
Mean Reversion Strategy Optimizer

Adapter for the core Mean Reversion strategy (EthBtcStrategy).
Wraps search space and strategy instantiation into the Optimizer Framework.

Author: AI Audit System
Date: 2024-12-30
"""

from typing import Any, Dict
import optuna

from ..base import BaseOptimizer
from core.ethbtc_accum_bot import StratParams, EthBtcStrategy


class MeanReversionOptimizer(BaseOptimizer):
    """
    Optimizer adapter for the Mean Reversion strategy.
    
    Ported from tools/optimizer_cli.py
    """
    
    def get_strategy_name(self) -> str:
        return "Mean Reversion"
    
    def suggest_params(self, trial: optuna.Trial, **kwargs) -> Dict[str, Any]:
        """
        Define the parameter search space for the MR strategy.
        
        Args:
            trial: Optuna trial
            force_trend_kind: (Optional) Lock trend_kind
            force_sizing_mode: (Optional) Lock position_sizing_mode
            force_long_only: (Optional) Lock long_only
            long_only_mode: (Optional) "true", "false", or "both"
            
        Returns:
            Dict of parameters
        """
        # 1. Handle long_only mode (the "shorting switch")
        long_only_mode = kwargs.get("long_only_mode", "both")
        if long_only_mode == "true":
            long_only_choices = [True]
        elif long_only_mode == "false":
            long_only_choices = [False]
        else:  # "both"
            long_only_choices = [True, False]
            
        # 2. Build parameter dict (respecting FORCE_FLAGS from kwargs)
        params = {
            "trend_kind": kwargs.get("force_trend_kind") or trial.suggest_categorical("trend_kind", ["sma", "roc"]),
            "trend_lookback": kwargs.get("force_trend_lookback") or trial.suggest_categorical("trend_lookback", [120, 160, 200, 240, 300]),
            
            "flip_band_entry": trial.suggest_float("flip_band_entry", 0.01, 0.06),
            "flip_band_exit": trial.suggest_float("flip_band_exit", 0.005, 0.03),
            
            "vol_window": trial.suggest_categorical("vol_window", [45, 60, 90]),
            "vol_adapt_k": trial.suggest_categorical("vol_adapt_k", [0.0, 0.0025, 0.005, 0.0075]),
            
            "target_vol": trial.suggest_categorical("target_vol", [0.3, 0.4, 0.5, 0.6]),
            "min_mult": trial.suggest_float("min_mult", 0.3, 0.7, step=0.1),
            "max_mult": trial.suggest_float("max_mult", 1.2, 2.0, step=0.1),
            
            "cooldown_minutes": trial.suggest_categorical("cooldown_minutes", [60, 120, 180, 240]),
            "step_allocation": trial.suggest_categorical("step_allocation", [0.33, 0.5, 0.66, 1.0]),
            "max_position": trial.suggest_categorical("max_position", [0.6, 0.8, 1.0]),
            
            "position_sizing_mode": kwargs.get("force_position_sizing_mode") or trial.suggest_categorical("position_sizing_mode", ["static", "volatility"]),
            "position_sizing_target_vol": trial.suggest_float("position_sizing_target_vol", 0.3, 0.7),
            "position_sizing_min_step": trial.suggest_float("position_sizing_min_step", 0.1, 0.3),
            "position_sizing_max_step": 1.0,  # Fixed as per original script
            
            "gate_window_days": trial.suggest_categorical("gate_window_days", [30, 60, 90]),
            "gate_roc_threshold": trial.suggest_categorical("gate_roc_threshold", [0.0, 0.01, 0.02]),
            
            "funding_limit_long": trial.suggest_float("funding_limit_long", 0.01, 0.10),
            "funding_limit_short": trial.suggest_float("funding_limit_short", -0.10, -0.01),
            
            "rebalance_threshold_w": trial.suggest_categorical("rebalance_threshold_w", [0.0, 0.01]),
            "min_trade_btc": 0.0,
            
            "long_only": kwargs.get("force_long_only") if "force_long_only" in kwargs else trial.suggest_categorical("long_only", long_only_choices),
        }
        
        return params

    def create_strategy(self, params: Dict[str, Any]) -> EthBtcStrategy:
        """Instantiate EthBtcStrategy with StratParams."""
        return EthBtcStrategy(StratParams(**params))
