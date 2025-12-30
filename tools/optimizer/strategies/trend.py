"""
Trend Strategy Optimizer

Adapter for the Trend strategy (TrendStrategy).
Wraps search space and strategy instantiation into the Optimizer Framework.

Author: AI Audit System
Date: 2024-12-30
"""

from typing import Any, Dict
import optuna

from ..base import BaseOptimizer
from core.trend_strategy import TrendParams, TrendStrategy


class TrendOptimizer(BaseOptimizer):
    """
    Optimizer adapter for the Trend strategy.
    
    Ported from tools/optimize_trend.py
    """
    
    def get_strategy_name(self) -> str:
        return "Trend"
    
    def suggest_params(self, trial: optuna.Trial, **kwargs) -> Dict[str, Any]:
        """
        Define the parameter search space for the Trend strategy.
        
        Args:
            trial: Optuna trial
            allow_shorts: (Optional) Whether to permit shorting
            
        Returns:
            Dict of parameters
        """
        allow_shorts = kwargs.get("allow_shorts", False)
        
        params = {
            "fast_period": trial.suggest_int("fast_period", 10, 200, step=10),
            "slow_period": trial.suggest_int("slow_period", 40, 400, step=20),
            "ma_type": trial.suggest_categorical("ma_type", ["ema", "sma"]),
            "cooldown_minutes": trial.suggest_categorical("cooldown_minutes", [60, 120, 240, 360]),
            
            "funding_limit_long": trial.suggest_float("funding_limit_long", 0.01, 0.10),
            "funding_limit_short": trial.suggest_float("funding_limit_short", -0.10, -0.01),
            
            "position_sizing_mode": trial.suggest_categorical("position_sizing_mode", ["static", "volatility"]),
            "position_sizing_target_vol": trial.suggest_float("position_sizing_target_vol", 0.3, 0.7),
            "position_sizing_min_step": trial.suggest_float("position_sizing_min_step", 0.1, 0.3),
            "position_sizing_max_step": 1.0,
            
            "step_allocation": 1.0,
            "max_position": 1.0,
            
            "long_only": trial.suggest_categorical("long_only", [True, False]) if allow_shorts else True,
        }
        
        return params

    def create_strategy(self, params: Dict[str, Any]) -> TrendStrategy:
        """Instantiate TrendStrategy with TrendParams."""
        return TrendStrategy(TrendParams(**params))
