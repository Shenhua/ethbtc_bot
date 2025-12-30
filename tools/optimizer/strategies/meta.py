"""
Meta Strategy Optimizer

Adapter for the Meta strategy (MetaStrategy).
Combines MR and Trend strategies with an ADX threshold.

Author: AI Audit System
Date: 2024-12-30
"""

from typing import Any, Dict, Optional
import optuna

from ..base import BaseOptimizer
from core.ethbtc_accum_bot import StratParams
from core.trend_strategy import TrendParams
from core.meta_strategy import MetaStrategy


class MetaOptimizer(BaseOptimizer):
    """
    Optimizer adapter for the Meta strategy.
    
    Ported from tools/optimize_meta.py and enhanced with Optuna support.
    """
    
    def __init__(
        self,
        fee_params: Any,
        mr_params: StratParams,
        trend_params: TrendParams,
        full_df: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(fee_params, **kwargs)
        self.mr_params = mr_params
        self.trend_params = trend_params
        self.cached_signals = None
        
        # Pre-Calculate Signals for Speed (Phase 9 Optimization)
        if full_df is not None:
            self.log.info("Pre-calculating Meta signals for optimization...")
            # Create temp strategy to generate base signals
            temp = MetaStrategy(mr_params, trend_params)
            # Use 'data' if it's passed (cli passes 'data' but mapped to full_df here)
            # Depending on how we modify cli.py
            try:
                # Generate full signals once
                res = temp.generate_positions(full_df)
                
                # Cache relevant columns (renaming to match what MetaStrategy expects)
                # MetaStrategy expects: v_mr, v_tr, regime_score
                # generated return has: target_w, regime_score, regime_state, sig_mr, sig_trend
                self.cached_signals = res[["sig_mr", "sig_trend", "regime_score"]].rename(
                    columns={"sig_mr": "v_mr", "sig_trend": "v_tr"}
                )
                self.log.info(f"Cached signals for {len(res)} bars.")
            except Exception as e:
                self.log.warning(f"Failed to cache Meta signals: {e}")
    
    def get_strategy_name(self) -> str:
        return "Meta"
    
    def suggest_params(self, trial: optuna.Trial, **kwargs) -> Dict[str, Any]:
        """
        Define the parameter search space for the Meta strategy.
        Focuses on the adx_threshold.
        """
        return {
            "adx_threshold": trial.suggest_float("adx_threshold", 10.0, 40.0, step=5.0)
        }

    def create_strategy(self, params: Dict[str, Any]) -> MetaStrategy:
        """Instantiate MetaStrategy with existing MR/Trend params and new threshold."""
        return MetaStrategy(
            mr_params=self.mr_params,
            trend_params=self.trend_params,
            adx_threshold=float(params["adx_threshold"]),
            precomputed_signals=self.cached_signals
        )
