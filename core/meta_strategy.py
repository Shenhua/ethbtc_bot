from __future__ import annotations
import pandas as pd
import numpy as np
from core.ethbtc_accum_bot import EthBtcStrategy, StratParams
from core.trend_strategy import TrendStrategy, TrendParams
from core.regime import get_regime_score

class MetaStrategy:
    def __init__(self, 
                 mr_params: StratParams, 
                 trend_params: TrendParams, 
                 adx_threshold: float = 25.0):
        """
        Ensemble Strategy.
        :param adx_threshold: The 'Regime Switch' level. 
                              ADX < threshold = Mean Reversion.
                              ADX > threshold = Trend.
        """
        self.mr = EthBtcStrategy(mr_params)
        self.trend = TrendStrategy(trend_params)
        self.adx_threshold = adx_threshold

    def generate_positions(self, df: pd.DataFrame, funding=None) -> pd.DataFrame:
        # Ensure we have OHLC for Regime detection
        if isinstance(df, pd.Series):
            raise ValueError("MetaStrategy requires a DataFrame with OHLC columns, not just Close.")

        # 1. Generate Sub-Signals
        # Mean Reversion (uses close)
        df_mr = self.mr.generate_positions(df["close"], funding)
        sig_mr = df_mr["target_w"]
        
        # Trend (uses close)
        df_trend = self.trend.generate_positions(df["close"], funding)
        sig_trend = df_trend["target_w"]
        
        # 2. Calculate Regime Score (Trend Strength)
        regime_score = get_regime_score(df)
        
        # 3. Decision Logic (Vectorized)
        # Mask where Trend is Strong
        mask_trend = regime_score > self.adx_threshold
        
        # Apply signals based on mask
        # Where Trend is Strong -> Use Trend Signal
        # Where Trend is Weak -> Use Mean Reversion Signal
        final_target = np.where(mask_trend, sig_trend, sig_mr)
        
        return pd.DataFrame({
            "target_w": final_target,
            "regime_score": regime_score, # Export for debugging/plotting
            "sig_mr": sig_mr,
            "sig_trend": sig_trend
        })