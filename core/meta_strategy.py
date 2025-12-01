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
        if isinstance(df, pd.Series): raise ValueError("Need OHLC")

        df_mr = self.mr.generate_positions(df["close"], funding)
        sig_mr = df_mr["target_w"]
        
        df_trend = self.trend.generate_positions(df["close"], funding)
        sig_trend = df_trend["target_w"]
        
        regime_score = get_regime_score(df)
        
        # FORCE ALIGNMENT
        # Use common index
        common_idx = df.index
        
        # Extract values as aligned numpy arrays
        v_mr = sig_mr.reindex(common_idx).fillna(0.0).values
        v_tr = sig_trend.reindex(common_idx).fillna(0.0).values
        v_sc = regime_score.reindex(common_idx).fillna(0.0).values
        
        mask_trend = v_sc > self.adx_threshold
        
        final = np.where(mask_trend, v_tr, v_mr)
        
        return pd.DataFrame({
            "target_w": final,
            "regime_score": v_sc,
            "sig_mr": v_mr,
            "sig_trend": v_tr
        }, index=common_idx)