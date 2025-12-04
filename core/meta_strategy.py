from __future__ import annotations
import pandas as pd
import numpy as np
import logging
from core.ethbtc_accum_bot import EthBtcStrategy, StratParams
from core.trend_strategy import TrendStrategy, TrendParams
from core.regime import get_regime_score

log = logging.getLogger("meta_strategy")

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

        # 1. Generate Sub-Signals
        log.debug(f"[META] Generating Mean Reversion signal")
        df_mr = self.mr.generate_positions(df["close"], funding)
        sig_mr = df_mr["target_w"]
        log.debug(f"[META] MR signal: {sig_mr.iloc[-1]:.4f}")
        
        log.debug(f"[META] Generating Trend signal")
        # FIX #6: Pass full DataFrame for consistency (Trend can handle both)
        df_trend = self.trend.generate_positions(df, funding)
        sig_trend = df_trend["target_w"]
        log.debug(f"[META] Trend signal: {sig_trend.iloc[-1]:.4f}")
        
        # 2. Calculate Regime Score
        # get_regime_score now handles the resampling 'left'/'closed' internally
        regime_score = get_regime_score(df)
        
        # 3. FORCE ALIGNMENT
        common_idx = df.index.intersection(regime_score.index)
        
        v_mr = sig_mr.reindex(common_idx).fillna(0.0).values
        v_tr = sig_trend.reindex(common_idx).fillna(0.0).values
        v_sc = regime_score.reindex(common_idx).fillna(0.0).values
        
        # --- 4. HYSTERESIS LOGIC (The Churn Killer) ---
        # Instead of a simple check, we use a latching mechanism.
        # We only switch UP if score > (thresh + buffer)
        # We only switch DOWN if score < (thresh - buffer)
        buffer = 2.0 
        upper_bound = self.adx_threshold + buffer
        lower_bound = max(0.0, self.adx_threshold - buffer)
        
        # 1 = Trend, -1 = MR, 0 = Hold previous
        # We use numpy to create a signal series
        regime_signal = np.zeros_like(v_sc)
        regime_signal[v_sc > upper_bound] = 1  # Enter Trend
        regime_signal[v_sc < lower_bound] = -1 # Enter MR
        
        # Convert to pandas to use ffill() (Forward Fill propagates the state)
        # 0s become NaNs, then filled with previous state
        regime_series = pd.Series(regime_signal, index=common_idx)
        regime_series = regime_series.replace(0, np.nan).ffill().fillna(-1) # Default to MR start
        
        # Create final boolean mask
        mask_trend = (regime_series == 1).values
        # ------------------------------------------------
        
        final = np.where(mask_trend, v_tr, v_mr)
        log.debug(f"[META] Final signal: {final[-1]:.4f} (regime={'TREND' if mask_trend[-1] else 'MR'}, score={v_sc[-1]:.2f})")
        
        # FIX #7: Export regime state for observability
        return pd.DataFrame({
            "target_w": final,
            "regime_score": v_sc,
            "regime_state": regime_series.values,  # -1=MR, 1=Trend
            "sig_mr": v_mr,
            "sig_trend": v_tr
        }, index=common_idx)