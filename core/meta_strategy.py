from __future__ import annotations
import pandas as pd
import numpy as np
import logging
from typing import Optional
from core.ethbtc_accum_bot import EthBtcStrategy, StratParams
from core.trend_strategy import TrendStrategy, TrendParams
from core.regime import get_regime_score

log = logging.getLogger("meta_strategy")

class MetaStrategy:
    def __init__(self, 
                 mr_params: StratParams, 
                 trend_params: TrendParams, 
                 adx_threshold: float = 25.0,
                 precomputed_signals: Optional[pd.DataFrame] = None):
        """
        Ensemble Strategy.
        :param adx_threshold: The 'Regime Switch' level. 
                              ADX < threshold = Mean Reversion.
                              ADX > threshold = Trend.
        :param precomputed_signals: Optional DataFrame with pre-calced [v_mr, v_tr, regime_score] columns.
        """
        self.mr = EthBtcStrategy(mr_params)
        self.trend = TrendStrategy(trend_params)
        self.adx_threshold = adx_threshold
        self.pre = precomputed_signals

    def _empty_result(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return an empty result DataFrame with correct structure."""
        return pd.DataFrame({
            "target_w": np.zeros(len(df)),
            "regime_score": np.zeros(len(df)),
            "regime_state": np.full(len(df), -1),  # Default to MR
            "sig_mr": np.zeros(len(df)),
            "sig_trend": np.zeros(len(df))
        }, index=df.index)

    def generate_positions(self, df: pd.DataFrame, funding=None) -> pd.DataFrame:
        if isinstance(df, pd.Series): raise ValueError("Need OHLC")

        # 0. Check for Cached Signals (Optimization Mode)
        v_mr, v_tr, v_sc = None, None, None
        common_idx = df.index # Default
        
        if self.pre is not None:
            # Slicing from Master Cache is fast
            # We align via intersection in case backtest window is a subset
            common = df.index.intersection(self.pre.index)
            if len(common) > 0:
                sliced = self.pre.loc[common]
                v_mr = sliced["v_mr"].values
                v_tr = sliced["v_tr"].values
                v_sc = sliced["regime_score"].values
                common_idx = common
            else:
                log.warning("[META] Cached signals provided but index disjoint! Fallback to calc.")

        # 1. Generate Sub-Signals (if not cached)
        if v_mr is None:
            log.debug("[META] Generating Mean Reversion signal")
            df_mr = self.mr.generate_positions(df["close"], funding)
            sig_mr = df_mr["target_w"]
            log.debug(f"[META] MR signal: {sig_mr.iloc[-1]:.4f}")
            
            log.debug("[META] Generating Trend signal")
            df_trend = self.trend.generate_positions(df, funding)
            sig_trend = df_trend["target_w"]
            log.debug(f"[META] Trend signal: {sig_trend.iloc[-1]:.4f}")
            
            # 2. Calculate Regime Score
            regime_score = get_regime_score(df)
            
            # 3. FORCE ALIGNMENT
            common_idx = df.index.intersection(regime_score.index)
            if len(common_idx) == 0:
                return self._empty_result(df)

            v_mr = sig_mr.reindex(common_idx).fillna(0.0).values
            v_tr = sig_trend.reindex(common_idx).fillna(0.0).values
            v_sc = regime_score.reindex(common_idx).fillna(0.0).values
        else:
             # Just ensure we have index
             pass
        
        # Guard against empty alignment
        if len(common_idx) == 0: return self._empty_result(df)
        
        assert v_sc is not None
        assert v_mr is not None
        assert v_tr is not None
        
        v_sc_arr = np.asarray(v_sc)
        
        # --- 4. HYSTERESIS LOGIC (The Churn Killer) ---
        # Instead of a simple check, we use a latching mechanism.
        # We only switch UP if score > (thresh + buffer)
        # We only switch DOWN if score < (thresh - buffer)
        buffer = 2.0 
        upper_bound = self.adx_threshold + buffer
        lower_bound = max(0.0, self.adx_threshold - buffer)
        
        # 1 = Trend, -1 = MR, 0 = Hold previous
        # We use numpy to create a signal series
        regime_signal = np.zeros_like(v_sc_arr)
        regime_signal[v_sc_arr > upper_bound] = 1  # Enter Trend
        regime_signal[v_sc_arr < lower_bound] = -1 # Enter MR
        
        # Convert to pandas to use ffill() (Forward Fill propagates the state)
        # 0s become NaNs, then filled with previous state
        regime_series = pd.Series(regime_signal, index=common_idx)
        regime_series = regime_series.replace(0, np.nan).ffill().fillna(-1) # Default to MR start
        
        # Create final boolean mask
        mask_trend = (regime_series == 1).values
        mask_trend_arr = np.asarray(mask_trend)
        v_tr_arr = np.asarray(v_tr)
        v_mr_arr = np.asarray(v_mr)
        final = np.where(mask_trend_arr, v_tr_arr, v_mr_arr)
        log.debug(f"[META] Final signal: {final[-1]:.4f} (regime={'TREND' if mask_trend_arr[-1] else 'MR'}, score={v_sc_arr[-1]:.2f})")
        
        # FIX #7: Export regime state for observability
        return pd.DataFrame({
            "target_w": final,
            "regime_score": v_sc,
            "regime_state": regime_series.values,  # -1=MR, 1=Trend
            "sig_mr": v_mr,
            "sig_trend": v_tr
        }, index=common_idx)