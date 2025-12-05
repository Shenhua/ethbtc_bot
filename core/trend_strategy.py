from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class TrendParams:
    # Core Trend Params
    fast_period: int = 50    # e.g. 50-bar EMA
    slow_period: int = 200   # e.g. 200-bar EMA
    ma_type: str = "ema"     # 'ema' or 'sma'
    
    # Risk / Execution
    cooldown_minutes: int = 60
    step_allocation: float = 1.0  # Trend followers usually go "All In" on breakout
    max_position: float = 1.0
    long_only: bool = True       
    
    # Funding Rate Filters
    funding_limit_long: float = 0.05
    funding_limit_short: float = -0.05
    
    # Dynamic Position Sizing
    position_sizing_mode: str = "static"
    position_sizing_target_vol: float = 0.5
    position_sizing_min_step: float = 0.1
    position_sizing_max_step: float = 1.0

    # Legacy compatibility (ignored but prevents config crashes)
    trend_kind: str = "trend"
    trend_lookback: int = 0
    flip_band_entry: float = 0.0
    flip_band_exit: float = 0.0
    vol_window: int = 0
    vol_adapt_k: float = 0.0
    bar_interval_minutes: int = 15
    target_vol: float = 0.0
    min_mult: float = 1.0
    max_mult: float = 1.0
    rebalance_threshold_w: float = 0.0
    min_trade_btc: float = 0.0
    gate_window_days: int = 0
    gate_roc_threshold: float = 0.0

class TrendStrategy:
    def __init__(self, p: TrendParams): 
        self.p = p

    def generate_positions(self, df: pd.DataFrame | pd.Series, funding: pd.Series = None) -> pd.DataFrame:          # Support both Series (just close) and DataFrame (OHLC)
        if isinstance(df, pd.Series):
            close = df
        else:
            close = df["close"]

        # 1. Calculate Moving Averages
        if self.p.ma_type == "sma":
            fast = close.rolling(self.p.fast_period).mean()
            slow = close.rolling(self.p.slow_period).mean()
        else:
            # EMA is generally more responsive/standard for crypto
            fast = close.ewm(span=self.p.fast_period, adjust=False).mean()
            slow = close.ewm(span=self.p.slow_period, adjust=False).mean()

        # 2. Generate Signal (Crossover)
        # Signal = 1 if Fast > Slow (Golden Cross)
        # Signal = -1 if Fast < Slow (Death Cross)
        raw_sig = np.where(fast > slow, 1.0, -1.0)
        sig = pd.Series(raw_sig, index=close.index)

        # 3. Apply Cooldown & Hysteresis
        # Prevents "whipsaw" if lines are tangled
        clean_sig = pd.Series(0.0, index=close.index)
        state = 0.0
        last_flip_ts = close.index[0]
        min_delta = pd.Timedelta(minutes=self.p.cooldown_minutes)
        
        for t in close.index:
            s = sig.loc[t]
            
            # Only flip if cooldown passed
            if s != state:
                if (t - last_flip_ts) >= min_delta:
                    state = s
                    last_flip_ts = t
                else:
                    # Keep previous state
                    pass
            
            clean_sig.loc[t] = state

        # 4. Funding Filter (Safety)
        if funding is not None:
            # Align
            funding = funding.reindex(close.index).ffill().fillna(0.0)
            
            # Mask: True if funding prohibits this side
            block_long = (funding > self.p.funding_limit_long)
            block_short = (funding < self.p.funding_limit_short)
            
            # Logic: 
            # If Signal is Long (1) AND Block Long is True -> Force Neutral (0) IF we weren't already Long?
            # Actually, standard safety is: If unsafe, go to cash (0). 
            # But "Don't Enter" implies holding.
            
            # Vectorized Hold Logic:
            # If (Signal=1 AND Block=True), effective signal = 0.0 (Safety First approach)
            # OR effective signal = Previous Signal (Hold approach).
            # Given "Safety" context, we usually want to exit crowded trades. 
            # However, to fix "premature exit", we can apply a "Neutral" zone only if the trend hasn't reversed.
            
            # Re-implementation: Strict Safety (Exit on excessive funding) is usually desired in crypto.
            # If the user intention was "Filter Entries", we use pandas to forward fill 0s.
            
            # Correct logic for "Filter Entry":
            # If Signal flips 0->1, but funding is high, stay 0.
            # If Signal is already 1, and funding gets high, stay 1 (Hold).
            
            final_sig = clean_sig.copy()
            
            # We iterate to apply stateful "Filter Entry Only" logic
            # (Vectorizing this specific state machine is complex, falling back to efficient loop)
            curr = 0.0
            for i in range(len(final_sig)):
                raw = clean_sig.iat[i]
                f = funding.iat[i]
                
                # Default to raw signal
                proposed = raw
                
                # Check Long Entry Block
                if raw > 0 and curr <= 0: # Trying to enter Long
                    if f > self.p.funding_limit_long:
                        proposed = 0.0 # Block entry
                
                # Check Short Entry Block
                if raw < 0 and curr >= 0: # Trying to enter Short
                    if f < self.p.funding_limit_short:
                        proposed = 0.0 # Block entry
                        
                curr = proposed
                final_sig.iat[i] = curr
                
            clean_sig = final_sig

        # 5. Allocation
        lo = 0.0 if self.p.long_only else -self.p.max_position
        target_w = clean_sig.clip(lo, self.p.max_position)
        
        return pd.DataFrame({"target_w": target_w})