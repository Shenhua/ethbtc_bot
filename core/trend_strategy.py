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
    
    # Funding Counter-Trend (opens opposite position on extreme funding)
    funding_counter_enabled: bool = False
    extreme_funding_long_threshold: float = 0.0005  # >0.05% funding → SHORT signal
    extreme_funding_short_threshold: float = -0.0005  # <-0.05% funding → LONG signal
    funding_counter_position_size: float = 0.5  # Position size for counter trades
    funding_counter_cooldown_minutes: int = 480  # 8 hours (1 funding period)
    
    # Volume Confirmation (only enter if volume confirms the move)
    volume_confirm_enabled: bool = False
    volume_threshold_mult: float = 1.5  # Volume > 1.5x average to confirm
    volume_lookback_bars: int = 20  # 20-bar rolling average
    
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

        # 3b. Volume Confirmation Filter (NEW)
        # Only allow ENTRIES when volume > threshold * average
        # Existing positions are held regardless of volume
        if self.p.volume_confirm_enabled and isinstance(df, pd.DataFrame) and "volume" in df.columns:
            volume = df["volume"]
            avg_vol = volume.rolling(self.p.volume_lookback_bars).mean()
            vol_confirmed = volume > (avg_vol * self.p.volume_threshold_mult)
            
            # Apply "filter entry only" logic:
            # - If signal flips (new entry), require volume confirmation
            # - If already in position, allow holding
            sig_change = clean_sig != clean_sig.shift()
            
            # Block new entries without volume confirmation, but allow holds
            for t in close.index:
                if sig_change.loc[t] and not vol_confirmed.loc[t]:
                    # New entry but volume too low - revert to previous state
                    prev_idx = close.index.get_loc(t) - 1
                    if prev_idx >= 0:
                        clean_sig.loc[t] = clean_sig.iloc[prev_idx]
                    else:
                        clean_sig.loc[t] = 0.0

        # 4. Funding Counter-Trend Signal (NEW)
        # Opens SHORT when funding extremely positive (overleveraged longs)
        # Opens LONG when funding extremely negative
        funding_counter_signal = pd.Series(0.0, index=close.index)
        
        if self.p.funding_counter_enabled and funding is not None:
            # Align funding
            funding_aligned = funding.reindex(close.index).ffill().fillna(0.0)
            
            # Extreme funding thresholds
            extreme_long = funding_aligned > self.p.extreme_funding_long_threshold
            extreme_short = funding_aligned < self.p.extreme_funding_short_threshold
            
            # Counter-trend signals (opposite of crowded side)
            raw_counter = np.where(extreme_long, -self.p.funding_counter_position_size,
                          np.where(extreme_short, self.p.funding_counter_position_size, 0.0))
            
            # Apply cooldown to counter signals
            counter_sig = pd.Series(raw_counter, index=close.index)
            counter_clean = pd.Series(0.0, index=close.index)
            counter_state = 0.0
            last_counter_ts = close.index[0]
            counter_delta = pd.Timedelta(minutes=self.p.funding_counter_cooldown_minutes)
            
            for t in close.index:
                cs = counter_sig.loc[t]
                if cs != 0.0 and cs != counter_state:
                    if (t - last_counter_ts) >= counter_delta:
                        counter_state = cs
                        last_counter_ts = t
                elif cs == 0.0:
                    counter_state = 0.0
                counter_clean.loc[t] = counter_state
            
            funding_counter_signal = counter_clean
            
            # NOTE: Counter signal applied AFTER long_only clipping in step 7 below

        # 5. Funding Filter (Safety)
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
            
            # Vectorized "Filter Entry Only" Logic
            # We want to block Entry if funding is bad, but allow Holding if already in.
            # This is equivalent to: In a block of Signal=1, we become 1 only AFTER the first safe funding bar.
            
            # 1. Identify Blocks
            # A new block starts whenever the raw signal changes
            diff = clean_sig != clean_sig.shift()
            block_id = diff.cumsum()
            
            # 2. Define "Safe to Enter/Hold" Conditions
            # For 0 signal, always safe (trivial)
            # For 1 signal, safe if funding <= limit OR we were safe previously in this block (handled by cummax)
            # For -1 signal, safe if funding >= limit ...
            
            # Create boolean masks for valid funding
            valid_long = (funding <= self.p.funding_limit_long)
            valid_short = (funding >= self.p.funding_limit_short)
            
            # 3. Apply Logic per Block Type
            # Positive Blocks
            is_pos = clean_sig > 0
            # Within a positive block, we are "Active" (1.0) if we have encountered ANY valid funding bar so far
            # We use groupby().cummax() to propagate the "Entry Permit"
            pos_permit = valid_long.groupby(block_id).cummax()
            
            # Negative Blocks
            is_neg = clean_sig < 0
            neg_permit = valid_short.groupby(block_id).cummax()
            
            # 4. Synthesize Final Signal
            final_sig = clean_sig.copy()
            
            # Apply masks: Signal exists AND Permit exists
            # If Signal is 1 but Permit is False, result is 0
            # If Signal is -1 but Permit is False, result is 0
            final_sig[is_pos & ~pos_permit] = 0.0
            final_sig[is_neg & ~neg_permit] = 0.0
            
            clean_sig = final_sig

        # 6. Allocation (apply long_only clipping BEFORE counter-trend)
        lo = 0.0 if self.p.long_only else -self.p.max_position
        target_w = clean_sig.clip(lo, self.p.max_position)
        
        # 7. Apply Funding Counter-Trend AFTER long_only clipping
        # This allows counter-shorts even when long_only=True
        if self.p.funding_counter_enabled and funding is not None:
            target_w = np.where(funding_counter_signal != 0, funding_counter_signal, target_w)
            target_w = pd.Series(target_w, index=close.index)
        
        return pd.DataFrame({"target_w": target_w})