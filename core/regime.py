import pandas as pd
import numpy as np
import logging

log = logging.getLogger("regime")


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    # ... (Calculation logic remains standard, omitted for brevity but assumed present) ...
    # 1. Directional Movement
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm = np.where((plus_dm > 0) & (plus_dm > minus_dm), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > 0) & (minus_dm > plus_dm), minus_dm, 0.0)
    
    # 2. True Range & Smoothing
    tr1 = pd.DataFrame({
        'hl': high - low, 
        'hc': abs(high - close.shift(1)), 
        'lc': abs(low - close.shift(1))
    }).max(axis=1)
    
    atr = tr1.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, adjust=False).mean().fillna(0.0)
    log.debug(f"ADX calculation complete: mean={adx.mean():.2f}, last={adx.iloc[-1] if len(adx) > 0 else 0:.2f}")
    return adx

def get_regime_score(df_15m: pd.DataFrame, adx_period: int = 14) -> pd.Series:
    """
    Multi-Timeframe Regime Analysis.
    Combines 15m ADX and 1h ADX to form a consensus 'Trend Score'.
    """
    df = df_15m.copy()
    
    # FIX ITEM 5: Robust Timezone Handling
    if df.index.tz is None:
        # Assume UTC if naive (or risk misalignment, but better than stripping)
        df.index = df.index.tz_localize('UTC')
    else:
        # Standardize to UTC
        df.index = df.index.tz_convert('UTC')
    
    # Ensure monotonic index for resampling
    df = df[~df.index.duplicated(keep='last')]
    df = df.sort_index()
    
    # Verify monotonicity - if still not monotonic, something is wrong with the data
    if not df.index.is_monotonic_increasing:
        # Last resort: reset index and create a clean datetime index
        df = df.reset_index(drop=False).sort_values('close_time' if 'close_time' in df.columns else df.columns[0])
        df = df.set_index(df.columns[0] if 'close_time' not in df.columns else 'close_time')
        df = df[~df.index.duplicated(keep='last')]
    
    # Store the cleaned index for consistent reindexing
    clean_index = df.index.copy()
    
    # Final safety check
    if not clean_index.is_monotonic_increasing:
        raise ValueError("Unable to create monotonic index from input data")

    # 1. Calculate Base ADX (15m)
    log.debug(f"Calculating regime score for {len(df)} bars")
    adx_15 = calculate_adx(df['high'], df['low'], df['close'], period=adx_period)
    log.debug(f"Base ADX (15m): {adx_15.iloc[-1]:.2f}")
    
    # FIX ITEM 3 & 13: Resample using 'left' label/closed to represent COMPLETED intervals.
    # This avoids look-ahead bias.
    agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    
    # 30m Resample
    df_30m = df.resample('30min', label='left', closed='left').agg(agg_dict).dropna()
    log.debug(f"Resampled to 30m: {len(df_30m)} bars")
    if len(df_30m) > adx_period:
        adx_30m = calculate_adx(df_30m['high'], df_30m['low'], df_30m['close'], period=adx_period)
        # Manual forward fill to avoid monotonicity issues
        adx_30m_aligned = pd.Series(index=clean_index, dtype=float)
        adx_30m_sorted = adx_30m.sort_index()  # Ensure source is sorted
        for ts in clean_index:
            # Find the most recent 30m value at or before this timestamp
            mask = adx_30m_sorted.index <= ts
            if mask.any():
                adx_30m_aligned.loc[ts] = adx_30m_sorted[mask].iloc[-1]
            else:
                adx_30m_aligned.loc[ts] = 0.0
        adx_30m_aligned = adx_30m_aligned.fillna(0.0)
    else:
        adx_30m_aligned = pd.Series(0.0, index=clean_index)

    # 1H Resample
    df_1h = df.resample('1h', label='left', closed='left').agg(agg_dict).dropna()
    if len(df_1h) > adx_period:
        adx_1h = calculate_adx(df_1h['high'], df_1h['low'], df_1h['close'], period=adx_period)
        # Manual forward fill
        adx_1h_aligned = pd.Series(index=clean_index, dtype=float)
        adx_1h_sorted = adx_1h.sort_index()
        for ts in clean_index:
            mask = adx_1h_sorted.index <= ts
            if mask.any():
                adx_1h_aligned.loc[ts] = adx_1h_sorted[mask].iloc[-1]
            else:
                adx_1h_aligned.loc[ts] = 0.0
        adx_1h_aligned = adx_1h_aligned.fillna(0.0)
    else:
        adx_1h_aligned = pd.Series(0.0, index=clean_index)
    
    # 4. Consensus Logic
    trend_score = (0.2 * adx_15) + (0.3 * adx_30m_aligned) + (0.5 * adx_1h_aligned)
    
    return trend_score