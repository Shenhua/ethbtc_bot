import pandas as pd
import numpy as np

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculates ADX (0-100) manually using pandas.
    ADX measures Trend Strength regardless of direction.
    > 25 usually indicates a strong trend.
    < 20 indicates a ranging/choppy market.
    """
    # 1. Directional Movement
    plus_dm = high.diff()
    minus_dm = low.diff()
    
    # +DM
    plus_dm = np.where((plus_dm > 0) & (plus_dm > minus_dm), plus_dm, 0.0)
    # -DM
    minus_dm = np.where((minus_dm > 0) & (minus_dm > plus_dm), minus_dm, 0.0)
    
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    # 2. True Range
    tr1 = pd.DataFrame({
        'hl': high - low, 
        'hc': abs(high - close.shift(1)), 
        'lc': abs(low - close.shift(1))
    }).max(axis=1)
    
    # 3. Smoothing (Wilder's Smoothing)
    atr = tr1.ewm(alpha=1/period, adjust=False).mean()
    plus_di_smooth = plus_dm.ewm(alpha=1/period, adjust=False).mean()
    minus_di_smooth = minus_dm.ewm(alpha=1/period, adjust=False).mean()
    
    # 4. Directional Indices
    atr = atr.replace(0, np.nan)
    plus_di = 100 * (plus_di_smooth / atr)
    minus_di = 100 * (minus_di_smooth / atr)
    
    # 5. ADX
    sum_di = plus_di + minus_di
    sum_di = sum_di.replace(0, np.nan)
    
    dx = 100 * abs(plus_di - minus_di) / sum_di
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx.fillna(0.0)

def get_regime_score(df_15m: pd.DataFrame, adx_period: int = 14) -> pd.Series:
    """
    Multi-Timeframe Regime Analysis.
    Combines 15m ADX and 1h ADX to form a consensus 'Trend Score'.
    """
    # Ensure index is standard for resampling
    df = df_15m.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None) # Remove TZ for safe resampling

    # 1. Calculate Base ADX (15m)
    adx_15 = calculate_adx(df['high'], df['low'], df['close'], period=adx_period)
    
    # 2. Resample to 30m
    df_30m = df.resample('30min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna()
    
    if len(df_30m) > adx_period:
        adx_30m = calculate_adx(df_30m['high'], df_30m['low'], df_30m['close'], period=adx_period)
        # Shift, Reindex back to 15m, and Forward Fill
        adx_30m_aligned = adx_30m.shift(1).reindex(df.index, method='ffill').fillna(0.0)
    else:
        adx_30m_aligned = pd.Series(0.0, index=df.index)

    # 3. Resample to 1H
    df_1h = df.resample('1h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna()
    
    if len(df_1h) > adx_period:
        adx_1h = calculate_adx(df_1h['high'], df_1h['low'], df_1h['close'], period=adx_period)
        # Shift, Reindex back to 15m, and Forward Fill
        adx_1h_aligned = adx_1h.shift(1).reindex(df.index, method='ffill').fillna(0.0)
    else:
        adx_1h_aligned = pd.Series(0.0, index=df.index)
    
    # 4. Consensus Logic (Weighted) on the aligned dataframes
    trend_score_raw = (0.2 * adx_15) + (0.3 * adx_30m_aligned) + (0.5 * adx_1h_aligned)
    
    # 5. FINAL ALIGNMENT: Force the index to match the original input exactly.
    # This restores the UTC timezone if the input had it.
    trend_score = trend_score_raw.copy()
    trend_score.index = df_15m.index 
    
    return trend_score