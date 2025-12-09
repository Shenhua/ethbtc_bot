"""
Feature Engineering for ML Regime Detection.

This module provides the `build_regime_features()` function that calculates
all features used by the ML regime classifier. It is the SINGLE SOURCE OF TRUTH
for feature calculation, used by both backtesting and live trading to ensure parity.

Features:
- ADX (15m) - Trend strength from existing regime module
- RSI (14) - Overbought/oversold indicator
- Volume Ratio - Current volume vs 20-bar SMA
- Bollinger Band Width - Volatility compression
- Price ROC (4h) - 16-bar rate of change
- Returns Volatility - 20-bar rolling std of returns
- Funding Rate - From external data
- Funding Z-Score - 30-day rolling standardization
- Fear & Greed Index - Daily sentiment (forward-filled)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import logging
from typing import Optional

log = logging.getLogger("regime_features")


def build_regime_features(
    ohlcv: pd.DataFrame,
    funding: Optional[pd.Series] = None,
    fear_greed: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Build feature matrix for ML regime classification.
    
    All features are calculated point-in-time (no look-ahead bias).
    Missing external data is handled gracefully with sensible defaults.
    
    Args:
        ohlcv: DataFrame with columns 'open', 'high', 'low', 'close', 'volume'.
               Must have a DatetimeIndex.
        funding: Optional Series of funding rates. Will be forward-filled to
                 match ohlcv index. If None, uses 0.0.
        fear_greed: Optional Series of Fear & Greed Index (0-100). Daily values
                    will be forward-filled. If None, uses 50 (neutral).
    
    Returns:
        DataFrame with feature columns, aligned to ohlcv index.
        All NaN values are filled with sensible defaults.
    """
    df = ohlcv.copy()
    features = pd.DataFrame(index=df.index)
    
    log.debug(f"Building features for {len(df)} bars")
    
    # =========================================================================
    # 1. ADX (15m) - Using existing calculation
    # =========================================================================
    from core.regime import calculate_adx
    features["adx_15m"] = calculate_adx(df["high"], df["low"], df["close"], period=14)
    
    # =========================================================================
    # 2. RSI (14-period)
    # =========================================================================
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    features["rsi_14"] = 100 - (100 / (1 + rs))
    
    # =========================================================================
    # 3. Volume Ratio (current volume / 20-bar SMA)
    # =========================================================================
    vol_sma = df["volume"].rolling(20).mean()
    features["volume_ratio"] = df["volume"] / vol_sma.replace(0, np.nan)
    
    # =========================================================================
    # 4. Bollinger Band Width (2 std / SMA20)
    # =========================================================================
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    features["bb_width"] = (2 * std20) / sma20.replace(0, np.nan)
    
    # =========================================================================
    # 5. Price Rate of Change (4h = 16 bars at 15m)
    # =========================================================================
    features["roc_4h"] = df["close"].pct_change(16) * 100
    
    # =========================================================================
    # 6. Returns Volatility (20-bar rolling std of returns)
    # =========================================================================
    returns = df["close"].pct_change()
    features["returns_std"] = returns.rolling(20).std() * 100
    
    # =========================================================================
    # 7. Funding Rate (external data)
    # =========================================================================
    if funding is not None:
        # Align funding to ohlcv index via forward-fill
        funding_aligned = funding.reindex(df.index, method="ffill")
        features["funding_rate"] = funding_aligned.fillna(0.0)
        
        # Z-score (30-day rolling = 480 bars at 15m, but use 8h for funding = 32*30)
        # Funding is typically every 8 hours, so we use a smaller window
        rolling_mean = features["funding_rate"].rolling(120, min_periods=30).mean()
        rolling_std = features["funding_rate"].rolling(120, min_periods=30).std()
        features["funding_zscore"] = (
            (features["funding_rate"] - rolling_mean) / rolling_std.replace(0, 1)
        )
    else:
        features["funding_rate"] = 0.0
        features["funding_zscore"] = 0.0
        log.debug("No funding data provided, using defaults")
    
    # =========================================================================
    # 8. Fear & Greed Index (daily, forward-filled)
    # =========================================================================
    if fear_greed is not None:
        # Fear & Greed is daily, forward-fill to 15m bars
        fg_aligned = fear_greed.reindex(df.index, method="ffill")
        features["fear_greed"] = fg_aligned.fillna(50.0)  # Neutral default
    else:
        features["fear_greed"] = 50.0  # Neutral default
        log.debug("No Fear & Greed data provided, using neutral default")
    
    # =========================================================================
    # Fill any remaining NaN with sensible defaults
    # =========================================================================
    defaults = {
        "adx_15m": 25.0,        # Neutral ADX
        "rsi_14": 50.0,         # Neutral RSI
        "volume_ratio": 1.0,    # Average volume
        "bb_width": 0.02,       # Typical width
        "roc_4h": 0.0,          # No change
        "returns_std": 1.0,     # Typical volatility
        "funding_rate": 0.0,    # Neutral funding
        "funding_zscore": 0.0,  # Neutral z-score
        "fear_greed": 50.0,     # Neutral sentiment
    }
    
    for col, default in defaults.items():
        if col in features.columns:
            features[col] = features[col].fillna(default)
    
    log.debug(f"Feature columns: {list(features.columns)}")
    log.debug(f"Features shape: {features.shape}")
    
    return features


def get_feature_names() -> list[str]:
    """Return list of feature names in order."""
    return [
        "adx_15m",
        "rsi_14",
        "volume_ratio",
        "bb_width",
        "roc_4h",
        "returns_std",
        "funding_rate",
        "funding_zscore",
        "fear_greed",
    ]
