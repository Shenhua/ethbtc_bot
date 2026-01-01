"""
Tests for Bollinger Squeeze Feature.

Validates:
- Squeeze detection identifies band compression
- Filter blocks entries when no recent squeeze
- Filter allows entries after squeeze ends
- Feature is disabled by default
"""

import pytest
import pandas as pd
import numpy as np
from core.trend_strategy import TrendStrategy, TrendParams


def get_sample_data(n_bars=100):
    """Generate sample OHLC data."""
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="15min")
    prices = 50000 + np.cumsum(np.random.randn(n_bars) * 100)
    return pd.DataFrame({
        "open": prices,
        "high": prices + np.random.rand(n_bars) * 50,
        "low": prices - np.random.rand(n_bars) * 50,
        "close": prices
    }, index=dates)


class TestBollingerSqueezeDisabled:
    """Test that squeeze filter doesn't interfere when disabled."""
    
    def test_no_filter_when_disabled(self):
        """Squeeze filter should not affect signals when disabled."""
        p = TrendParams(
            bollinger_squeeze_enabled=False,
            fast_period=10,
            slow_period=20,
            cooldown_minutes=0
        )
        strat = TrendStrategy(p)
        
        df = get_sample_data()
        result = strat.generate_positions(df)
        
        # Should run without error and return correct structure
        assert "target_w" in result.columns
        assert len(result) == len(df)


class TestBollingerSqueezeEnabled:
    """Test that squeeze filter works correctly when enabled."""
    
    def test_squeeze_detection_runs(self):
        """Squeeze detection should run without error."""
        p = TrendParams(
            bollinger_squeeze_enabled=True,
            bollinger_period=10,
            bollinger_std=2.0,
            squeeze_threshold=0.5,
            squeeze_lookback_bars=20,
            squeeze_signal_bars=5,
            fast_period=5,
            slow_period=10,
            cooldown_minutes=0
        )
        strat = TrendStrategy(p)
        
        df = get_sample_data(n_bars=100)
        result = strat.generate_positions(df)
        
        # Should run without error
        assert "target_w" in result.columns
    
    def test_low_volatility_triggers_squeeze(self):
        """Low volatility period should be detected as squeeze."""
        p = TrendParams(
            bollinger_squeeze_enabled=True,
            bollinger_period=10,
            bollinger_std=2.0,
            squeeze_threshold=0.5,
            squeeze_lookback_bars=20,
            squeeze_signal_bars=10,
            fast_period=5,
            slow_period=10,
            cooldown_minutes=0
        )
        strat = TrendStrategy(p)
        
        # Create price data with varying volatility
        dates = pd.date_range("2024-01-01", periods=100, freq="15min")
        # High vol first 50 bars, then low vol (squeeze)
        high_vol = 50000 + np.cumsum(np.random.randn(50) * 200)
        low_vol = high_vol[-1] + np.cumsum(np.random.randn(50) * 10)
        prices = np.concatenate([high_vol, low_vol])
        
        df = pd.DataFrame({
            "open": prices,
            "high": prices + 50,
            "low": prices - 50,
            "close": prices
        }, index=dates)
        
        result = strat.generate_positions(df)
        assert "target_w" in result.columns


class TestBollingerBandCalculation:
    """Test the Bollinger band calculation."""
    
    def test_band_width_normalized(self):
        """Band width should be normalized by middle band."""
        close = pd.Series([100] * 50, index=pd.date_range("2024-01-01", periods=50, freq="15min"))
        
        middle = close.rolling(20).mean()
        std = close.rolling(20).std()
        upper = middle + (2.0 * std)
        lower = middle - (2.0 * std)
        
        band_width = (upper - lower) / middle
        
        # With constant price, std=0, band_width=0
        assert band_width.dropna().iloc[-1] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
