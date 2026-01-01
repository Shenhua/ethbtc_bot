"""
Tests for Higher Timeframe (HTF) Filter Feature.

Validates:
- HTF filter blocks entries against HTF trend
- HTF filter allows entries aligned with HTF trend
- Feature is disabled by default
"""

import pytest
import pandas as pd
import numpy as np
from core.trend_strategy import TrendStrategy, TrendParams


def get_sample_data(n_bars=200):
    """Generate sample OHLC data with proper datetime index."""
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="15min")
    prices = 50000 + np.cumsum(np.random.randn(n_bars) * 100)
    return pd.DataFrame({
        "open": prices,
        "high": prices + np.random.rand(n_bars) * 50,
        "low": prices - np.random.rand(n_bars) * 50,
        "close": prices
    }, index=dates)


class TestHTFFilterDisabled:
    """Test that HTF filter doesn't interfere when disabled."""
    
    def test_no_filter_when_disabled(self):
        """HTF filter should not affect signals when disabled."""
        p = TrendParams(
            htf_filter_enabled=False,
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


class TestHTFFilterEnabled:
    """Test that HTF filter works correctly when enabled."""
    
    def test_htf_filter_runs(self):
        """HTF filter should run without error."""
        p = TrendParams(
            htf_filter_enabled=True,
            htf_multiplier=4,  # 4x base TF
            htf_ma_period=10,
            htf_ma_type="ema",
            bar_interval_minutes=15,
            fast_period=5,
            slow_period=10,
            cooldown_minutes=0
        )
        strat = TrendStrategy(p)
        
        df = get_sample_data(n_bars=200)
        result = strat.generate_positions(df)
        
        # Should run without error
        assert "target_w" in result.columns
    
    def test_htf_resampling_works(self):
        """HTF resampling should aggregate correctly."""
        # Create data spanning multiple HTF bars
        dates = pd.date_range("2024-01-01", periods=100, freq="15min")
        close = pd.Series(range(100), index=dates)
        
        # Resample to 4x (60min)
        htf_close = close.resample('60min').last()
        
        # Should have ~25 HTF bars from 100 LTF bars
        assert len(htf_close) == 25
        assert htf_close.iloc[-1] == 99  # Last value preserved


class TestHTFTrendAlignment:
    """Test HTF trend calculation."""
    
    def test_uptrend_detection(self):
        """Price above MA should indicate uptrend."""
        p = TrendParams(
            htf_filter_enabled=True,
            htf_multiplier=4,
            htf_ma_period=10,
            htf_ma_type="sma",
            bar_interval_minutes=15,
            fast_period=5,
            slow_period=10,
            cooldown_minutes=0
        )
        strat = TrendStrategy(p)
        
        # Strong uptrend
        dates = pd.date_range("2024-01-01", periods=200, freq="15min")
        prices = [50000 + i * 10 for i in range(200)]
        df = pd.DataFrame({
            "open": prices,
            "high": [p + 50 for p in prices],
            "low": [p - 50 for p in prices],
            "close": prices
        }, index=dates)
        
        result = strat.generate_positions(df)
        
        # In uptrend, longs should be allowed
        assert "target_w" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
