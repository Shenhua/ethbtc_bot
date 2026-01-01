"""
Tests for Volume Confirmation Feature.

Validates:
- Volume filter blocks entries when volume is too low
- Volume filter allows entries when volume is high enough
- Existing positions are held regardless of volume
- Filter is disabled by default
"""

import pytest
import pandas as pd
import numpy as np
from core.trend_strategy import TrendStrategy, TrendParams


def get_sample_data_with_volume(n_bars=100):
    """Generate sample OHLC data with volume."""
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="15min")
    prices = 50000 + np.cumsum(np.random.randn(n_bars) * 100)
    # Volume with some spikes
    base_volume = 1000
    volume = np.random.uniform(base_volume * 0.5, base_volume * 1.5, n_bars)
    return pd.DataFrame({
        "open": prices,
        "high": prices + np.random.rand(n_bars) * 50,
        "low": prices - np.random.rand(n_bars) * 50,
        "close": prices,
        "volume": volume
    }, index=dates)


class TestVolumeConfirmDisabled:
    """Test that volume filter doesn't interfere when disabled."""
    
    def test_no_filter_when_disabled(self):
        """Volume filter should not affect signals when disabled."""
        p = TrendParams(
            volume_confirm_enabled=False,
            fast_period=10,
            slow_period=20
        )
        strat = TrendStrategy(p)
        
        df = get_sample_data_with_volume()
        # Set very low volume
        df["volume"] = 1  # Minimal volume
        
        result = strat.generate_positions(df)
        
        # Should still have signals despite low volume
        assert result["target_w"].abs().sum() > 0


class TestVolumeConfirmEnabled:
    """Test that volume filter works correctly when enabled."""
    
    def test_entry_blocked_on_low_volume(self):
        """Entry should be blocked if volume < threshold."""
        p = TrendParams(
            volume_confirm_enabled=True,
            volume_threshold_mult=2.0,  # Require 2x average
            volume_lookback_bars=10,
            fast_period=5,
            slow_period=10,
            cooldown_minutes=0
        )
        strat = TrendStrategy(p)
        
        df = get_sample_data_with_volume(n_bars=50)
        # Set constant low volume (below any threshold)
        df["volume"] = 100
        
        result = strat.generate_positions(df)
        
        # Most entries should be blocked due to low volume
        # But we allow holding, so not all zeros
        # Check that fewer position changes occur
        pos_changes = (result["target_w"] != result["target_w"].shift()).sum()
        assert pos_changes < 10, "Should have fewer signals with volume filter"
    
    def test_entry_allowed_on_high_volume(self):
        """Entry should be allowed if volume > threshold."""
        p = TrendParams(
            volume_confirm_enabled=True,
            volume_threshold_mult=1.5,
            volume_lookback_bars=10,
            fast_period=5,
            slow_period=10,
            cooldown_minutes=0
        )
        strat = TrendStrategy(p)
        
        df = get_sample_data_with_volume(n_bars=50)
        # Set very high volume on all bars
        df["volume"] = 10000  # Constant high
        
        result = strat.generate_positions(df)
        
        # Should have normal signals with high volume
        assert result["target_w"].abs().sum() > 0
    
    def test_holding_allowed_on_low_volume(self):
        """Existing positions should be held even if volume drops."""
        p = TrendParams(
            volume_confirm_enabled=True,
            volume_threshold_mult=2.0,
            volume_lookback_bars=5,
            fast_period=5,
            slow_period=10,
            cooldown_minutes=0
        )
        strat = TrendStrategy(p)
        
        df = get_sample_data_with_volume(n_bars=50)
        # Spike volume at start, then drop
        df["volume"] = 100  # Low
        df.iloc[:10, df.columns.get_loc("volume")] = 5000  # High for first 10 bars
        
        result = strat.generate_positions(df)
        
        # Should have some non-zero positions (entries allowed early, held later)
        non_zero = (result["target_w"] != 0).sum()
        assert non_zero > 0


class TestVolumeConfirmNoVolumeData:
    """Test graceful handling when volume data is missing."""
    
    def test_no_volume_column_no_crash(self):
        """Strategy should work if volume column is missing."""
        p = TrendParams(
            volume_confirm_enabled=True,  # Enabled but no data
            fast_period=10,
            slow_period=20
        )
        strat = TrendStrategy(p)
        
        # Create df WITHOUT volume column
        dates = pd.date_range("2024-01-01", periods=50, freq="15min")
        prices = 50000 + np.cumsum(np.random.randn(50) * 100)
        df = pd.DataFrame({
            "open": prices,
            "high": prices + 50,
            "low": prices - 50,
            "close": prices
        }, index=dates)
        
        # Should not crash
        result = strat.generate_positions(df)
        assert "target_w" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
