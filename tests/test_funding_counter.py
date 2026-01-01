"""
Tests for Funding Counter-Trend Strategy Feature.

Validates:
- Counter-trend signals fire on extreme funding
- Counter-trend respects cooldown
- Counter-trend is disabled by default
- Override of trend signal works correctly
"""

import pytest
import pandas as pd
import numpy as np
from core.trend_strategy import TrendStrategy, TrendParams


def get_sample_data(n_bars=100):
    """Generate sample OHLC data with datetime index."""
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="15min")
    prices = 50000 + np.cumsum(np.random.randn(n_bars) * 100)
    return pd.DataFrame({
        "open": prices,
        "high": prices + np.random.rand(n_bars) * 50,
        "low": prices - np.random.rand(n_bars) * 50,
        "close": prices
    }, index=dates)


def get_sample_funding(index, base_rate=0.0001):
    """Generate sample funding rate series."""
    return pd.Series(
        np.random.uniform(base_rate - 0.0001, base_rate + 0.0001, len(index)),
        index=index
    )


class TestFundingCounterDisabled:
    """Test that funding counter doesn't fire when disabled."""
    
    def test_no_counter_when_disabled(self):
        """Counter-trend should not fire when funding_counter_enabled=False."""
        p = TrendParams(
            funding_counter_enabled=False,
            extreme_funding_long_threshold=0.0005,
            long_only=False
        )
        strat = TrendStrategy(p)
        
        df = get_sample_data()
        # Extreme positive funding on all bars
        funding = pd.Series(0.001, index=df.index)  # Very high
        
        result = strat.generate_positions(df, funding=funding)
        
        # Should only have trend signals (-1 or +1), no counter shorts
        # Counter would produce -0.5 (position_size)
        unique_vals = result["target_w"].unique()
        # With long_only=False, we expect -1, 0, or 1, not fractional values
        assert all(v in [-1.0, 0.0, 1.0] for v in unique_vals)


class TestFundingCounterEnabled:
    """Test that funding counter fires correctly when enabled."""
    
    def test_counter_short_on_extreme_positive_funding(self):
        """Should open short when funding > extreme_funding_long_threshold."""
        p = TrendParams(
            funding_counter_enabled=True,
            extreme_funding_long_threshold=0.0005,
            funding_counter_position_size=0.5,
            funding_counter_cooldown_minutes=0,  # No cooldown for test
            long_only=False
        )
        strat = TrendStrategy(p)
        
        df = get_sample_data()
        # Extreme positive funding on all bars
        funding = pd.Series(0.001, index=df.index)  # > 0.0005 threshold
        
        result = strat.generate_positions(df, funding=funding)
        
        # Should have counter short signals (-0.5) where funding is extreme
        assert (result["target_w"] == -0.5).any(), "Expected counter-short signal"
    
    def test_counter_long_on_extreme_negative_funding(self):
        """Should open long when funding < extreme_funding_short_threshold."""
        p = TrendParams(
            funding_counter_enabled=True,
            extreme_funding_short_threshold=-0.0005,
            funding_counter_position_size=0.5,
            funding_counter_cooldown_minutes=0,
            long_only=False
        )
        strat = TrendStrategy(p)
        
        df = get_sample_data()
        # Extreme negative funding
        funding = pd.Series(-0.001, index=df.index)  # < -0.0005 threshold
        
        result = strat.generate_positions(df, funding=funding)
        
        # Should have counter long signals (0.5)
        assert (result["target_w"] == 0.5).any(), "Expected counter-long signal"
    
    def test_no_counter_on_normal_funding(self):
        """Should not fire counter signal on normal funding rates."""
        p = TrendParams(
            funding_counter_enabled=True,
            extreme_funding_long_threshold=0.0005,
            extreme_funding_short_threshold=-0.0005,
            funding_counter_position_size=0.5,
            long_only=False
        )
        strat = TrendStrategy(p)
        
        df = get_sample_data()
        # Normal funding (within thresholds)
        funding = pd.Series(0.0001, index=df.index)
        
        result = strat.generate_positions(df, funding=funding)
        
        # Should only have trend signals, not counter fractional values
        unique_vals = result["target_w"].unique()
        assert 0.5 not in unique_vals and -0.5 not in unique_vals


class TestFundingCounterCooldown:
    """Test that cooldown prevents rapid-fire signals."""
    
    def test_cooldown_prevents_rapid_fire(self):
        """Counter signals should respect cooldown_minutes."""
        p = TrendParams(
            funding_counter_enabled=True,
            extreme_funding_long_threshold=0.0005,
            funding_counter_position_size=0.5,
            funding_counter_cooldown_minutes=60,  # 60 min cooldown
            long_only=False
        )
        strat = TrendStrategy(p)
        
        # Create data with 15-min bars
        df = get_sample_data(n_bars=20)  # 5 hours of data
        
        # Alternate between extreme and normal funding
        funding = pd.Series(0.0001, index=df.index)
        # Set extreme funding at bar 0, 4, 8, 12, 16 (every hour)
        funding.iloc[0] = 0.001
        funding.iloc[4] = 0.001
        funding.iloc[8] = 0.001
        funding.iloc[12] = 0.001
        funding.iloc[16] = 0.001
        
        result = strat.generate_positions(df, funding=funding)
        
        # With 60-min cooldown on 15-min bars, signals at bar 0 and 4
        # should not both fire (too close)
        counter_signals = result["target_w"][result["target_w"] == -0.5]
        # Should have fewer counter signals due to cooldown
        assert len(counter_signals) < 5, "Cooldown should prevent all 5 signals"


class TestFundingCounterPositionSize:
    """Test that counter uses correct position size."""
    
    def test_position_size_respected(self):
        """Counter position should use funding_counter_position_size."""
        for pos_size in [0.25, 0.5, 0.75, 1.0]:
            p = TrendParams(
                funding_counter_enabled=True,
                extreme_funding_long_threshold=0.0005,
                funding_counter_position_size=pos_size,
                funding_counter_cooldown_minutes=0,
                long_only=False
            )
            strat = TrendStrategy(p)
            
            df = get_sample_data()
            funding = pd.Series(0.001, index=df.index)  # Extreme
            
            result = strat.generate_positions(df, funding=funding)
            
            # Counter short should be negative of position size
            assert (-pos_size in result["target_w"].values), f"Expected -{pos_size} counter"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
