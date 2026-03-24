"""
Tests for Dynamic Kelly Criterion and RollingTradeStats.

Validates:
- Rolling statistics calculation
- Kelly criterion with dynamic vs static params
- Edge cases (insufficient data, negative Kelly)
"""

import pytest
from core.position_sizer import (
    PositionSizer, PositionSizerConfig, RollingTradeStats
)


class TestRollingTradeStats:
    """Tests for RollingTradeStats class."""
    
    def test_init_empty(self):
        """New stats tracker should be empty."""
        stats = RollingTradeStats(lookback=100, min_trades=20)
        assert len(stats) == 0
        assert not stats.has_sufficient_data()
        assert stats.get_stats() is None
    
    def test_add_trades(self):
        """Adding trades should increase count."""
        stats = RollingTradeStats(lookback=100, min_trades=5)
        
        stats.add_trade(0.02)
        stats.add_trade(-0.01)
        stats.add_trade(0.03)
        
        assert len(stats) == 3
        assert not stats.has_sufficient_data()
    
    def test_sufficient_data_threshold(self):
        """Stats should be available after min_trades reached."""
        stats = RollingTradeStats(lookback=100, min_trades=5)
        
        # Add 4 trades (mix of wins/losses) - not enough
        stats.add_trade(0.01)
        stats.add_trade(-0.01)
        stats.add_trade(0.02)
        stats.add_trade(-0.02)
        assert stats.get_stats() is None

        # Add 5th trade (a win) - now enough AND we have both wins and losses
        stats.add_trade(0.01)
        assert stats.get_stats() is not None
    
    def test_win_rate_calculation(self):
        """Win rate should be correctly calculated."""
        stats = RollingTradeStats(lookback=100, min_trades=5)
        
        # 3 wins, 2 losses = 60% win rate
        stats.add_trade(0.02)
        stats.add_trade(-0.01)
        stats.add_trade(0.03)
        stats.add_trade(-0.02)
        stats.add_trade(0.01)
        
        result = stats.get_stats()
        assert result is not None
        
        win_rate, avg_win, avg_loss = result
        assert win_rate == pytest.approx(0.6, rel=0.01)
    
    def test_avg_win_avg_loss(self):
        """Average win and loss should be correctly calculated."""
        stats = RollingTradeStats(lookback=100, min_trades=5)
        
        # Wins: 0.02, 0.04 -> avg = 0.03
        # Losses: -0.01, -0.03 -> avg = 0.02 (absolute)
        stats.add_trade(0.02)
        stats.add_trade(-0.01)
        stats.add_trade(0.04)
        stats.add_trade(-0.03)
        stats.add_trade(0.01)  # Another win
        
        result = stats.get_stats()
        win_rate, avg_win, avg_loss = result
        
        # 3 wins: 0.02, 0.04, 0.01 -> avg = 0.0233
        assert avg_win == pytest.approx(0.0233, rel=0.1)
        # 2 losses: 0.01, 0.03 -> avg = 0.02
        assert avg_loss == pytest.approx(0.02, rel=0.01)
    
    def test_lookback_window(self):
        """Old trades should be dropped after lookback exceeded."""
        stats = RollingTradeStats(lookback=5, min_trades=2)

        # Fill window with mostly wins: 4 wins + 1 loss → 80% win rate
        stats.add_trade(0.01)
        stats.add_trade(0.01)
        stats.add_trade(0.01)
        stats.add_trade(0.01)
        stats.add_trade(-0.01)

        assert len(stats) == 5
        assert stats.get_stats()[0] == pytest.approx(0.8)  # 80% win rate

        # Push in 5 more trades (4 losses + 1 win) — old window drops off
        stats.add_trade(-0.01)
        stats.add_trade(-0.01)
        stats.add_trade(-0.01)
        stats.add_trade(-0.01)
        stats.add_trade(0.01)

        assert len(stats) == 5
        # New window: 4 losses + 1 win → 20% win rate
        assert stats.get_stats()[0] == pytest.approx(0.2)  # 20% win rate now


class TestDynamicKelly:
    """Tests for Dynamic Kelly Criterion in PositionSizer."""
    
    def test_kelly_uses_static_when_insufficient_data(self):
        """Kelly should use static config when trade history is insufficient."""
        config = PositionSizerConfig(
            mode="kelly",
            kelly_win_rate=0.6,
            kelly_avg_win=0.03,
            kelly_avg_loss=0.02,
            kelly_min_trades=10
        )
        sizer = PositionSizer(config)
        
        # Add only 5 trades (less than min_trades=10)
        for _ in range(5):
            sizer.record_trade(0.01)
        
        # Should use static config values
        step = sizer.calculate_step(realized_vol=0.5)
        assert step > 0
    
    def test_kelly_uses_dynamic_when_sufficient_data(self):
        """Kelly should use dynamic stats when enough trades recorded."""
        config = PositionSizerConfig(
            mode="kelly",
            kelly_win_rate=0.5,  # Static default
            kelly_avg_win=0.01,
            kelly_avg_loss=0.01,
            kelly_min_trades=5,
            kelly_fraction=0.5
        )
        sizer = PositionSizer(config)
        
        # Record 5 very profitable trades (80% win rate, good R:R)
        sizer.record_trade(0.05)
        sizer.record_trade(0.04)
        sizer.record_trade(0.03)
        sizer.record_trade(0.02)
        sizer.record_trade(-0.01)
        
        # Now has sufficient data - should use dynamic stats
        step1 = sizer.calculate_step(realized_vol=0.5)
        
        # Record 5 losing trades to worsen stats
        for _ in range(5):
            sizer.record_trade(-0.05)
        
        step2 = sizer.calculate_step(realized_vol=0.5)
        
        # Step should decrease after poor performance
        assert step2 < step1
    
    def test_negative_kelly_uses_min_step(self):
        """Negative Kelly (losing strategy) should return min_step."""
        config = PositionSizerConfig(
            mode="kelly",
            kelly_win_rate=0.3,  # Low win rate
            kelly_avg_win=0.01,  # Small wins
            kelly_avg_loss=0.05,  # Large losses
            kelly_min_trades=5,
            min_step=0.1
        )
        sizer = PositionSizer(config)
        
        # Record losing trades to confirm negative Kelly
        for _ in range(5):
            sizer.record_trade(-0.05)
        
        step = sizer.calculate_step(realized_vol=0.5)
        
        # Should return min_step for negative expectancy
        assert step == config.min_step


class TestPositionSizerModes:
    """Test all position sizer modes work correctly."""
    
    def test_static_mode(self):
        """Static mode should return base_step."""
        config = PositionSizerConfig(mode="static", base_step=0.5)
        sizer = PositionSizer(config)
        
        assert sizer.calculate_step(0.5) == 0.5
        assert sizer.calculate_step(1.0) == 0.5
        assert sizer.calculate_step(0.1) == 0.5
    
    def test_volatility_mode_scaling(self):
        """Volatility mode should scale inversely with realized vol."""
        config = PositionSizerConfig(
            mode="volatility",
            base_step=0.5,
            target_vol=0.5,
            min_step=0.1,
            max_step=1.0
        )
        sizer = PositionSizer(config)
        
        # Low vol = larger step
        step_low = sizer.calculate_step(0.25)
        # High vol = smaller step
        step_high = sizer.calculate_step(1.0)
        
        assert step_low > step_high
        assert step_low <= 1.0
        assert step_high >= 0.1
    
    def test_kelly_mode_basic(self):
        """Kelly mode should produce reasonable step sizes."""
        config = PositionSizerConfig(
            mode="kelly",
            kelly_win_rate=0.55,
            kelly_avg_win=0.02,
            kelly_avg_loss=0.01,
            kelly_fraction=0.5,
            min_step=0.1,
            max_step=1.0
        )
        sizer = PositionSizer(config)
        
        step = sizer.calculate_step(0.5)
        
        assert 0.1 <= step <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
