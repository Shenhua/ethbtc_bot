"""
Unit tests for Dynamic Position Sizer.
"""
import sys
import os
import pytest

# --- MAGIC PATH FIX ---
# Allow importing 'core' even if running from tests/ folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ----------------------

from core.position_sizer import PositionSizer, PositionSizerConfig
class TestPositionSizerConfig:
    """Test configuration validation."""
    
    def test_valid_config(self):
        """Test valid configuration passes validation."""
        config = PositionSizerConfig(
            mode="volatility",
            base_step=0.5,
            target_vol=0.5,
            min_step=0.1,
            max_step=1.0
        )
        assert config.mode == "volatility"
        assert config.base_step == 0.5
    
    def test_invalid_base_step(self):
        """Test invalid base_step raises error."""
        with pytest.raises(ValueError, match="base_step must be in"):
            PositionSizerConfig(base_step=1.5)
    
    def test_min_greater_than_max(self):
        """Test min_step > max_step raises error."""
        with pytest.raises(ValueError, match="min_step .* > max_step"):
            PositionSizerConfig(min_step=0.8, max_step=0.5)
    
    def test_negative_target_vol(self):
        """Test negative target_vol raises error."""
        with pytest.raises(ValueError, match="target_vol must be positive"):
            PositionSizerConfig(target_vol=-0.5)


class TestStaticMode:
    """Test static position sizing mode."""
    
    def test_static_mode_returns_base_step(self):
        """Static mode should always return base_step."""
        config = PositionSizerConfig(mode="static", base_step=0.5)
        sizer = PositionSizer(config)
        
        # Should return base_step regardless of volatility
        assert sizer.calculate_step(0.1) == 0.5
        assert sizer.calculate_step(0.5) == 0.5
        assert sizer.calculate_step(1.0) == 0.5


class TestVolatilityTargeting:
    """Test volatility targeting mode."""
    
    def test_normal_volatility(self):
        """Test step size when realized_vol equals target_vol."""
        config = PositionSizerConfig(
            mode="volatility",
            base_step=0.5,
            target_vol=0.5,
            min_step=0.1,
            max_step=1.0
        )
        sizer = PositionSizer(config)
        
        # When rv = target, step should equal base_step
        assert sizer.calculate_step(0.5) == 0.5
    
    def test_low_volatility_increases_step(self):
        """Test step size increases in low volatility."""
        config = PositionSizerConfig(
            mode="volatility",
            base_step=0.5,
            target_vol=0.5,
            min_step=0.1,
            max_step=1.0
        )
        sizer = PositionSizer(config)
        
        # Low volatility -> larger step
        # step = 0.5 * (0.5 / 0.25) = 1.0
        assert sizer.calculate_step(0.25) == 1.0  # Capped at max_step
    
    def test_high_volatility_decreases_step(self):
        """Test step size decreases in high volatility."""
        config = PositionSizerConfig(
            mode="volatility",
            base_step=0.5,
            target_vol=0.5,
            min_step=0.1,
            max_step=1.0
        )
        sizer = PositionSizer(config)
        
        # High volatility -> smaller step
        # step = 0.5 * (0.5 / 1.0) = 0.25
        assert sizer.calculate_step(1.0) == 0.25
    
    def test_very_high_volatility_hits_floor(self):
        """Test step size is clamped to min_step."""
        config = PositionSizerConfig(
            mode="volatility",
            base_step=0.5,
            target_vol=0.5,
            min_step=0.2,
            max_step=1.0
        )
        sizer = PositionSizer(config)
        
        # Very high volatility -> clamp to min_step
        # step = 0.5 * (0.5 / 5.0) = 0.05 -> clamped to 0.2
        assert sizer.calculate_step(5.0) == 0.2
    
    def test_zero_volatility_fallback(self):
        """Test handling of zero volatility."""
        config = PositionSizerConfig(
            mode="volatility",
            base_step=0.5,
            target_vol=0.5
        )
        sizer = PositionSizer(config)
        
        # Zero volatility should fallback to base_step
        assert sizer.calculate_step(0.0) == 0.5
    
    def test_nan_volatility_fallback(self):
        """Test handling of NaN volatility."""
        config = PositionSizerConfig(
            mode="volatility",
            base_step=0.5,
            target_vol=0.5
        )
        sizer = PositionSizer(config)
        
        # NaN volatility should fallback to base_step
        assert sizer.calculate_step(float('nan')) == 0.5


class TestKellyCriterion:
    """Test Kelly Criterion mode."""
    
    def test_kelly_basic(self):
        """Test Kelly Criterion basic calculation."""
        config = PositionSizerConfig(
            mode="kelly",
            base_step=0.5,
            target_vol=0.5,
            kelly_win_rate=0.6,
            kelly_avg_win=0.02,
            kelly_avg_loss=0.01,
            min_step=0.1,
            max_step=1.0
        )
        sizer = PositionSizer(config)
        
        # Kelly = (0.6*0.02 - 0.4*0.01) / 0.02 = (0.012 - 0.004) / 0.02 = 0.4
        # Half-Kelly = 0.2
        # With vol adjustment at rv=0.5: vol_adj = 0.5
        # step = 0.2 * 0.5 = 0.1
        step = sizer.calculate_step(0.5)
        assert 0.1 <= step <= 1.0  # Within bounds
        assert step == pytest.approx(0.1, abs=0.01)
    
    def test_kelly_positive_expectancy(self):
        """Test Kelly with positive expectancy returns valid step."""
        config = PositionSizerConfig(
            mode="kelly",
            kelly_win_rate=0.55,
            kelly_avg_win=0.03,
            kelly_avg_loss=0.02
        )
        sizer = PositionSizer(config)
        
        step = sizer.calculate_step(0.5)
        assert step > 0  # Should be positive
        assert step <= 1.0  # Should be clamped
    
    def test_kelly_invalid_avg_win(self):
        """Test Kelly with invalid avg_win falls back to base_step."""
        config = PositionSizerConfig(
            mode="kelly",
            base_step=0.5,
            kelly_avg_win=0.0  # Invalid
        )
        sizer = PositionSizer(config)
        
        # Should fallback to base_step
        assert sizer.calculate_step(0.5) == 0.5


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_small_step(self):
        """Test with very small min_step."""
        config = PositionSizerConfig(
            mode="volatility",
            base_step=0.1,
            target_vol=0.5,
            min_step=0.01,
            max_step=1.0
        )
        sizer = PositionSizer(config)
        
        # High volatility scenario
        step = sizer.calculate_step(10.0)
        assert step == 0.01  # Should hit floor
    
    def test_step_equals_one(self):
        """Test max_step = 1.0 case."""
        config = PositionSizerConfig(
            mode="volatility",
            base_step=0.5,
            target_vol=1.0,
            max_step=1.0
        )
        sizer = PositionSizer(config)
        
        # Very low volatility
        step = sizer.calculate_step(0.1)
        assert step == 1.0  # Should hit ceiling


class TestRealWorldScenarios:
    """Test with realistic market scenarios."""
    
    def test_bull_market_scenario(self):
        """Test during bull market (low volatility)."""
        config = PositionSizerConfig(
            mode="volatility",
            base_step=0.5,
            target_vol=0.5,
            min_step=0.2,
            max_step=1.0
        )
        sizer = PositionSizer(config)
        
        # Bull market: low volatility ~20%
        step = sizer.calculate_step(0.2)
        assert step > 0.5  # Should increase
        assert step <= 1.0  # Respects max
    
    def test_crash_scenario(self):
        """Test during market crash (high volatility)."""
        config = PositionSizerConfig(
            mode="volatility",
            base_step=0.5,
            target_vol=0.5,
            min_step=0.1,
            max_step=1.0
        )
        sizer = PositionSizer(config)
        
        # Crash: high volatility ~200%
        step = sizer.calculate_step(2.0)
        assert step < 0.5  # Should decrease
        assert step >= 0.1  # Respects min
    
    def test_sideways_market(self):
        """Test during sideways market (normal volatility)."""
        config = PositionSizerConfig(
            mode="volatility",
            base_step=0.5,
            target_vol=0.5,
            min_step=0.1,
            max_step=1.0
        )
        sizer = PositionSizer(config)
        
        # Sideways: normal volatility ~50%
        step = sizer.calculate_step(0.5)
        assert step == 0.5  # Should stay at base
