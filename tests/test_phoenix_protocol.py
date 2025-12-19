"""
Tests for Phoenix Protocol recovery logic.

The Phoenix Protocol allows the bot to resume trading after a 
Max Drawdown event if certain conditions are met:
1. Time cooldown has passed (drawdown_reset_days)
2. Regime score is favorable (>= drawdown_reset_score)

These tests verify the RiskManager's Phoenix-related methods.
"""
import pytest
import pandas as pd
from core.risk_manager import RiskManager, RiskConfig


class TestPhoenixConditions:
    """Tests for Phoenix Protocol activation conditions."""
    
    def test_phoenix_not_triggered_before_cooldown(self):
        """Phoenix should NOT trigger if time cooldown hasn't passed."""
        config = RiskConfig(
            drawdown_reset_days=1.0,  # 24 hour cooldown
            drawdown_reset_score=30.0
        )
        rm = RiskManager(config)
        
        crash_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        # Only 12 hours later (less than 24 hour cooldown)
        check_time = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
        
        state = {
            "risk_maxdd_hit": True,
            "risk_maxdd_hit_ts": crash_time.isoformat()
        }
        
        # High score but not enough time
        can_reset = rm.can_phoenix_reset(state, check_time, current_regime_score=50.0)
        assert can_reset == False
    
    def test_phoenix_not_triggered_with_low_score(self):
        """Phoenix should NOT trigger if regime score is too low."""
        config = RiskConfig(
            drawdown_reset_days=1.0,
            drawdown_reset_score=30.0  # Requires score >= 30
        )
        rm = RiskManager(config)
        
        crash_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        # 48 hours later (enough time)
        check_time = pd.Timestamp("2024-01-03 00:00:00", tz="UTC")
        
        state = {
            "risk_maxdd_hit": True,
            "risk_maxdd_hit_ts": crash_time.isoformat()
        }
        
        # Low score (market is choppy)
        can_reset = rm.can_phoenix_reset(state, check_time, current_regime_score=15.0)
        assert can_reset == False
    
    def test_phoenix_triggers_with_both_conditions_met(self):
        """Phoenix SHOULD trigger when both conditions are satisfied."""
        config = RiskConfig(
            drawdown_reset_days=1.0,
            drawdown_reset_score=30.0
        )
        rm = RiskManager(config)
        
        crash_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        # 36 hours later (> 24 hour cooldown)
        check_time = pd.Timestamp("2024-01-02 12:00:00", tz="UTC")
        
        state = {
            "risk_maxdd_hit": True,
            "risk_maxdd_hit_ts": crash_time.isoformat()
        }
        
        # High score and enough time
        can_reset = rm.can_phoenix_reset(state, check_time, current_regime_score=35.0)
        assert can_reset == True
    
    def test_phoenix_not_triggered_when_not_in_maxdd_state(self):
        """Phoenix should NOT trigger if we're not in maxdd state."""
        config = RiskConfig(
            drawdown_reset_days=1.0,
            drawdown_reset_score=30.0
        )
        rm = RiskManager(config)
        
        check_time = pd.Timestamp("2024-01-02 00:00:00", tz="UTC")
        
        state = {
            "risk_maxdd_hit": False,  # Not in crash state
        }
        
        can_reset = rm.can_phoenix_reset(state, check_time, current_regime_score=50.0)
        assert can_reset == False


class TestPhoenixReset:
    """Tests for Phoenix Protocol state reset."""
    
    def test_reset_clears_maxdd_state(self):
        """Verify reset_phoenix clears crash state correctly."""
        config = RiskConfig()
        rm = RiskManager(config)
        
        state = {
            "risk_maxdd_hit": True,
            "risk_maxdd_hit_ts": "2024-01-01T00:00:00",
            "risk_equity_high": 0.8,  # Old HWM from before crash
            "alert_sent_maxdd": True
        }
        
        rm.reset_phoenix(state, wealth=0.9)
        
        assert state["risk_maxdd_hit"] == False
        assert state["risk_maxdd_hit_ts"] is None
        assert state["risk_equity_high"] == 0.9  # Reset to current wealth
        assert state["alert_sent_maxdd"] == False
    
    def test_reset_sets_new_hwm(self):
        """Verify reset establishes new HWM at current wealth."""
        config = RiskConfig()
        rm = RiskManager(config)
        
        state = {
            "risk_maxdd_hit": True,
            "risk_maxdd_hit_ts": "2024-01-01T00:00:00",
            "risk_equity_high": 1.0,  # Old peak before crash
        }
        
        # After crash, wealth is now 0.7
        rm.reset_phoenix(state, wealth=0.7)
        
        # HWM should be reset to current wealth (0.7), not old peak (1.0)
        assert state["risk_equity_high"] == 0.7


class TestPhoenixDisabled:
    """Tests for when Phoenix Protocol is disabled."""
    
    def test_phoenix_disabled_with_zero_reset_days(self):
        """Phoenix should never trigger if reset_days is 0."""
        config = RiskConfig(
            drawdown_reset_days=0.0,  # Disabled
            drawdown_reset_score=30.0
        )
        rm = RiskManager(config)
        
        crash_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        check_time = pd.Timestamp("2024-01-10 00:00:00", tz="UTC")  # 9 days later
        
        state = {
            "risk_maxdd_hit": True,
            "risk_maxdd_hit_ts": crash_time.isoformat()
        }
        
        can_reset = rm.can_phoenix_reset(state, check_time, current_regime_score=50.0)
        assert can_reset == False
    
    def test_phoenix_missing_timestamp_returns_false(self):
        """Phoenix should not trigger if crash timestamp is missing."""
        config = RiskConfig(
            drawdown_reset_days=1.0,
            drawdown_reset_score=30.0
        )
        rm = RiskManager(config)
        
        check_time = pd.Timestamp("2024-01-02 00:00:00", tz="UTC")
        
        state = {
            "risk_maxdd_hit": True,
            # Missing risk_maxdd_hit_ts
        }
        
        can_reset = rm.can_phoenix_reset(state, check_time, current_regime_score=50.0)
        assert can_reset == False


class TestPhoenixEdgeCases:
    """Edge case tests for Phoenix Protocol."""
    
    def test_phoenix_exactly_at_cooldown_boundary(self):
        """Test behavior at exact cooldown boundary."""
        config = RiskConfig(
            drawdown_reset_days=1.0,  # Exactly 24 hours
            drawdown_reset_score=30.0
        )
        rm = RiskManager(config)
        
        crash_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        # Exactly 24 hours later
        check_time = pd.Timestamp("2024-01-02 00:00:00", tz="UTC")
        
        state = {
            "risk_maxdd_hit": True,
            "risk_maxdd_hit_ts": crash_time.isoformat()
        }
        
        # Should trigger at exactly the boundary
        can_reset = rm.can_phoenix_reset(state, check_time, current_regime_score=35.0)
        assert can_reset == True
    
    def test_phoenix_exactly_at_score_boundary(self):
        """Test behavior at exact score boundary."""
        config = RiskConfig(
            drawdown_reset_days=1.0,
            drawdown_reset_score=30.0  # Exactly 30
        )
        rm = RiskManager(config)
        
        crash_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        check_time = pd.Timestamp("2024-01-02 12:00:00", tz="UTC")
        
        state = {
            "risk_maxdd_hit": True,
            "risk_maxdd_hit_ts": crash_time.isoformat()
        }
        
        # Score exactly at threshold
        can_reset = rm.can_phoenix_reset(state, check_time, current_regime_score=30.0)
        assert can_reset == True
    
    def test_phoenix_just_below_score_threshold(self):
        """Test behavior just below score threshold."""
        config = RiskConfig(
            drawdown_reset_days=1.0,
            drawdown_reset_score=30.0
        )
        rm = RiskManager(config)
        
        crash_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        check_time = pd.Timestamp("2024-01-02 12:00:00", tz="UTC")
        
        state = {
            "risk_maxdd_hit": True,
            "risk_maxdd_hit_ts": crash_time.isoformat()
        }
        
        # Score just below threshold
        can_reset = rm.can_phoenix_reset(state, check_time, current_regime_score=29.9)
        assert can_reset == False
