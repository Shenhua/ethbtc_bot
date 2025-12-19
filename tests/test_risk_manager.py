"""
Tests for RiskManager class.

Verifies that the extracted RiskManager preserves all behavior
from the original _ensure_risk_state and _update_risk_state functions.
"""
import pytest
import pandas as pd
from core.risk_manager import RiskManager, RiskConfig


class TestRiskConfig:
    def test_from_config_defaults(self):
        """Test RiskConfig with minimal config object."""
        class MockRisk:
            risk_mode = "fixed_basis"
            max_dd_frac = 0.1
            max_dd_btc = 0.5
            max_daily_loss_frac = 0.05
            max_daily_loss_btc = 0.1
            drawdown_reset_days = 7.0
            drawdown_reset_score = 30.0
        
        config = RiskConfig.from_config(MockRisk())
        assert config.risk_mode == "fixed_basis"
        assert config.max_dd_frac == 0.1
        assert config.max_dd_btc == 0.5


class TestRiskManagerEnsureState:
    def test_initializes_missing_keys(self):
        """Test that ensure_state adds all required keys."""
        config = RiskConfig()
        rm = RiskManager(config)
        
        state = {}
        ts = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
        
        rm.ensure_state(state, wealth=1.0, ts=ts)
        
        assert "risk_equity_high" in state
        assert "risk_current_date" in state
        assert "risk_daily_start_wealth" in state
        assert "risk_daily_limit_hit" in state
        assert "risk_maxdd_hit" in state
        
        assert state["risk_equity_high"] == 1.0
        assert state["risk_daily_start_wealth"] == 1.0
        assert state["risk_daily_limit_hit"] == False
        assert state["risk_maxdd_hit"] == False
    
    def test_preserves_existing_values(self):
        """Test that ensure_state doesn't overwrite existing values."""
        config = RiskConfig()
        rm = RiskManager(config)
        
        state = {"risk_equity_high": 2.0}
        ts = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
        
        rm.ensure_state(state, wealth=1.0, ts=ts)
        
        # Should NOT overwrite existing HWM
        assert state["risk_equity_high"] == 2.0


class TestRiskManagerUpdate:
    def test_updates_hwm_when_not_crashed(self):
        """Test HWM updates when not in max DD state."""
        config = RiskConfig(risk_mode="dynamic", max_dd_frac=0.2)
        rm = RiskManager(config)
        
        state = {"risk_equity_high": 1.0, "risk_maxdd_hit": False}
        ts = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
        
        rm.ensure_state(state, wealth=1.0, ts=ts)
        rm.update(state, wealth=1.5, ts=ts)
        
        assert state["risk_equity_high"] == 1.5  # HWM should update

    def test_no_hwm_update_when_crashed(self):
        """Test HWM doesn't update when in max DD state."""
        config = RiskConfig(risk_mode="dynamic", max_dd_frac=0.2)
        rm = RiskManager(config)
        
        state = {"risk_equity_high": 1.0, "risk_maxdd_hit": True}
        ts = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
        
        rm.ensure_state(state, wealth=1.0, ts=ts)
        rm.update(state, wealth=1.5, ts=ts)
        
        assert state["risk_equity_high"] == 1.0  # HWM should NOT update

    def test_detects_max_drawdown(self):
        """Test max drawdown detection."""
        config = RiskConfig(risk_mode="dynamic", max_dd_frac=0.2)  # 20% max DD
        rm = RiskManager(config)
        
        state = {}
        ts = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
        
        rm.ensure_state(state, wealth=1.0, ts=ts)
        
        # Wealth drops 25% (below 20% threshold)
        rm.update(state, wealth=0.75, ts=ts)
        
        assert state["risk_maxdd_hit"] == True
        assert "risk_maxdd_hit_ts" in state

    def test_daily_loss_reset_on_new_day(self):
        """Test daily loss counters reset on new day."""
        config = RiskConfig(risk_mode="dynamic", max_daily_loss_frac=0.1)
        rm = RiskManager(config)
        
        state = {}
        day1 = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
        day2 = pd.Timestamp("2024-01-02 12:00:00", tz="UTC")
        
        rm.ensure_state(state, wealth=1.0, ts=day1)
        state["risk_daily_limit_hit"] = True
        
        # New day should reset
        rm.update(state, wealth=0.95, ts=day2)
        
        assert state["risk_daily_limit_hit"] == False


class TestRiskManagerHelpers:
    def test_is_halted(self):
        """Test is_halted returns true when limits hit."""
        config = RiskConfig()
        rm = RiskManager(config)
        
        assert rm.is_halted({}) == False
        assert rm.is_halted({"risk_maxdd_hit": True}) == True
        assert rm.is_halted({"risk_daily_limit_hit": True}) == True
        assert rm.is_halted({"risk_maxdd_hit": True, "risk_daily_limit_hit": True}) == True
    
    def test_get_drawdown_pct(self):
        """Test drawdown percentage calculation."""
        config = RiskConfig()
        rm = RiskManager(config)
        
        state = {"risk_equity_high": 1.0}
        
        assert rm.get_drawdown_pct(state, wealth=1.0) == 0.0
        assert rm.get_drawdown_pct(state, wealth=0.8) == pytest.approx(0.2)
        assert rm.get_drawdown_pct(state, wealth=0.5) == pytest.approx(0.5)
    
    def test_can_phoenix_reset(self):
        """Test Phoenix reset conditions check."""
        config = RiskConfig(drawdown_reset_days=1.0, drawdown_reset_score=30.0)
        rm = RiskManager(config)
        
        crash_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        check_time = pd.Timestamp("2024-01-02 12:00:00", tz="UTC")  # >24h later
        
        state = {
            "risk_maxdd_hit": True,
            "risk_maxdd_hit_ts": crash_time.isoformat()
        }
        
        # Score too low
        assert rm.can_phoenix_reset(state, check_time, current_regime_score=20.0) == False
        
        # Score high enough
        assert rm.can_phoenix_reset(state, check_time, current_regime_score=35.0) == True
    
    def test_reset_phoenix(self):
        """Test Phoenix reset clears state correctly."""
        config = RiskConfig()
        rm = RiskManager(config)
        
        state = {
            "risk_maxdd_hit": True,
            "risk_maxdd_hit_ts": "2024-01-01T00:00:00",
            "risk_equity_high": 0.8,
            "alert_sent_maxdd": True
        }
        
        rm.reset_phoenix(state, wealth=0.9)
        
        assert state["risk_maxdd_hit"] == False
        assert state["risk_maxdd_hit_ts"] == None
        assert state["risk_equity_high"] == 0.9
        assert state["alert_sent_maxdd"] == False
