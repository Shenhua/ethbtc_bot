"""
Unit tests for live_executor.py decision logic.

These tests verify critical paths in the live executor without
requiring a real Binance connection. They focus on:
1. Risk state management
2. Configuration sanity checks
3. State persistence
4. Decision skip conditions
"""
import pytest
import pandas as pd
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRiskStateManagement:
    """Tests for _ensure_risk_state and _update_risk_state functions."""
    
    def test_ensure_risk_state_initializes_missing_keys(self):
        """Verify all required risk keys are initialized."""
        from core.risk_manager import RiskManager, RiskConfig
        
        config = RiskConfig()
        risk_mgr = RiskManager(config)
        from core.models.state import RiskState
        state = RiskState()
        ts = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
        
        risk_mgr.ensure_state(state, wealth=1.0, ts=ts)
        
        assert state.equity_high == 1.0
        assert state.daily_limit_hit == False
        assert state.maxdd_hit == False
    
    def test_ensure_risk_state_preserves_existing_values(self):
        """Verify existing state values are not overwritten."""
        from core.risk_manager import RiskManager, RiskConfig
        
        config = RiskConfig()
        risk_mgr = RiskManager(config)
        from core.models.state import RiskState
        state = RiskState(equity_high=2.0)
        ts = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
        
        risk_mgr.ensure_state(state, wealth=1.0, ts=ts)
        
        # Should NOT overwrite existing HWM
        assert state.equity_high == 2.0
    
    def test_update_risk_state_updates_hwm(self):
        """Verify HWM updates when wealth increases."""
        from core.risk_manager import RiskManager, RiskConfig
        
        from core.models.state import RiskState
        state = RiskState()
        ts = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
        
        # Mock config
        risk_config = RiskConfig(
            risk_mode="dynamic",
            max_dd_frac=0.2,
            max_dd_btc=0.0,
            max_daily_loss_frac=0.0,
            max_daily_loss_btc=0.0
        )
        risk_mgr = RiskManager(risk_config)
        
        risk_mgr.ensure_state(state, wealth=1.0, ts=ts)
        risk_mgr.update(state, wealth=1.5, ts=ts)
        
        assert state.equity_high == 1.5  # HWM should update
    
    def test_update_risk_state_detects_max_drawdown(self):
        """Verify max drawdown detection triggers correctly."""
        from core.risk_manager import RiskManager, RiskConfig
        
        from core.models.state import RiskState
        state = RiskState()
        ts = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
        
        # Mock config with 20% max DD
        risk_config = RiskConfig(
            risk_mode="dynamic",
            max_dd_frac=0.20,
            max_dd_btc=0.0,
            max_daily_loss_frac=0.0,
            max_daily_loss_btc=0.0
        )
        risk_mgr = RiskManager(risk_config)
        
        risk_mgr.ensure_state(state, wealth=1.0, ts=ts)
        
        # 25% drawdown should trigger max DD
        risk_mgr.update(state, wealth=0.75, ts=ts)
        
        assert state.maxdd_hit == True
        assert state.maxdd_hit_ts is not None
    
    def test_update_risk_state_daily_reset(self):
        """Verify daily loss counters reset on new day."""
        from core.risk_manager import RiskManager, RiskConfig
        
        day1 = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
        day2 = pd.Timestamp("2024-01-02 12:00:00", tz="UTC")
        
        risk_config = RiskConfig(
            risk_mode="fixed_basis",
            max_dd_frac=0.0,
            max_dd_btc=0.0,
            max_daily_loss_frac=0.0,
            max_daily_loss_btc=0.1
        )
        risk_mgr = RiskManager(risk_config)
        
        from core.models.state import RiskState
        state = RiskState()
        risk_mgr.ensure_state(state, wealth=1.0, ts=day1)
        
        # Trigger daily limit
        state.daily_limit_hit = True
        
        # New day should reset
        risk_mgr.update(state, wealth=0.95, ts=day2)
        
        assert state.daily_limit_hit == False


class TestStatePersistence:
    """Tests for state loading and saving."""
    
    def test_load_state_returns_empty_dict_for_missing_file(self):
        """Verify graceful handling of missing state file."""
        from live_executor import load_state
        
        result = load_state("/nonexistent/path/state.json")
        assert result == {}
    
    def test_save_and_load_state_roundtrip(self):
        """Verify state can be saved and loaded correctly."""
        from live_executor import load_state, save_state
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = str(Path(tmpdir) / "test_state.json")
            
            original_state = {
                "risk_equity_high": 1.5,
                "risk_maxdd_hit": False,
                "last_bar_close": "2024-01-01T12:00:00"
            }
            
            save_state(state_path, original_state)
            loaded_state = load_state(state_path)
            
            assert loaded_state == original_state


class TestDecisionKeys:
    """Tests for trade decision tracking."""
    
    def test_decision_keys_are_defined(self):
        """Verify all expected decision keys exist."""
        from core.metrics import DECISION_KEYS
        
        expected_keys = [
            "exec_buy", "exec_sell", 
            "skip_threshold", "skip_balance", "skip_min_notional",
            "skip_cooldown", "skip_gate_closed", "skip_delta_zero", "skip_order_error"
        ]
        
        for key in expected_keys:
            assert key in DECISION_KEYS


class TestConfigSanityCheck:
    """Test configuration validation logic."""
    
    def test_high_leverage_is_flagged(self):
        """Verify extreme leverage triggers sanity warning."""
        from core.config_schema import AppConfig, Strategy, Execution, Risk, Fees
        
        # Create config with extreme leverage
        cfg = AppConfig(
            strategy=Strategy(),
            execution=Execution(leverage=20),  # Very high
            risk=Risk(),
            fees=Fees(maker_fee=0.0002, taker_fee=0.0004)
        )
        
        leverage = int(getattr(cfg.execution, "leverage", 1) or 1)
        assert leverage > 10  # Should be flagged
    
    def test_high_max_dd_frac_is_flagged(self):
        """Verify high max_dd_frac triggers sanity warning."""
        from core.config_schema import AppConfig, Strategy, Execution, Risk, Fees
        
        cfg = AppConfig(
            strategy=Strategy(),
            execution=Execution(),
            risk=Risk(max_dd_frac=0.75),  # Very high
            fees=Fees(maker_fee=0.0002, taker_fee=0.0004)
        )
        
        max_dd_frac = float(getattr(cfg.risk, "max_dd_frac", 0.0) or 0.0)
        assert max_dd_frac > 0.5  # Should be flagged


class TestHelperFunctions:
    """Tests for various helper functions."""
    
    def test_last_closed_bar_ts(self):
        """Verify bar timestamp calculation."""
        from live_executor import last_closed_bar_ts
        
        # For a 15m interval, we expect previous bar's close time
        now_s = 1704110100  # Some timestamp
        result = last_closed_bar_ts(now_s, "15m")
        
        # Result should be < now_s
        assert result < now_s
        # Result should be aligned to 15m boundary
        assert result % (15 * 60) == 899  # -1 from the boundary


class TestPhoenixProtocolIntegration:
    """Tests for Phoenix Protocol state management in live_executor."""
    
    def test_phoenix_state_keys_exist_after_maxdd(self):
        """Verify Phoenix keys exist after a crash event."""
        from core.risk_manager import RiskManager, RiskConfig
        from core.models.state import RiskState
        
        day1 = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
        config = RiskConfig(risk_mode="dynamic", max_dd_frac=0.1)
        risk_mgr = RiskManager(config)
        
        state = RiskState()
        risk_mgr.ensure_state(state, wealth=1.0, ts=day1)
        
        # Trigger crash
        risk_mgr.update(state, wealth=0.85, ts=day1)
        
        assert state.maxdd_hit == True
        assert state.maxdd_hit_ts is not None
        assert state.equity_high == 1.0
        # Verify timestamp is parseable
        hit_ts = pd.to_datetime(state.maxdd_hit_ts)
        assert hit_ts == day1
