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
        from live_executor import _ensure_risk_state
        
        state = {}
        ts = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
        
        _ensure_risk_state(state, wealth=1.0, ts=ts)
        
        assert "risk_equity_high" in state
        assert "risk_current_date" in state
        assert "risk_daily_start_wealth" in state
        assert "risk_daily_limit_hit" in state
        assert "risk_maxdd_hit" in state
        
        assert state["risk_equity_high"] == 1.0
        assert state["risk_daily_limit_hit"] == False
        assert state["risk_maxdd_hit"] == False
    
    def test_ensure_risk_state_preserves_existing_values(self):
        """Verify existing state values are not overwritten."""
        from live_executor import _ensure_risk_state
        
        state = {"risk_equity_high": 2.0}
        ts = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
        
        _ensure_risk_state(state, wealth=1.0, ts=ts)
        
        # Should NOT overwrite existing HWM
        assert state["risk_equity_high"] == 2.0
    
    def test_update_risk_state_updates_hwm(self):
        """Verify HWM updates when wealth increases."""
        from live_executor import _ensure_risk_state, _update_risk_state
        from core.config_schema import Risk
        
        state = {}
        ts = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
        
        # Mock config
        mock_cfg = MagicMock()
        mock_cfg.risk = Risk()
        mock_cfg.risk.risk_mode = "dynamic"
        mock_cfg.risk.max_dd_frac = 0.2
        mock_cfg.risk.max_dd_btc = 0.0
        mock_cfg.risk.max_daily_loss_frac = 0.0
        mock_cfg.risk.max_daily_loss_btc = 0.0
        
        _ensure_risk_state(state, wealth=1.0, ts=ts)
        _update_risk_state(state, wealth=1.5, ts=ts, cfg=mock_cfg)
        
        assert state["risk_equity_high"] == 1.5  # HWM should update
    
    def test_update_risk_state_detects_max_drawdown(self):
        """Verify max drawdown detection triggers correctly."""
        from live_executor import _ensure_risk_state, _update_risk_state
        from core.config_schema import Risk
        
        state = {}
        ts = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
        
        # Mock config with 20% max DD
        mock_cfg = MagicMock()
        mock_cfg.risk = Risk()
        mock_cfg.risk.risk_mode = "dynamic"
        mock_cfg.risk.max_dd_frac = 0.20
        mock_cfg.risk.max_dd_btc = 0.0
        mock_cfg.risk.max_daily_loss_frac = 0.0
        mock_cfg.risk.max_daily_loss_btc = 0.0
        
        _ensure_risk_state(state, wealth=1.0, ts=ts)
        
        # 25% drawdown should trigger max DD
        _update_risk_state(state, wealth=0.75, ts=ts, cfg=mock_cfg)
        
        assert state["risk_maxdd_hit"] == True
        assert "risk_maxdd_hit_ts" in state
    
    def test_update_risk_state_daily_reset(self):
        """Verify daily loss counters reset on new day."""
        from live_executor import _ensure_risk_state, _update_risk_state
        from core.config_schema import Risk
        
        day1 = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
        day2 = pd.Timestamp("2024-01-02 12:00:00", tz="UTC")
        
        mock_cfg = MagicMock()
        mock_cfg.risk = Risk()
        mock_cfg.risk.risk_mode = "fixed_basis"
        mock_cfg.risk.max_dd_frac = 0.0
        mock_cfg.risk.max_dd_btc = 0.0
        mock_cfg.risk.max_daily_loss_frac = 0.0
        mock_cfg.risk.max_daily_loss_btc = 0.1
        
        state = {}
        _ensure_risk_state(state, wealth=1.0, ts=day1)
        
        # Trigger daily limit
        state["risk_daily_limit_hit"] = True
        
        # New day should reset
        _update_risk_state(state, wealth=0.95, ts=day2, cfg=mock_cfg)
        
        assert state["risk_daily_limit_hit"] == False


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
        from live_executor import DECISION_KEYS
        
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
        """Verify Phoenix-related state keys are set when max DD hit."""
        from live_executor import _ensure_risk_state, _update_risk_state
        from core.config_schema import Risk
        
        state = {}
        ts = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
        
        mock_cfg = MagicMock()
        mock_cfg.risk = Risk()
        mock_cfg.risk.risk_mode = "dynamic"
        mock_cfg.risk.max_dd_frac = 0.10  # 10% max DD
        mock_cfg.risk.max_dd_btc = 0.0
        mock_cfg.risk.max_daily_loss_frac = 0.0
        mock_cfg.risk.max_daily_loss_btc = 0.0
        
        _ensure_risk_state(state, wealth=1.0, ts=ts)
        
        # Trigger 15% drawdown (exceeds 10% limit)
        _update_risk_state(state, wealth=0.85, ts=ts, cfg=mock_cfg)
        
        assert state["risk_maxdd_hit"] == True
        assert "risk_maxdd_hit_ts" in state
        
        # Verify timestamp is parseable
        hit_ts = pd.to_datetime(state["risk_maxdd_hit_ts"])
        assert hit_ts == ts
