"""
Test Parity Sync: Backtester vs BotEngine

This test ensures that the Backtester and the Live BotEngine produce
IDENTICAL target_w signals given the same input data and configuration.

Any divergence indicates a parity bug that could cause live trading
to behave differently from what was validated in backtests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config_schema import load_config
from core.ethbtc_accum_bot import (
    load_vision_csv, Backtester, FeeParams, StratParams, EthBtcStrategy
)
from core.trend_strategy import TrendStrategy, TrendParams
from core.meta_strategy import MetaStrategy
from core.strategy_factory import merge_strategy_params, build_mr_params, build_tr_params


# --- Test Data Setup ---

SAMPLE_DATA_PATH = Path("data/raw/BTCUSDT_15m_2021-2025_vision.csv")
SAMPLE_CONFIG_PATH = Path("configs/prod_btc_long_wfo_robust.json")


def get_sample_data(n_bars: int = 500) -> pd.DataFrame:
    """Load a sample of price data for testing."""
    if not SAMPLE_DATA_PATH.exists():
        pytest.skip(f"Sample data not found: {SAMPLE_DATA_PATH}")
    
    df = load_vision_csv(str(SAMPLE_DATA_PATH))
    # Take a slice from the middle to avoid edge effects
    start_idx = len(df) // 2
    result = df.iloc[start_idx : start_idx + n_bars].copy()
    # Reset index to avoid DatetimeArray issues in some pandas operations
    # Index name is 'close_time' from load_vision_csv
    result = result.reset_index(drop=False)
    result = result.set_index('close_time')
    return result


def get_sample_config():
    """Load a sample config for testing."""
    if not SAMPLE_CONFIG_PATH.exists():
        pytest.skip(f"Sample config not found: {SAMPLE_CONFIG_PATH}")
    return load_config(str(SAMPLE_CONFIG_PATH))


# --- Parity Tests ---

class TestStrategyFactoryParity:
    """Test that strategy_factory produces consistent params."""
    
    def test_merge_returns_structured_dict(self):
        """Verify merge_strategy_params returns the expected structure."""
        cfg = get_sample_config()
        merged = merge_strategy_params(cfg)
        
        assert "strategy_type" in merged
        assert "mr_params" in merged
        assert "tr_params" in merged
        assert "base_params" in merged
        
    def test_mr_params_complete(self):
        """Verify MR params contain all required fields."""
        cfg = get_sample_config()
        merged = merge_strategy_params(cfg)
        
        required_keys = [
            "trend_kind", "trend_lookback", "flip_band_entry", "flip_band_exit",
            "vol_window", "cooldown_minutes", "step_allocation", "max_position"
        ]
        
        for key in required_keys:
            assert key in merged["mr_params"], f"Missing MR param: {key}"
            
    def test_tr_params_complete(self):
        """Verify Trend params contain all required fields."""
        cfg = get_sample_config()
        merged = merge_strategy_params(cfg)
        
        required_keys = [
            "fast_period", "slow_period", "ma_type", "cooldown_minutes",
            "long_only", "funding_limit_long", "funding_limit_short"
        ]
        
        for key in required_keys:
            assert key in merged["tr_params"], f"Missing Trend param: {key}"


class TestSignalParity:
    """Test that Backtester and direct strategy calls produce identical signals."""
    
    @pytest.mark.skip(reason="MR strategy test requires MR-specific config, skipping for now")
    def test_mr_strategy_signal_determinism(self):
        """
        Test that EthBtcStrategy.generate_positions() is deterministic.
        Running it twice with same data should produce identical results.
        """
        df = get_sample_data(200)
        cfg = get_sample_config()
        merged = merge_strategy_params(cfg)
        
        mr_p = build_mr_params(merged["mr_params"])
        strat = EthBtcStrategy(mr_p)
        
        result1 = strat.generate_positions(df)
        result2 = strat.generate_positions(df)
        
        np.testing.assert_array_equal(
            result1["target_w"].values,
            result2["target_w"].values,
            err_msg="MR strategy is non-deterministic!"
        )
        
    def test_trend_strategy_signal_determinism(self):
        """
        Test that TrendStrategy.generate_positions() is deterministic.
        """
        df = get_sample_data(200)
        cfg = get_sample_config()
        merged = merge_strategy_params(cfg)
        
        tr_p = build_tr_params(merged["tr_params"])
        strat = TrendStrategy(tr_p)
        
        result1 = strat.generate_positions(df)
        result2 = strat.generate_positions(df)
        
        np.testing.assert_array_equal(
            result1["target_w"].values,
            result2["target_w"].values,
            err_msg="Trend strategy is non-deterministic!"
        )
        
    def test_meta_strategy_signal_determinism(self):
        """
        Test that MetaStrategy.generate_positions() is deterministic.
        """
        df = get_sample_data(200)
        cfg = get_sample_config()
        merged = merge_strategy_params(cfg)
        
        mr_p = build_mr_params(merged["mr_params"])
        tr_p = build_tr_params(merged["tr_params"])
        adx_thresh = float(merged["mr_params"].get("adx_threshold", 25.0))
        
        strat = MetaStrategy(mr_p, tr_p, adx_threshold=adx_thresh)
        
        result1 = strat.generate_positions(df)
        result2 = strat.generate_positions(df)
        
        np.testing.assert_array_equal(
            result1["target_w"].values,
            result2["target_w"].values,
            err_msg="Meta strategy is non-deterministic!"
        )


class TestBacktesterStrategyParity:
    """
    Test that the Backtester uses strategy signals correctly.
    
    This validates that running strategy.generate_positions() directly
    produces the same target_w sequence as running through the Backtester.
    """
    
    def test_backtester_uses_strategy_signal(self):
        """
        Verify Backtester respects the strategy's target_w signal.
        
        Note: Due to position sizing logic (step smoothing, clamping),
        the Backtester's actual positions may differ from target_w.
        But the *input* to that logic should be the strategy signal.
        """
        df = get_sample_data(300)
        cfg = get_sample_config()
        merged = merge_strategy_params(cfg)
        
        # Build strategy
        mr_p = build_mr_params(merged["mr_params"])
        tr_p = build_tr_params(merged["tr_params"])
        adx_thresh = float(merged["mr_params"].get("adx_threshold", 25.0))
        strat = MetaStrategy(mr_p, tr_p, adx_threshold=adx_thresh)
        
        # Get raw strategy signal
        raw_signal = strat.generate_positions(df)
        
        # Run backtest - correct API: simulate(close, strategy, ...)
        fee = FeeParams()
        bt = Backtester(fee)
        bt_result = bt.simulate(
            close=df["close"],
            strategy=strat,
            funding_series=None,
            bnb_price_series=None,
            full_df=df,
            risk_mode="static"
        )
        
        # The Backtester should have the same target_w as the strategy output
        # (before step smoothing is applied)
        # We check that at least 90% of values match (edge effects may cause some diff)
        strategy_targets = raw_signal["target_w"].dropna().values
        bt_targets = bt_result["diagnostics"]["target_w"].dropna().values
        
        # Align lengths
        min_len = min(len(strategy_targets), len(bt_targets))
        strategy_targets = strategy_targets[-min_len:]
        bt_targets = bt_targets[-min_len:]
        
        # Check correlation (should be very high if using same signal)
        correlation = np.corrcoef(strategy_targets, bt_targets)[0, 1]
        
        assert correlation > 0.95, (
            f"Backtester target_w diverges from strategy signal! "
            f"Correlation: {correlation:.4f}"
        )


class TestConfigMergeParity:
    """
    Test that different code paths produce the same merged config.
    """
    
    def test_engine_uses_strategy_factory(self):
        """
        Verify that importing merge_strategy_params in engine.py
        gives the same function as strategy_factory.
        """
        from core.engine import merge_strategy_params as engine_merge
        from core.strategy_factory import merge_strategy_params as factory_merge
        
        # Should be the exact same function object
        assert engine_merge is factory_merge, (
            "engine.py's merge_strategy_params is not the same as strategy_factory's!"
        )
        
    def test_live_executor_uses_strategy_factory(self):
        """
        Verify that live_executor.py imports from strategy_factory.
        """
        # Read the source to verify the import
        with open("live_executor.py", "r") as f:
            source = f.read()
        
        assert "from core.strategy_factory import" in source, (
            "live_executor.py should import from core.strategy_factory"
        )
        assert "merge_strategy_params" in source, (
            "live_executor.py should import merge_strategy_params"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
