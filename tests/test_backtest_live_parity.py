"""
Tests to ensure parity between Backtest and Live Executor logic.
Specifically verifies that:
1. Config overloading works identically
2. Strategy construction produces identical objects
3. Signal generation is identical given same data
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.config_schema import AppConfig, Strategy, Execution, Risk, Fees
from core.strategy_factory import merge_strategy_params, build_strategy

# --- FIXTURES ---

@pytest.fixture
def mock_meta_config():
    """Create a mock AppConfig with Meta Strategy overrides."""
    return AppConfig(
        strategy=Strategy(
            strategy_type="meta",
            trend_kind="sma",  # Base
            trend_lookback=100, # Base
            long_only=True,     # Base
            mean_reversion_overrides={
                "long_only": 0,           # Override: Allow shorts
                "trend_lookback": 50,     # Override: Faster lookback
                "step_allocation": 0.33,  # Override: Smaller steps
                "flip_band_entry": 0.03,  # Override specific param
            },
            trend_overrides={
                "long_only": False,       # Override: Allow shorts
                "fast_period": 20,        # Override
            }
        ),
        execution=Execution(),
        risk=Risk(),
        fees=Fees(maker_fee=0.0002, taker_fee=0.0004)
    )

# ... (skip sample_ohlc_data) ...

# --- TEST 1: Config Merging Logic ---

# ... (skip test_merge_strategy_params_meta) ...

def test_merge_strategy_params_non_meta():
    """Test that non-meta config returns base params in mr/tr slots."""
    cfg = AppConfig(
        strategy=Strategy(strategy_type="mean_reversion", trend_lookback=999),
        execution=Execution(),
        risk=Risk(),
        fees=Fees(maker_fee=0.0002, taker_fee=0.0004)
    )
    merged = merge_strategy_params(cfg)
    
    # mr_params should be copy of base
    assert merged["mr_params"]["trend_lookback"] == 999
    # tr_params should also be copy of base (fallback)
    assert merged["tr_params"]["trend_lookback"] == 999 

# ... (skip tests 2 & 3) ...

def test_live_executor_parity_check():
    """
    Simulate what live_executor does: extracting params from merged dict
    and verify it matches what the strategy object thinks.
    """
    conf = AppConfig(
        strategy=Strategy(
            strategy_type="meta",
            mean_reversion_overrides={"flip_band_entry": 0.05}
        ),
        execution=Execution(),
        risk=Risk(),
        fees=Fees(maker_fee=0.0002, taker_fee=0.0004)
    )
    
    # 1. Live Executor Flow: Merge config, get params dict
    merged = merge_strategy_params(conf)
    mr_params = merged["mr_params"]
    live_val = float(mr_params.get("flip_band_entry", 0.0))
    
    # 2. Backtest Flow: Build strategy object
    strat, _ = build_strategy(conf)
    bt_val = strat.mr.p.flip_band_entry
    
    assert live_val == 0.05
    assert bt_val == 0.05
    assert live_val == bt_val
