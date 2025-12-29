import pytest
import pandas as pd
import json
import os
import numpy as np
import io
from core.strategy_factory import build_strategy
from core.config_schema import load_config

def test_golden_regression_parity():
    """Verify that current strategy logic matches the golden snapshot."""
    snapshot_path = "tests/golden_snapshot.json"
    if not os.path.exists(snapshot_path):
        pytest.skip("Golden snapshot not found. Run tools/capture_snapshot.py first.")
        
    with open(snapshot_path, "r") as f:
        snapshot = json.load(f)
        
    cfg = load_config(snapshot["config_path"])
    strat, _ = build_strategy(cfg)
    
    # Load data from snapshot
    df = pd.read_json(io.StringIO(snapshot["input_data"]))
    # Ensure index is datetime and UTC (matching capture_snapshot)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    
    current_signals = strat.generate_positions(df)
    golden_signals = pd.read_json(io.StringIO(snapshot["signals"]))
    
    # Compare target weights (cast to float64 for parity)
    pd.testing.assert_series_equal(
        current_signals["target_w"].astype(float), 
        golden_signals["target_w"].astype(float),
        obj="Strategy Target Weights"
    )
    
    print("✅ Regression check passed: Parity with golden snapshot confirmed.")
