# capture_snapshot.py
import json
import pandas as pd
import numpy as np
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from core.strategy_factory import build_strategy
from core.config_schema import load_config

def capture():
    # Load a standard config
    cfg_path = "configs/prod_meta_live.json"
    if not os.path.exists(cfg_path):
        print(f"Error: {cfg_path} not found")
        return
        
    cfg = load_config(cfg_path)
    
    # Create deterministic mock data
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "open": np.linspace(20000, 21000, 100),
        "high": np.linspace(20100, 21100, 100),
        "low": np.linspace(19900, 20900, 100),
        "close": np.linspace(20050, 21050, 100),
        "volume": np.random.rand(100) * 100
    }, index=dates)
    
    # Build strategy
    strat, _ = build_strategy(cfg)
    
    # Generate signals
    signals = strat.generate_positions(df)
    
    # Save to snapshot
    snapshot = {
        "config_path": cfg_path,
        "input_data": df.to_json(date_format='iso'),
        "signals": signals.to_json(date_format='iso')
    }
    
    os.makedirs("tests", exist_ok=True)
    with open("tests/golden_snapshot.json", "w") as f:
        json.dump(snapshot, f, indent=2)
    print("Golden snapshot captured to tests/golden_snapshot.json")

if __name__ == "__main__":
    capture()
