#!/usr/bin/env python3
import sys, os
import pandas as pd
import numpy as np

# Path fix
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.meta_strategy import MetaStrategy
from core.ethbtc_accum_bot import StratParams
from core.trend_strategy import TrendParams
from core.regime import get_regime_score

def test_meta_logic():
    print("--- Starting Meta-Strategy Lab Test ---")
    
    # 1. Create Fake Data (1000 bars)
    # We create a sine wave (Ranging) that turns into a linear trend (Trending)
    x = np.linspace(0, 100, 1000)
    price = 100 + 10 * np.sin(x)  # Range
    
    # Add a strong trend at the end
    for i in range(500, 1000):
        price[i] = price[i-1] * 1.001 # Exponential trend
        
    df = pd.DataFrame({
        "open": price, "high": price*1.001, "low": price*0.999, "close": price
    })
    df.index = pd.date_range("2024-01-01", periods=1000, freq="15min")
    
    print(f"Generated {len(df)} candles of synthetic data.")

    # 2. Initialize Strategy
    # MR buys dips, Trend buys breakouts
    mr_p = StratParams(trend_lookback=50, flip_band_entry=0.01)
    tr_p = TrendParams(fast_period=10, slow_period=50)
    
    # Threshold 25: Below 25 = MR, Above 25 = Trend
    meta = MetaStrategy(mr_p, tr_p, adx_threshold=25.0)
    
    # 3. Run Logic
    print("Calculating signals...")
    results = meta.generate_positions(df)
    
    # 4. Analyze Results
    print("\n--- Results Analysis ---")
    print(results.tail())
    
    # Check if Regime Score actually calculated
    avg_score = results["regime_score"].mean()
    max_score = results["regime_score"].max()
    print(f"\nAvg Regime Score: {avg_score:.2f}")
    print(f"Max Regime Score: {max_score:.2f}")
    
    if max_score < 5:
        print("❌ FAILURE: Regime score is too low. ADX calculation might be broken.")
    else:
        print("✅ SUCCESS: Regime score detected trends.")

    # Check Switching Logic
    # Find a row where Score > 25. Target should match 'sig_trend'.
    # Find a row where Score < 25. Target should match 'sig_mr'.
    
    high_adx = results[results["regime_score"] > 25]
    if not high_adx.empty:
        match = (high_adx["target_w"] == high_adx["sig_trend"]).all()
        print(f"High ADX Logic: {'✅ Matched Trend' if match else '❌ FAILED'}")
    else:
        print("⚠️ No High ADX periods found in synthetic data.")

    low_adx = results[results["regime_score"] < 25]
    if not low_adx.empty:
        match = (low_adx["target_w"] == low_adx["sig_mr"]).all()
        print(f"Low ADX Logic:  {'✅ Matched MeanRev' if match else '❌ FAILED'}")
        
if __name__ == "__main__":
    test_meta_logic()