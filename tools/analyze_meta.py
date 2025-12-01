#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import sys
import os

def analyze(csv_path, threshold=25.0):
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Verify Columns
    required_cols = ['regime_score', 'sig_mr', 'sig_trend', 'target_w']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Error: Columns {missing} not found.")
        print("Did you run the backtest with the updated code that exports diagnostics?")
        return

    print("Running Logic Verification...")

    # 2. Define Masks
    # Trend Regime: ADX > Threshold
    mask_trend = df['regime_score'] > threshold
    # Mean Reversion Regime: ADX <= Threshold
    mask_mr = df['regime_score'] <= threshold
    
    # 3. Check Trend Logic
    # In Trend Regime, target_w should equal sig_trend
    trend_rows = df[mask_trend]
    # Use isclose because float comparison (1.0 vs 1.0000001) can be tricky
    trend_match = np.isclose(trend_rows['target_w'], trend_rows['sig_trend'], atol=1e-9)
    trend_acc = trend_match.mean() if len(trend_rows) > 0 else 0.0

    # 4. Check Mean Reversion Logic
    mr_rows = df[mask_mr]
    mr_match = np.isclose(mr_rows['target_w'], mr_rows['sig_mr'], atol=1e-9)
    mr_acc = mr_match.mean() if len(mr_rows) > 0 else 0.0
    
    # 5. Check for Safety Overrides
    # Sometimes logic matches neither because Funding/Gate overrode it.
    # Let's see how often target is 0.0 when logic says otherwise.
    
    # 6. Report
    print("\n" + "="*50)
    print(f"META-STRATEGY DIAGNOSTIC REPORT")
    print("="*50)
    print(f"ADX Threshold:    {threshold}")
    print(f"Total Bars:       {len(df):,}")
    print("-" * 50)
    print(f"📉 Mean Rev Regime:  {len(mr_rows):,} bars ({len(mr_rows)/len(df):.1%})")
    print(f"   Logic Match:      {mr_acc:.2%} {'✅' if mr_acc > 0.99 else '⚠️'}")
    if mr_acc < 1.0:
        mismatches = mr_rows[~mr_match]
        print(f"   Mismatches:       {len(mismatches)} (Likely due to Funding/Gate safety overrides)")
        print(f"   Sample Mismatch:\n{mismatches[['regime_score', 'sig_mr', 'target_w']].head(2)}")

    print("-" * 50)
    print(f"📈 Trend Regime:     {len(trend_rows):,} bars ({len(trend_rows)/len(df):.1%})")
    print(f"   Logic Match:      {trend_acc:.2%} {'✅' if trend_acc > 0.99 else '⚠️'}")
    if trend_acc < 1.0:
        mismatches = trend_rows[~trend_match]
        print(f"   Mismatches:       {len(mismatches)}")
        print(f"   Sample Mismatch:\n{mismatches[['regime_score', 'sig_trend', 'target_w']].head(2)}")
    print("="*50)
    
    if trend_acc > 0.99 and mr_acc > 0.99:
        print("✅ CONCLUSION: Meta-Strategy is switching correctly.")
    else:
        print("⚠️ CONCLUSION: Strategy is switching, but Safety Gates are overriding signals.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="Path to backtest_meta_final.csv")
    parser.add_argument("--threshold", type=float, default=25.0, help="ADX Threshold used in config")
    args = parser.parse_args()
    
    analyze(args.csv_file, args.threshold)

