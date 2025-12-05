#!/usr/bin/env python3
"""
Smart WFO Window Selection

Instead of picking the last window, selects the best window based on:
1. OOS performance (60% weight)
2. Train/Test consistency (30% weight)
3. Recency (10% weight)
"""

import pandas as pd
import numpy as np
import argparse
import json
import sys

def main():
    ap = argparse.ArgumentParser(description="Smart WFO window selection")
    ap.add_argument("--wfo-csv", required=True, help="WFO results CSV file")
    ap.add_argument("--out", required=True, help="Output JSON config file")
    ap.add_argument("--strategy", default="weighted", 
                    choices=["best_oos", "weighted", "consistent", "recent"],
                    help="Selection strategy")
    args = ap.parse_args()
    
    # Load WFO results
    df = pd.read_csv(args.wfo_csv)
    
    if df.empty:
        print("ERROR: No WFO results found!", file=sys.stderr)
        sys.exit(1)
    
    # Parse best_params JSON
    df["params"] = df["best_params"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    
    # Calculate metrics
    df["train_test_ratio"] = df["oos_profit"] / df["train_profit"].replace(0, 1e-9)
    df["train_test_gap"] = abs(df["oos_profit"] - df["train_profit"])
    df["suspicious"] = (df["train_test_ratio"] > 1.5) | (df["train_test_ratio"] < 0.7)
    
    # Recency weight (exponential decay)
    n = len(df)
    df["recency_weight"] = np.exp(np.linspace(0, 1, n))
    
    # Selection strategy
    if args.strategy == "best_oos":
        # Simply pick window with best OOS performance
        df["score"] = df["oos_profit"]
        
    elif args.strategy == "weighted":
        # Balanced: OOS + consistency + recency
        df["score"] = (
            df["oos_profit"] * 0.6
            + (df["oos_profit"] + df["train_profit"]) / 2 * 0.3
            + df["recency_weight"] / df["recency_weight"].max() * 0.1
            - df["train_test_gap"] * 0.2
        )
        
    elif args.strategy == "consistent":
        # Prefer consistency over raw performance
        df["harmonic_mean"] = 2 * df["oos_profit"] * df["train_profit"] / (df["oos_profit"] + df["train_profit"] + 1e-9)
        df["score"] = df["harmonic_mean"] - df["train_test_gap"] * 0.5
        
    elif args.strategy == "recent":
        # Recent + good performance
        df["score"] = df["oos_profit"] * df["recency_weight"] / df["recency_weight"].max()
    
    # Filter suspicious if possible
    candidates = df[~df["suspicious"]].copy()
    if candidates.empty:
        print("⚠️  All windows are suspicious - using all")
        candidates = df.copy()
    
    # Select best
    best = candidates.sort_values("score", ascending=False).iloc[0]
    
    # Print selection
    print(f"\n{'='*60}")
    print(f"SMART WFO SELECTION (Strategy: {args.strategy})")
    print(f"{'='*60}")
    print(f"Selected Window: {best['window_end']}")
    print(f"OOS Profit: {best['oos_profit']:.4f}")
    print(f"Train Profit: {best['train_profit']:.4f}")
    print(f"Consistency Ratio: {best['train_test_ratio']:.2f}")
    print(f"Score: {best['score']:.4f}")
    if best['suspicious']:
        print(f"⚠️  WARNING: This window is flagged as suspicious!")
    print(f"{'='*60}\n")
    
    # Show top 5 for comparison
    print("Top 5 Windows:")
    print("-" * 60)
    for idx, row in candidates.head(5).iterrows():
        sus = "⚠️" if row['suspicious'] else "✅"
        print(f"  {row['window_end']}: OOS={row['oos_profit']:.4f} "
              f"Train={row['train_profit']:.4f} Ratio={row['train_test_ratio']:.2f} "
              f"Score={row['score']:.4f} {sus}")
    print()
    
    # Save params
    params = best["params"]
    if isinstance(params, str):
        params = json.loads(params)
        
    with open(args.out, "w") as f:
        json.dump(params, f, indent=2)
    
    print(f"✅ Saved to: {args.out}")

if __name__ == "__main__":
    main()
