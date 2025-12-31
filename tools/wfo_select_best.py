#!/usr/bin/env python3
"""
Smart WFO Window Selection

Supports multiple selection strategies:
1. best_oos - Pick best OOS performance
2. weighted - Balanced OOS + consistency + recency
3. consistent - Prefer consistency over raw performance
4. recent - Recent + good performance
5. ensemble - Average params from top N windows (NEW)
"""

import pandas as pd
import numpy as np
import argparse
import json
import sys
from collections import Counter


def ensemble_average_params(top_windows: pd.DataFrame, n_top: int = 5) -> dict:
    """
    Average numeric parameters across top N windows.
    Use mode (most common) for categorical parameters.
    
    Args:
        top_windows: DataFrame sorted by score descending
        n_top: Number of top windows to average
        
    Returns:
        Averaged parameter dict
    """
    windows = top_windows.head(n_top)
    
    # Collect all params
    all_params = []
    for _, row in windows.iterrows():
        params = row["params"]
        if isinstance(params, str):
            params = json.loads(params)
        all_params.append(params)
    
    if not all_params:
        return {}
    
    # Get all param keys
    all_keys = set()
    for p in all_params:
        all_keys.update(p.keys())
    
    averaged = {}
    for key in all_keys:
        values = [p.get(key) for p in all_params if key in p]
        if not values:
            continue
            
        # Determine type from first value
        first_val = values[0]
        
        if isinstance(first_val, bool):
            # Boolean: use mode (majority vote)
            averaged[key] = Counter(values).most_common(1)[0][0]
            
        elif isinstance(first_val, (int, float)):
            # Numeric: take mean
            avg = np.mean([v for v in values if isinstance(v, (int, float))])
            # Preserve int type if all were ints
            if all(isinstance(v, int) for v in values):
                averaged[key] = int(round(avg))
            else:
                averaged[key] = round(float(avg), 6)
                
        elif isinstance(first_val, str):
            # Categorical: use mode
            averaged[key] = Counter(values).most_common(1)[0][0]
            
        else:
            # Unknown type: take first
            averaged[key] = first_val
    
    return averaged


def main():
    ap = argparse.ArgumentParser(description="Smart WFO window selection")
    ap.add_argument("--wfo-csv", required=True, help="WFO results CSV file")
    ap.add_argument("--out", required=True, help="Output JSON config file")
    ap.add_argument("--strategy", default="weighted", 
                    choices=["best_oos", "weighted", "consistent", "recent", "ensemble", "stable", "stable_ensemble"],
                    help="Selection strategy")
    ap.add_argument("--ensemble-n", type=int, default=5,
                    help="Number of top windows to average for ensemble strategy")
    ap.add_argument("--stability-lambda", type=float, default=0.1,
                    help="Stability penalty weight for 'stable' strategy (default: 0.1)")
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
    
    # Calculate stability penalty per window
    # For each numeric param, compute z-score of each window's value
    # Penalty = mean(abs(z-scores)) across all numeric params
    all_params_list = df["params"].tolist()
    
    # Find all numeric param keys
    numeric_keys = set()
    for p in all_params_list:
        if isinstance(p, dict):
            for k, v in p.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    numeric_keys.add(k)
    
    # Calculate global mean and std for each numeric param
    param_stats = {}
    for key in numeric_keys:
        values = [p.get(key) for p in all_params_list if isinstance(p, dict) and key in p and isinstance(p[key], (int, float))]
        if len(values) > 1:
            param_stats[key] = {"mean": np.mean(values), "std": np.std(values) + 1e-9}
    
    # Calculate per-window stability penalty
    stability_penalties = []
    for p in all_params_list:
        if not isinstance(p, dict):
            stability_penalties.append(0)
            continue
        z_scores = []
        for key, stats in param_stats.items():
            if key in p and isinstance(p[key], (int, float)):
                z = abs(p[key] - stats["mean"]) / stats["std"]
                z_scores.append(z)
        # Average z-score (lower = more stable/typical)
        stability_penalties.append(np.mean(z_scores) if z_scores else 0)
    
    df["stability_penalty"] = stability_penalties
    
    # Calculate base score for all strategies
    df["score"] = (
        df["oos_profit"] * 0.6
        + (df["oos_profit"] + df["train_profit"]) / 2 * 0.3
        + df["recency_weight"] / df["recency_weight"].max() * 0.1
        - df["train_test_gap"] * 0.2
    )
    
    # Strategy-specific scoring
    if args.strategy == "best_oos":
        df["score"] = df["oos_profit"]
        
    elif args.strategy == "consistent":
        df["harmonic_mean"] = 2 * df["oos_profit"] * df["train_profit"] / (df["oos_profit"] + df["train_profit"] + 1e-9)
        df["score"] = df["harmonic_mean"] - df["train_test_gap"] * 0.5
        
    elif args.strategy == "recent":
        df["score"] = df["oos_profit"] * df["recency_weight"] / df["recency_weight"].max()
        
    elif args.strategy in ["stable", "stable_ensemble"]:
        # Combine weighted score with stability penalty
        # Windows with params close to global mean are favored
        df["score"] = (
            df["oos_profit"] * 0.6
            + (df["oos_profit"] + df["train_profit"]) / 2 * 0.3
            + df["recency_weight"] / df["recency_weight"].max() * 0.1
            - df["train_test_gap"] * 0.2
            - df["stability_penalty"] * args.stability_lambda
        )
    
    # Filter suspicious if possible
    candidates = df[~df["suspicious"]].copy()
    if candidates.empty:
        print("⚠️  All windows are suspicious - using all")
        candidates = df.copy()
    
    # Sort by score
    candidates = candidates.sort_values("score", ascending=False)
    
    # Handle ensemble strategy differently
    if args.strategy in ["ensemble", "stable_ensemble"]:
        n_top = min(args.ensemble_n, len(candidates))
        params = ensemble_average_params(candidates, n_top)
        
        print(f"\n{'='*60}")
        if args.strategy == "stable_ensemble":
            print(f"STABLE ENSEMBLE AVERAGING (Top {n_top} stable windows)")
        else:
            print(f"ENSEMBLE PARAMETER AVERAGING (Top {n_top} windows)")
        print(f"{'='*60}")
        print(f"Averaging parameters from top {n_top} performing windows:")
        for i, (_, row) in enumerate(candidates.head(n_top).iterrows()):
            sus = "⚠️" if row['suspicious'] else "✅"
            print(f"  {i+1}. {row['window_end']}: OOS={row['oos_profit']:.4f} Score={row['score']:.4f} {sus}")
        print()
        print("Averaged Parameters:")
        for key, val in sorted(params.items()):
            print(f"  {key}: {val}")
        print(f"{'='*60}\n")
        
    else:
        # Standard single-window selection
        best = candidates.iloc[0]
        
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
        
        params = best["params"]
        if isinstance(params, str):
            params = json.loads(params)
    
    # Post-processing: Cleanup redundant config
    if "exchange_type" in params:
        del params["exchange_type"]
    if "execution" in params and isinstance(params["execution"], dict):
        params["execution"].pop("exchange_type", None)

    # Save params
    with open(args.out, "w") as f:
        json.dump(params, f, indent=2)
    
    print(f"✅ Saved to: {args.out}")

if __name__ == "__main__":
    main()
