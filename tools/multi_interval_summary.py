#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
import os, sys
import pandas as pd
import numpy as np

# --- MAGIC PATH FIX ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ----------------------

def load_runs(label: str, path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["__interval"] = label
    return df

def summarize_interval(df: pd.DataFrame, top_quantile: float = 0.2) -> dict:
    if "robust_score" not in df.columns:
        raise ValueError("Expected 'robust_score' in Optuna CSV.")
    
    # Keep only completed / valid trials
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["robust_score"])
    
    # Select top quantile
    q = df["robust_score"].quantile(1.0 - top_quantile)
    elite = df[df["robust_score"] >= q]
    if elite.empty:
        elite = df  # fall back to all if weird
    
    def _m(col, default=np.nan):
        return float(elite[col].mean()) if col in elite.columns else default
    
    return {
        "n_trials": len(df),
        "n_elite": len(elite),
        "mean_robust": _m("robust_score"),
        "mean_test_final_btc": _m("test_final_btc"),
        "mean_train_final_btc": _m("train_final_btc"),
        "mean_turns_test": _m("turns_test"),
        "mean_fees_btc": _m("fees_btc"),
        "mean_turnover_btc": _m("turnover_btc"),
    }

def main():
    ap = argparse.ArgumentParser(
        description="Compare Optuna runs for multiple intervals (15m, 30m, 1h, ...)."
    )
    ap.add_argument(
        "--run",
        nargs=2,
        action="append",
        metavar=("LABEL", "CSV"),
        help="Pair: interval label (e.g. 15m) and path to Optuna CSV.",
        required=True,
    )
    ap.add_argument("--top-quantile", type=float, default=0.2,
                    help="Fraction of best trials per interval to average (default: 0.2).")
    ap.add_argument("--out-json", help="Optional JSON file to save summary.")
    
    args = ap.parse_args()
    
    rows = []
    for label, path in args.run:
        df = load_runs(label, path)
        summ = summarize_interval(df, top_quantile=args.top_quantile)
        summ["interval"] = label
        rows.append(summ)
    
    summary_df = pd.DataFrame(rows).set_index("interval")
    print("=== Multi-Interval Summary ===")
    print(summary_df.to_string(float_format=lambda x: f"{x:.4f}"))
    
    if args.out_json:
        summary_df.to_json(args.out_json, orient="index", indent=2)
        print(f"\nSaved JSON summary to {args.out_json}")

if __name__ == "__main__":
    main()