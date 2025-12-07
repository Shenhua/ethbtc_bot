#!/usr/bin/env python3
"""
tools/multi_interval_opt.py - Multi-Interval Optimization Tool

This tool runs the optimizer across multiple timeframes (e.g. 5m, 15m, 1h)
and compares their performance to find the best interval.
"""
from __future__ import annotations
import argparse, json, os
import pandas as pd
import numpy as np
import sys

# --- MAGIC PATH FIX ---
# Allow importing 'core' even if running from tools/ folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ----------------------

from core.ethbtc_accum_bot import (
    load_vision_csv, FeeParams, _write_excel
)
# Note: Optimizer is no longer in ethbtc_accum_bot.py (moved to optimizer_cli logic)
# This script seems to rely on an older 'Optimizer' class structure which might be missing.
# We will document it as-is, but it likely needs refactoring if 'Optimizer' is gone.
# Assuming 'Optimizer' is a legacy class not present in current core.
# We will import OptunaOptimizer if possible or leave placeholders.

try:
    from tools.optuna_opt import OptunaOptimizer # Hypothetical replacement
except ImportError:
    pass

def compute_scores(df: pd.DataFrame, lam_turns: float = 1.0, gap_penalty: float = 0.25,
                   turns_scale: float = 1000.0, lam_fees: float = 1.0, lam_turnover: float = 0.0) -> pd.DataFrame:
    """
    Computes a robustness score for optimization results.
    """
    df = df.copy()
    if {"train_final_btc","test_final_btc"}.issubset(df.columns):
        df["gen_gap"] = df["train_final_btc"] - df["test_final_btc"]
    else:
        df["gen_gap"] = 0.0
    df["turns_test"] = df.get("turns_test", 0.0).astype(float)
    fees = df.get("fees_btc", 0.0).astype(float)
    tnov = df.get("turnover_btc", 0.0).astype(float)

    df["robust_score"] = (
        df["test_final_btc"].astype(float)
        - lam_turns * (df["turns_test"] / float(turns_scale))
        - gap_penalty * np.maximum(0.0, df["gen_gap"].astype(float))
        - lam_fees * fees
        - lam_turnover * tnov
    )
    return df

def pick_best(scored: pd.DataFrame, top_quantile: float = 0.95) -> pd.Series:
    """
    Picks the best parameter set from scored results.
    """
    thr = scored["test_final_btc"].quantile(top_quantile)
    pool = scored[scored["test_final_btc"] >= thr].copy()
    if pool.empty:
        pool = scored.copy()
    pool = pool.sort_values(["robust_score","turns_test","test_final_btc"], ascending=[False,True,False])
    return pool.iloc[0]

def main():
    """
    Main execution loop.
    Note: This script appears to depend on a legacy 'Optimizer' class that may not exist
    in the current codebase structure (refactored to optuna). It serves as a reference.
    """
    ap = argparse.ArgumentParser(description="Run optimizer across multiple intervals and compare")
    ap.add_argument("--data", nargs="+", required=True, help="List of ETHBTC CSVs (e.g., 5m 15m 30m 1h)")
    ap.add_argument("--bnb-data", help="Path to BNB/BTC CSV", default=None)
    ap.add_argument("--train-start", required=True)
    ap.add_argument("--train-end", required=True)
    ap.add_argument("--test-start", required=True)
    ap.add_argument("--test-end", required=True)
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--maker-fee", type=float, default=0.0002)
    ap.add_argument("--taker-fee", type=float, default=0.0004)
    ap.add_argument("--slippage-bps", type=float, default=1.0)
    ap.add_argument("--bnb-discount", type=float, default=0.25)
    ap.add_argument("--no-bnb", action="store_true")
    ap.add_argument("--excel-out", default="multi_interval_summary.xlsx")
    ap.add_argument("--lambda-turns", type=float, default=1.0)
    ap.add_argument("--gap-penalty", type=float, default=0.25)
    ap.add_argument("--turns-scale", type=float, default=1000.0)
    ap.add_argument("--lambda-fees", type=float, default=1.0)
    ap.add_argument("--lambda-turnover", type=float, default=0.0)
    args = ap.parse_args()

    fee = FeeParams(maker_fee=args.maker_fee, taker_fee=args.taker_fee,
                    slippage_bps=args.slippage_bps, bnb_discount=args.bnb_discount,
                    pay_fees_in_bnb=not args.no_bnb)

    sheets = {}
    summary_rows = []

    # Placeholder for unavailable Optimizer class
    print("WARNING: 'Optimizer' class not found in core. Script may fail.")

    for path in args.data:
        label = os.path.splitext(os.path.basename(path))[0]
        df = load_vision_csv(path)
        close = df["close"]

        # Align BNB series if provided
        bnb_series = None
        if args.bnb_data:
            df_bnb = load_vision_csv(args.bnb_data)
            bnb_series = df_bnb["close"].reindex(close.index, method="ffill")

        # opt = Optimizer(close, fee, bnb_px=bnb_series)
        # res = opt.walk_forward(args.train_start, args.train_end, args.test_start, args.test_end, n_random=args.n_random)
        # ... (Rest of logic disabled due to missing class)

    # ... (Summary logic disabled)
    print("This script requires refactoring to use 'optimizer_cli.py' or 'optuna_opt.py' logic.")

if __name__ == "__main__":
    main()
