#!/usr/bin/env python3
from __future__ import annotations

import sys, os, argparse
import pandas as pd
import numpy as np

# --- MAGIC PATH FIX ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ----------------------



from core.regime import get_regime_score  

def load_ohlc(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, format="mixed")
        df = df.set_index("time")
    df = df.sort_index()
    return df[["open", "high", "low", "close"]]

def compute_regime(df_ohlc: pd.DataFrame) -> pd.Series:
    return get_regime_score(df_ohlc)

def analyze_by_regime(backtest_csv: str, ohlc_csv: str, adx_thresholds=(20, 30)):
    bt = pd.read_csv(backtest_csv, index_col=0, parse_dates=True)
    ohlc = load_ohlc(ohlc_csv)
    ohlc = ohlc.reindex(bt.index).ffill()

    if "regime_score" in bt.columns:
        regime = bt["regime_score"]
    else:
        regime = compute_regime(ohlc)
    
    # Simple 3-bucket split:
    lo, hi = adx_thresholds
    bins = pd.cut(
        regime,
        bins=[-np.inf, lo, hi, np.inf],
        labels=["low", "medium", "high"],
    )

    if "wealth_btc" in bt.columns:
        wealth = bt["wealth_btc"]
    else:
        raise ValueError("Backtest CSV must contain 'wealth_btc' from cmd_backtest --out.")
    
    # Compute returns and max dd per bucket
    results = []
    for label in ["low", "medium", "high"]:
        mask = bins == label
        if not mask.any():
            continue
        sub = wealth[mask]
        if sub.empty:
            continue
        start = float(sub.iloc[0])
        end = float(sub.iloc[-1])
        ret = end / start - 1.0 if start > 0 else 0.0
        running_max = sub.cummax()
        dd = (running_max - sub).max() / running_max.max() if running_max.max() > 0 else 0.0
        results.append({"regime": label, "return": ret, "max_drawdown": dd})

    return pd.DataFrame(results).set_index("regime")

def main():
    ap = argparse.ArgumentParser(description="Analyze backtest performance by ADX regime.")
    ap.add_argument("--backtest-csv", required=True, help="CSV from cmd_backtest --out.")
    ap.add_argument("--ohlc-csv", required=True, help="Original OHLC CSV (vision format).")
    ap.add_argument("--low-threshold", type=float, default=20.0)
    ap.add_argument("--high-threshold", type=float, default=30.0)
    args = ap.parse_args()

    df = analyze_by_regime(
        args.backtest_csv,
        args.ohlc_csv,
        adx_thresholds=(args.low_threshold, args.high_threshold),
    )
    print("=== Performance by ADX Regime ===")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))

if __name__ == "__main__":
    main()