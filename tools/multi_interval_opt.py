
from __future__ import annotations
import argparse, json, os
import pandas as pd
import numpy as np

from core.ethbtc_accum_bot import (
    load_vision_csv, FeeParams, StratParams, EthBtcStrategy, Backtester, Optimizer, _write_excel
)

def compute_scores(df: pd.DataFrame, lam_turns: float = 1.0, gap_penalty: float = 0.25,
                   turns_scale: float = 1000.0, lam_fees: float = 1.0, lam_turnover: float = 0.0) -> pd.DataFrame:
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
    thr = scored["test_final_btc"].quantile(top_quantile)
    pool = scored[scored["test_final_btc"] >= thr].copy()
    if pool.empty:
        pool = scored.copy()
    pool = pool.sort_values(["robust_score","turns_test","test_final_btc"], ascending=[False,True,False])
    return pool.iloc[0]

def main():
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
    for path in args.data:
        label = os.path.splitext(os.path.basename(path))[0]
        df = load_vision_csv(path)
        close = df["close"]

        # Align BNB series if provided
        bnb_series = None
        if args.bnb_data:
            df_bnb = load_vision_csv(args.bnb_data)
            bnb_series = df_bnb["close"].reindex(close.index, method="ffill")

        opt = Optimizer(close, fee, bnb_px=bnb_series)
        res = opt.walk_forward(args.train_start, args.train_end, args.test_start, args.test_end, n_random=args.n_random)
        scored = compute_scores(res, args.lambda_turns, args.gap_penalty, args.turns_scale,
                                args.lambda_fees, args.lambda_turnover)
        best = pick_best(scored)
        sheets[label] = scored
        summary_rows.append({
            "interval": label,
            "test_final_btc": best.get("test_final_btc"),
            "train_final_btc": best.get("train_final_btc"),
            "gen_gap": (best.get("train_final_btc") - best.get("test_final_btc")) if (best.get("train_final_btc") is not None and best.get("test_final_btc") is not None) else None,
            "turns_test": best.get("turns_test"),
            "fees_btc": best.get("fees_btc"),
            "turnover_btc": best.get("turnover_btc"),
            "robust_score": best.get("robust_score"),
            "params": json.dumps(best.get("params", {})),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(["robust_score","turns_test","test_final_btc"], ascending=[False,True,False])
    sheets["Summary"] = summary_df
    _write_excel(args.excel_out, sheets)
    print(f"Wrote multi-interval comparison → {args.excel_out}")

if __name__ == "__main__":
    main()
