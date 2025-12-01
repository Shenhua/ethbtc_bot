#!/usr/bin/env python3
import sys
import os

# --- MAGIC PATH FIX ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ----------------------

import argparse, json, logging
import pandas as pd
import numpy as np
from core.ethbtc_accum_bot import (
    load_vision_csv, FeeParams, StratParams, Backtester
)
from core.trend_strategy import TrendParams
from core.meta_strategy import MetaStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s [META-OPT] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger("meta_opt")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--funding-data")
    ap.add_argument("--mr-config", required=True, help="Path to best Mean Reversion config")
    ap.add_argument("--trend-config", required=True, help="Path to best Trend config")
    ap.add_argument("--out", default="results/opt_meta.csv")
    args = ap.parse_args()

    # 1. Load Data
    log.info("Loading data...")
    df = load_vision_csv(args.data)
    
    funding_series = None
    if args.funding_data:
        f_df = pd.read_csv(args.funding_data)
        f_df["time"] = pd.to_datetime(f_df["time"], format="mixed", utc=True)
        f_df = f_df.set_index("time").sort_index()
        funding_series = f_df["rate"]
        # Align
        funding_aligned = funding_series.reindex(df.index).ffill().fillna(0.0)
    else:
        funding_aligned = None

    # 2. Load Base Configs
    with open(args.mr_config) as f: mr_json = json.load(f)
    with open(args.trend_config) as f: tr_json = json.load(f)
    
    # Flatten if needed (handle 'strategy' key)
    mr_dict = mr_json.get("strategy", mr_json)
    tr_dict = tr_json.get("strategy", tr_json)

    mr_p = StratParams(
        trend_kind=mr_dict.get("trend_kind", "sma"),
        trend_lookback=int(mr_dict.get("trend_lookback", 200)),
        flip_band_entry=float(mr_dict.get("flip_band_entry", 0.025)),
        flip_band_exit=float(mr_dict.get("flip_band_exit", 0.015)),
        vol_window=int(mr_dict.get("vol_window", 60)),
        vol_adapt_k=float(mr_dict.get("vol_adapt_k", 0.0)),
        funding_limit_long=float(mr_dict.get("funding_limit_long", 0.05)),
        funding_limit_short=float(mr_dict.get("funding_limit_short", -0.05))
    )
    
    tr_p = TrendParams(
        fast_period=int(tr_dict.get("fast_period", 50)),
        slow_period=int(tr_dict.get("slow_period", 200)),
        ma_type=tr_dict.get("ma_type", "ema"),
        funding_limit_long=float(tr_dict.get("funding_limit_long", 0.05)),
        funding_limit_short=float(tr_dict.get("funding_limit_short", -0.05))
    )
    
    fee = FeeParams() # Default fees

    # 3. Grid Search ADX Thresholds
    results = []
    thresholds = [15, 20, 25, 30, 35, 40, 45, 50]
    
    log.info("Testing Meta-Strategy across ADX thresholds...")
    
    for thresh in thresholds:
        strat = MetaStrategy(mr_p, tr_p, adx_threshold=float(thresh))
        bt = Backtester(fee)
        
        # Run Simulation
        res = bt.simulate(df["close"], strat, funding_series=funding_aligned, full_df=df)
        
        summ = res["summary"]
        res_row = {
            "adx_threshold": thresh,
            "final_btc": summ["final_btc"],
            "drawdown": summ["max_drawdown_pct"],
            "trades": summ["n_trades"],
            "fees": summ["fees_btc"]
        }
        results.append(res_row)
        log.info(f"ADX {thresh}: Profit={summ['final_btc']:.4f} DD={summ['max_drawdown_pct']:.2%}")

    # 4. Save
    pd.DataFrame(results).sort_values("final_btc", ascending=False).to_csv(args.out, index=False)
    log.info(f"Saved optimization results to {args.out}")

if __name__ == "__main__":
    main()