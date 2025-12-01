#!/usr/bin/env python3
import sys
import os
import argparse
import json
import logging
import pandas as pd


# --- MAGIC PATH FIX ---
# Calculate the root directory (one level up from 'tools')
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
# ----------------------

from core.ethbtc_accum_bot import load_vision_csv, FeeParams, StratParams, Backtester
from core.trend_strategy import TrendParams
from core.meta_strategy import MetaStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s [META-OPT] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger("meta_opt")

def clean_params(params_dict, cls):
    """Filters dict keys to only those accepted by the dataclass"""
    valid_keys = cls.__annotations__.keys()
    return {k: v for k, v in params_dict.items() if k in valid_keys}

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
    
    funding_aligned = None
    if args.funding_data:
        f_df = pd.read_csv(args.funding_data)
        f_df["time"] = pd.to_datetime(f_df["time"], format="mixed", utc=True)
        f_df = f_df.set_index("time").sort_index()
        funding_aligned = f_df["rate"].reindex(df.index).ffill().fillna(0.0)

    # 2. Load and Sanitize Configs
    with open(args.mr_config) as f: mr_raw = json.load(f)
    with open(args.trend_config) as f: tr_raw = json.load(f)
    
    # Handle nested "strategy" keys if present
    mr_dict = mr_raw.get("strategy", mr_raw)
    tr_dict = tr_raw.get("strategy", tr_raw)

    # Clean params to prevent "unexpected keyword" errors
    mr_clean = clean_params(mr_dict, StratParams)
    tr_clean = clean_params(tr_dict, TrendParams)

    # Ensure required defaults if missing
    mr_clean.setdefault("trend_kind", "sma")
    tr_clean.setdefault("ma_type", "ema")

    mr_p = StratParams(**mr_clean)
    tr_p = TrendParams(**tr_clean)
    
    fee = FeeParams() 

    # 3. Grid Search ADX Thresholds
    results = []
    # Test a realistic range for ADX (usually 15-30)
    thresholds = [10, 15, 20, 25, 30, 35, 40]
    
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