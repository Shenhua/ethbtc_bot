#!/usr/bin/env python3
from __future__ import annotations
#!/usr/bin/env python3
import sys, os

# Path Fix
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, argparse, time, random, logging, math
import pandas as pd
import numpy as np
import optuna
from core.ethbtc_accum_bot import (
    load_vision_csv, load_json_config, 
    FeeParams, Backtester
)
from core.trend_strategy import TrendParams, TrendStrategy

# Setup Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [OPT-TREND] %(message)s', 
    datefmt='%H:%M:%S'
)
log = logging.getLogger("optimizer")
optuna.logging.set_verbosity(optuna.logging.INFO)

def suggest_params(trial):
    """
    Search Space for Trend Strategy
    """
    # 1. Moving Average Periods
    # Fast must be faster than Slow. We enforce this structure.
    fast = trial.suggest_int("fast_period", 10, 200, step=10)
    slow = trial.suggest_int("slow_period", 40, 400, step=20)
    
    # 2. Type
    ma_type = trial.suggest_categorical("ma_type", ["ema", "sma"])
    
    # 3. Execution
    cooldown = trial.suggest_categorical("cooldown_minutes", [60, 120, 240, 360])
    
    # 4. Funding Filter
    funding_long = trial.suggest_float("funding_limit_long", 0.01, 0.10)
    funding_short = trial.suggest_float("funding_limit_short", -0.10, -0.01)

    return TrendParams(
        fast_period=fast,
        slow_period=slow,
        ma_type=ma_type,
        cooldown_minutes=cooldown,
        step_allocation=1.0, # Trend goes All-In
        max_position=1.0,
        long_only=True,      # Spot is Long Only
        funding_limit_long=funding_long,
        funding_limit_short=funding_short
    )

class Objective:
    def __init__(self, args, fee, train_close, test_close, bnb_train, bnb_test, funding_train, funding_test):
        self.args = args
        self.fee = fee
        self.train_close = train_close
        self.test_close = test_close
        self.bnb_train = bnb_train
        self.bnb_test = bnb_test
        self.funding_train = funding_train
        self.funding_test = funding_test

    def __call__(self, trial):
        t0 = time.time()
        tid = trial.number
        
        try:
            p = suggest_params(trial)
            
            # Validity Check: Fast must be < Slow
            if p.fast_period >= p.slow_period:
                # Tell Optuna this is invalid without crashing
                raise optuna.TrialPruned("Fast >= Slow")

            # Run Simulation
            bt = Backtester(self.fee)
            
            # We only care about Test performance for this simple search
            # (But proper way is Train/Test split)
            res_tr = bt.simulate(
                self.train_close, TrendStrategy(p), 
                funding_series=self.funding_train, bnb_price_series=self.bnb_train
            )
            res_te = bt.simulate(
                self.test_close, TrendStrategy(p), 
                funding_series=self.funding_test, bnb_price_series=self.bnb_test
            )
            
            summ_te = res_te["summary"]
            summ_tr = res_tr["summary"]  
            # Test Metrics
            profit = float(summ_te["final_btc"])
            turns = float(summ_te["n_trades"])
            dd = float(summ_te["max_drawdown_pct"])
            fees = float(summ_te.get("fees_btc", 0.0))
            turnover = float(summ_te.get("turnover_btc", 0.0))
            
            # Train Metrics (The missing piece)
            train_profit = float(summ_tr["final_btc"])

            # Score: Profit penalized by Drawdown and Excessive Trading
            # Goal: High Profit, Low DD, Low Turns
            
            # 1. Profit Baseline
            score = profit 
            
            # 2. Penalty for blowing up account (Drawdown > 25%)
            if dd < -0.25:
                score -= 0.5 
                
            # 3. Penalty for Churning (too many trades is bad for Trend)
            # Simple penalty: more than 50 trades in test period
            if turns > 50:
                score -= 0.2 * (turns - 50) / 50

            # 5. Save Attributes for Analysis
            trial.set_user_attr("test_final_btc", profit)   # Alias for consistency
            trial.set_user_attr("train_final_btc", train_profit) # <--- SAVED NOW
            trial.set_user_attr("test_dd", dd)
            trial.set_user_attr("turns", turns)
            trial.set_user_attr("fees_btc", fees)
            trial.set_user_attr("turnover_btc", turnover)
            
            for k, v in p.__dict__.items():
                trial.set_user_attr(k, v)

            log.info(f"Trial {tid}: Profit={profit:.4f} (Train={train_profit:.4f}) DD={dd:.2%} (Fast={p.fast_period} Slow={p.slow_period})")
            return score

        except optuna.TrialPruned:
            raise
        except Exception as e:
            log.error(f"Trial {tid} Failed: {e}")
            return -1e9

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--funding-data")
    ap.add_argument("--bnb-data")
    ap.add_argument("--config")
    ap.add_argument("--train-start", required=True)
    ap.add_argument("--train-end", required=True)
    ap.add_argument("--test-start", required=True)
    ap.add_argument("--test-end", required=True)
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--out", default="results/opt_trend.csv")
    ap.add_argument("--jobs", type=int, default=1)
    
    # Fees defaults
    ap.add_argument("--maker-fee", type=float, default=0.0002)
    ap.add_argument("--taker-fee", type=float, default=0.0004)

    # --- NEW: Persistence Arguments ---
    ap.add_argument("--storage", default="sqlite:///data/db/optuna.db", help="Database URL")
    ap.add_argument("--study-name", default="trend_study", help="Unique name for this optimization")
    
    args = ap.parse_args()

    # Ensure DB directory exists
    if args.storage.startswith("sqlite:///"):
        path = args.storage.replace("sqlite:///", "")
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # Load Data
    print(f"Loading Data: {args.data}")
    df = load_vision_csv(args.data)
    df = df[df.index.notna()]
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    close = df["close"]
    
    funding_series = None
    if args.funding_data:
        print(f"Loading Funding: {args.funding_data}")
        f_df = pd.read_csv(args.funding_data)
        f_df["time"] = pd.to_datetime(f_df["time"], format="mixed", utc=True)
        f_df = f_df.set_index("time").sort_index()
        funding_series = f_df["rate"]

    # Slices
    train_close = close.loc[args.train_start:args.train_end].dropna()
    test_close = close.loc[args.test_start:args.test_end].dropna()
    
    # Align Funding
    f_tr = f_te = None
    if funding_series is not None:
        f_tr = funding_series.reindex(train_close.index, method="ffill").fillna(0.0)
        f_te = funding_series.reindex(test_close.index, method="ffill").fillna(0.0)

    # Fee Params
    fee = FeeParams(maker_fee=args.maker_fee, taker_fee=args.taker_fee)

    # --- UPDATED: Create Persistent Study ---
    print(f"Connecting to storage: {args.storage}")
    print(f"Resuming study: {args.study_name}")
    
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=args.storage,
        load_if_exists=True  # <--- This tells it to RESUME instead of overwrite
    )
    
    obj = Objective(args, fee, train_close, test_close, None, None, f_tr, f_te)
    
    print("Starting Optimization...")
    study.optimize(obj, n_trials=args.n_trials, n_jobs=args.jobs)
    
    # Save Results to CSV
    print(f"Saving results to {args.out}")
    rows = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            d = t.user_attrs.copy()
            d.update(t.params)
            d["score"] = t.value
            rows.append(d)
    
    if rows:
        df_out = pd.DataFrame(rows)
        # Ensure fees/turnover columns exist even if not present in all trials
        for col in ["fees_btc", "turnover_btc"]:
            if col not in df_out.columns:
                df_out[col] = 0.0
                
        sort_col = "test_profit" if "test_profit" in df_out.columns else "score"
        df_out.sort_values(sort_col, ascending=False).to_csv(args.out, index=False)
    else:
        print("No successful trials to save.")

if __name__ == "__main__":
    main()