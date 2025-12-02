#!/usr/bin/env python3
from __future__ import annotations
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
optuna.logging.set_verbosity(optuna.logging.WARNING)

def suggest_params(trial, allow_shorts=False):
    """Search Space for Trend Strategy"""
    fast = trial.suggest_int("fast_period", 10, 200, step=10)
    slow = trial.suggest_int("slow_period", 40, 400, step=20)
    ma_type = trial.suggest_categorical("ma_type", ["ema", "sma"])
    cooldown = trial.suggest_categorical("cooldown_minutes", [60, 120, 240, 360])
    
    # Funding limits (relevant for Futures)
    funding_long = trial.suggest_float("funding_limit_long", 0.01, 0.10)
    funding_short = trial.suggest_float("funding_limit_short", -0.10, -0.01)

    # Shorting Logic
    if allow_shorts:
        long_only = trial.suggest_categorical("long_only", [True, False])
    else:
        long_only = True

    return TrendParams(
        fast_period=fast, slow_period=slow, ma_type=ma_type,
        cooldown_minutes=cooldown, step_allocation=1.0, max_position=1.0,
        long_only=long_only, funding_limit_long=funding_long, funding_limit_short=funding_short
    )

class Objective:
    def __init__(self, fee, train_close, funding_train, allow_shorts=False):
        self.fee = fee
        self.train_close = train_close
        self.funding_train = funding_train
        self.allow_shorts = allow_shorts

    def __call__(self, trial):
        try:
            p = suggest_params(trial, self.allow_shorts)
            if p.fast_period >= p.slow_period:
                raise optuna.TrialPruned("Fast >= Slow")

            # 1. Run Simulation ONLY on Training Data (No Cheating!)
            bt = Backtester(self.fee)
            res_tr = bt.simulate(
                self.train_close, TrendStrategy(p), 
                funding_series=self.funding_train
            )
            
            summ_tr = res_tr["summary"]
            
            # 2. Calculate Score based on Training Data
            train_profit = float(summ_tr["final_btc"])
            turns = float(summ_tr["n_trades"])
            dd = float(summ_tr["max_drawdown_pct"])
            
            # Score: Profit penalized by extreme DD
            score = train_profit
            if dd < -0.25: score -= 0.5 
            if turns < 5: score -= 1.0 # Penalize inactive strategies
            
            # Store Training Metrics
            trial.set_user_attr("train_final_btc", train_profit)
            trial.set_user_attr("train_dd", dd)
            trial.set_user_attr("train_turns", turns)
            
            return score

        except optuna.TrialPruned:
            raise
        except Exception:
            return -1e9

def run_slice_optimization(args, fee, df, start_idx, end_idx, test_end_idx, funding_series):
    """Runs optimization for a specific window slice"""
    train_close = df["close"].iloc[start_idx:end_idx]
    test_close = df["close"].iloc[end_idx:test_end_idx]
    
    if len(train_close) < 100 or len(test_close) < 10:
        return None

    # Align Funding
    f_tr = f_te = None
    if funding_series is not None:
        f_tr = funding_series.reindex(train_close.index, method="ffill").fillna(0.0)
        f_te = funding_series.reindex(test_close.index, method="ffill").fillna(0.0)

    # Unique Study Name for this window
    window_name = f"{args.study_name}_{train_close.index[-1].strftime('%Y%m%d')}"
    
    study = optuna.create_study(
        study_name=window_name, direction="maximize",
        storage=args.storage, load_if_exists=True
    )
    
    # Optimize using ONLY Training Data + Allow Shorts setting
    obj = Objective(fee, train_close, f_tr, allow_shorts=args.allow_shorts)
    study.optimize(obj, n_trials=args.n_trials, n_jobs=args.jobs)
    
    # --- VALIDATION STEP ---
    # Take the best params from the PAST and test them on the FUTURE
    best_trial = study.best_trial
    best_params = best_trial.params
    
    # Re-construct params object safely
    p_best = TrendParams(
        fast_period=best_params["fast_period"],
        slow_period=best_params["slow_period"],
        ma_type=best_params["ma_type"],
        cooldown_minutes=best_params["cooldown_minutes"],
        funding_limit_long=best_params.get("funding_limit_long", 0.05),
        funding_limit_short=best_params.get("funding_limit_short", -0.05),
        long_only=best_params.get("long_only", True), # Respect the optimizer's choice
        step_allocation=1.0, max_position=1.0
    )

    # Run the "Out of Sample" Test
    bt = Backtester(fee)
    res_te = bt.simulate(test_close, TrendStrategy(p_best), funding_series=f_te)
    
    oos_profit = float(res_te["summary"]["final_btc"])
    train_profit = best_trial.value
    
    log.info(f"Window {train_close.index[-1].date()}: Train={train_profit:.4f} | OOS={oos_profit:.4f} | LongOnly={p_best.long_only}")
    
    return {
        "window_end": train_close.index[-1],
        "oos_start": test_close.index[0],
        "oos_end": test_close.index[-1],
        "oos_profit": oos_profit,
        "train_profit": train_profit,
        "best_params": json.dumps(best_params)
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--funding-data")
    ap.add_argument("--out", default="results/wfo_trend.csv")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--jobs", type=int, default=1)
    
    # WFO Settings
    ap.add_argument("--wfo", action="store_true", help="Enable Walk-Forward Optimization")
    ap.add_argument("--window-days", type=int, default=180, help="Training window size")
    ap.add_argument("--step-days", type=int, default=30, help="Step size for re-optimization")
    
    # Futures / Shorts
    ap.add_argument("--allow-shorts", action="store_true", help="Allow Short strategies (Futures)")

    # Standard Mode Arguments (Backwards Compatibility)
    ap.add_argument("--train-start")
    ap.add_argument("--train-end")
    ap.add_argument("--test-start")
    ap.add_argument("--test-end")

    # Fees
    ap.add_argument("--maker-fee", type=float, default=0.0002)
    ap.add_argument("--taker-fee", type=float, default=0.0004)

    ap.add_argument("--storage", default="sqlite:///data/db/optuna.db")
    ap.add_argument("--study-name", default="trend_study")
    
    args = ap.parse_args()

    if args.storage.startswith("sqlite:///"):
        os.makedirs(os.path.dirname(args.storage.replace("sqlite:///", "")), exist_ok=True)

    df = load_vision_csv(args.data).dropna()
    
    funding_series = None
    if args.funding_data:
        f_df = pd.read_csv(args.funding_data)
        f_df["time"] = pd.to_datetime(f_df["time"], format="mixed", utc=True)
        f_df = f_df.set_index("time").sort_index()
        funding_series = f_df["rate"]

    fee = FeeParams(maker_fee=args.maker_fee, taker_fee=args.taker_fee)

    # --- WFO MODE ---
    if args.wfo:
        log.info(f"Starting Walk-Forward Optimization (Window={args.window_days}d, Step={args.step_days}d)")
        if args.allow_shorts:
            log.info("🩳 Shorting (Futures) Enabled")

        bars_per_day = 96 # 15m candles
        window_bars = args.window_days * bars_per_day
        step_bars = args.step_days * bars_per_day
        
        wfo_results = []
        
        for i in range(0, len(df) - window_bars - step_bars, step_bars):
            train_end = i + window_bars
            test_end = train_end + step_bars
            
            res = run_slice_optimization(
                args, fee, df, i, train_end, test_end, funding_series
            )
            if res:
                wfo_results.append(res)
        
        if wfo_results:
            wfo_df = pd.DataFrame(wfo_results)
            wfo_df.to_csv(args.out, index=False)
            log.info(f"WFO Complete. Schedule saved to {args.out}")
        else:
            log.error("WFO failed: No valid windows found.")

    # --- STATIC MODE ---
    else:
        log.info("Starting Static Optimization")
        train_close = df["close"].loc[args.train_start:args.train_end].dropna()
        
        f_tr = None
        if funding_series is not None:
            f_tr = funding_series.reindex(train_close.index, method="ffill").fillna(0.0)

        study = optuna.create_study(
            study_name=args.study_name, direction="maximize",
            storage=args.storage, load_if_exists=True
        )
        # Use Train-Only objective here too for consistency? 
        # Usually static optimization wants to test on a specific hold-out set manually.
        # We will use the Objective class but we need to match the signature.
        # For static mode, we usually just want to fill the DB.
        
        obj = Objective(fee, train_close, f_tr, allow_shorts=args.allow_shorts)
        study.optimize(obj, n_trials=args.n_trials, n_jobs=args.jobs)
        
        # Save output
        rows = []
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                d = t.user_attrs.copy()
                d.update(t.params)
                rows.append(d)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        log.info("Static Optimization Complete.")

if __name__ == "__main__":
    main()