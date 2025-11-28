#!/usr/bin/env python3
from __future__ import annotations

import sys
import os
import math

# --- MAGIC PATH FIX ---
# Allows running this script from the root or tools/ folder without PYTHONPATH errors
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ----------------------

import json, argparse, time, random, logging
import pandas as pd
import numpy as np
import optuna
from core.ethbtc_accum_bot import (
    load_vision_csv, load_json_config, _write_excel,
    FeeParams, StratParams, EthBtcStrategy, Backtester
)

# 1. Setup Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [OPT] %(message)s', 
    datefmt='%H:%M:%S'
)
log = logging.getLogger("optimizer")

# Enable Optuna logs
optuna.logging.set_verbosity(optuna.logging.INFO)

def suggest_params(trial):
    """
    Define the search space for Optuna.
    """
    return StratParams(
        trend_kind=trial.suggest_categorical("trend_kind", ["sma", "roc"]),
        trend_lookback=trial.suggest_categorical("trend_lookback", [120, 160, 200, 240, 300]),
        
        flip_band_entry=trial.suggest_float("flip_band_entry", 0.01, 0.06),
        flip_band_exit=trial.suggest_float("flip_band_exit", 0.005, 0.03),
        
        vol_window=trial.suggest_categorical("vol_window", [45, 60, 90]),
        vol_adapt_k=trial.suggest_categorical("vol_adapt_k", [0.0, 0.0025, 0.005, 0.0075]),
        
        target_vol=trial.suggest_categorical("target_vol", [0.3, 0.4, 0.5, 0.6]),
        min_mult=0.5, 
        max_mult=1.5,
        
        cooldown_minutes=trial.suggest_categorical("cooldown_minutes", [60, 120, 180, 240]),
        step_allocation=trial.suggest_categorical("step_allocation", [0.33, 0.5, 0.66, 1.0]),
        max_position=trial.suggest_categorical("max_position", [0.6, 0.8, 1.0]),
        
        # Gates
        gate_window_days=trial.suggest_categorical("gate_window_days", [30, 60, 90]),
        gate_roc_threshold=trial.suggest_categorical("gate_roc_threshold", [0.0, 0.01, 0.02]),
        
        # Funding Rate Filters
        funding_limit_long=trial.suggest_float("funding_limit_long", 0.01, 0.10),
        funding_limit_short=trial.suggest_float("funding_limit_short", -0.10, -0.01),
        
        # Anti-Churn defaults
        rebalance_threshold_w=trial.suggest_categorical("rebalance_threshold_w", [0.0, 0.01]),
        min_trade_btc=0.0,
        
        # --- THE SHORTING SWITCH ---
        # True = Only Buy ETH. False = Buy & Sell Short.
        long_only=trial.suggest_categorical("long_only", [True, False])
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
            # 1. Sample
            p = suggest_params(trial)
            
            # 2. Run Simulation (Train)
            bt = Backtester(self.fee)
            res_tr = bt.simulate(
                self.train_close, EthBtcStrategy(p), 
                funding_series=self.funding_train, bnb_price_series=self.bnb_train
            )
            # 3. Run Simulation (Test)
            res_te = bt.simulate(
                self.test_close, EthBtcStrategy(p), 
                funding_series=self.funding_test, bnb_price_series=self.bnb_test
            )
            
            # 4. Calculate Metrics
            summ_tr = res_tr["summary"]
            summ_te = res_te["summary"]
            
            test_final = float(summ_te["final_btc"])
            train_final = float(summ_tr["final_btc"])
            turns = float(summ_te["n_trades"])
            fees = float(summ_te["fees_btc"])
            turnover = float(summ_te["turnover_btc"])
            
            gen_gap = max(0.0, train_final - test_final)
            if turns < 0: turns = 0
            if fees < 0: fees = 0
            
            # Prevent division by zero if turns_scale is 0 (unlikely but safe)
            t_scale = self.args.turns_scale if self.args.turns_scale > 0 else 1.0
            
            # CRITICAL FIX: Penalize strategies that don't trade at all
            # If the strategy doesn't trade (turns=0), it gets a terrible score
            if turns == 0:
                robust_score = -1000.0  # Massive penalty for not trading
            else:
                robust_score = (
                    test_final
                    - self.args.lambda_turns * (turns / t_scale)
                    - self.args.gap_penalty * gen_gap
                    - self.args.lambda_fees * fees
                    - self.args.lambda_turnover * turnover
                )
            
            # Check for valid float
            if not math.isfinite(robust_score):
                log.warning(f"Trial {tid}: Non-finite score {robust_score}. Profit={test_final}")
                return -1e9 # Return a bad finite number instead of -inf
            
            
            # 5. Store Attributes for CSV export
            trial.set_user_attr("train_final_btc", train_final)
            trial.set_user_attr("test_final_btc", test_final)
            trial.set_user_attr("turns_test", turns)
            trial.set_user_attr("fees_btc", fees)
            trial.set_user_attr("turnover_btc", turnover)
            trial.set_user_attr("robust_score", robust_score)
            
            for k, v in p.__dict__.items():
                trial.set_user_attr(k, v)

            log.info(f"Trial {tid} DONE: Score={robust_score:.4f} (Profit={test_final:.4f}) in {time.time()-t0:.2f}s")
            return robust_score

        except Exception as e:
            log.error(f"Trial {tid} CRASHED: {e}", exc_info=True)
            return -float('inf')

def main():
    ap = argparse.ArgumentParser(description="Bayesian Optimizer (Optuna)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--funding-data", help="Path to funding rates CSV")
    ap.add_argument("--bnb-data")
    ap.add_argument("--config")
    ap.add_argument("--train-start", required=True)
    ap.add_argument("--train-end", required=True)
    ap.add_argument("--test-start", required=True)
    ap.add_argument("--test-end", required=True)
    ap.add_argument("--n-trials", type=int, default=200)
    
    # Fees
    ap.add_argument("--maker-fee", type=float, default=0.0002)
    ap.add_argument("--taker-fee", type=float, default=0.0004)
    ap.add_argument("--bnb-discount", type=float, default=0.25)
    ap.add_argument("--no-bnb", action="store_true")
    ap.add_argument("--slippage-bps", type=float, default=1.0)
    
    # Scoring weights
    ap.add_argument("--lambda-turns", type=float, default=2.0)
    ap.add_argument("--gap-penalty", type=float, default=0.35)
    ap.add_argument("--turns-scale", type=float, default=800.0)
    ap.add_argument("--lambda-fees", type=float, default=2.0)
    ap.add_argument("--lambda-turnover", type=float, default=1.0)
    
    # Output
    ap.add_argument("--out", default="results/opt_results_smart.csv")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel jobs")
    ap.add_argument("--storage", default="sqlite:///data/db/optuna.db")
    ap.add_argument("--study-name", default="ethbtc_study")
    ap.add_argument("--top-quantile", type=float, default=0.95)
    ap.add_argument("--emit-config")
    ap.add_argument("--no-excel", action="store_true")
    
    # Compatibility args (ignored but accepted so old scripts don't break)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=32)
    ap.add_argument("--early-stop", type=int, default=120)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min-improve", type=float, default=0.005)
    
    args = ap.parse_args()

    cfg = load_json_config(args.config)
    fee = FeeParams(
        maker_fee=float(cfg.get("maker_fee", args.maker_fee)),
        taker_fee=float(cfg.get("taker_fee", args.taker_fee)),
        bnb_discount=float(cfg.get("bnb_discount", args.bnb_discount)),
        slippage_bps=float(cfg.get("slippage_bps", args.slippage_bps)),
        pay_fees_in_bnb=bool(cfg.get("pay_fees_in_bnb", not args.no_bnb)),
    )

    log.info(f"Loading price data from {args.data}...")
    df = load_vision_csv(args.data)
    # Drop NaT index if any
    df = df[df.index.notna()]
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    close = df["close"]
    
    print(f"Index Monotonic: {close.index.is_monotonic_increasing}")
    train_close = close.loc[args.train_start:args.train_end].dropna()
    test_close  = close.loc[args.test_start:args.test_end].dropna()

    bnb_train = bnb_test = None
    if cfg.get("bnb_data", args.bnb_data):
        df_bnb = load_vision_csv(cfg.get("bnb_data", args.bnb_data))["close"]
        bnb_train = df_bnb.reindex(train_close.index, method="ffill")
        bnb_test  = df_bnb.reindex(test_close.index,  method="ffill")

    funding_train = funding_test = None
    if args.funding_data:
        log.info(f"Loading funding data from {args.funding_data}...")
        f_df = pd.read_csv(args.funding_data)
        f_df["time"] = pd.to_datetime(f_df["time"], format="mixed", utc=True)
        f_df = f_df.set_index("time").sort_index()
        funding_series = f_df["rate"]
        funding_train = funding_series.reindex(train_close.index, method="ffill").fillna(0.0)
        funding_test  = funding_series.reindex(test_close.index, method="ffill").fillna(0.0)

    log.info(f"Starting Optuna study '{args.study_name}' with {args.n_trials} trials...")
    
    study = optuna.create_study(
        study_name=args.study_name, 
        direction="maximize",
        storage=args.storage,
        load_if_exists=True
    )
    
    obj = Objective(args, fee, train_close, test_close, bnb_train, bnb_test, funding_train, funding_test)
    
    try:
        study.optimize(obj, n_trials=args.n_trials, n_jobs=args.jobs)
    except KeyboardInterrupt:
        log.warning("Stopping optimization early...")

    log.info(f"Exporting results to {args.out}...")
    
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = t.user_attrs.copy()
        row.update(t.params) 
        rows.append(row)
        
    df_out = pd.DataFrame(rows)
    if "robust_score" not in df_out.columns and "value" in df_out.columns:
        df_out.rename(columns={"value": "robust_score"}, inplace=True)

    if not df_out.empty:
        df_out = df_out.sort_values("robust_score", ascending=False)
        df_out.to_csv(args.out, index=False)
        log.info(f"Done. Best score: {study.best_value:.4f}")
        print(json.dumps(study.best_trial.params, indent=2))
    else:
        log.warning("No successful trials found.")

if __name__ == "__main__":
    main()