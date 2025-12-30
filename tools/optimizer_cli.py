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

LONG_ONLY_MODE = "both"  # "true", "false", or "both"

# 1. Setup Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [OPT] %(message)s', 
    datefmt='%H:%M:%S'
)
log = logging.getLogger("optimizer")

# Enable Optuna logs
optuna.logging.set_verbosity(optuna.logging.INFO)

# Global force flags (set by CLI)
FORCE_FLAGS = {}

def suggest_params(trial):
    """
    Define the search space for Optuna.
    Respects FORCE_FLAGS to lock specific parameters.
    """
    # --- THE SHORTING SWITCH ---
    # Choices depend on global LONG_ONLY_MODE
    if LONG_ONLY_MODE == "true":
        long_only_choices = [True]
    elif LONG_ONLY_MODE == "false":
        long_only_choices = [False]
    else:  # "both"
        long_only_choices = [True, False]
    
    return StratParams(
        trend_kind=FORCE_FLAGS.get("trend_kind") or trial.suggest_categorical("trend_kind", ["sma", "roc"]),
        trend_lookback=FORCE_FLAGS.get("trend_lookback") or trial.suggest_categorical("trend_lookback", [120, 160, 200, 240, 300]),
        
        flip_band_entry=trial.suggest_float("flip_band_entry", 0.01, 0.06),
        flip_band_exit=trial.suggest_float("flip_band_exit", 0.005, 0.03),
        
        vol_window=trial.suggest_categorical("vol_window", [45, 60, 90]),
        vol_adapt_k=trial.suggest_categorical("vol_adapt_k", [0.0, 0.0025, 0.005, 0.0075]),
        
        target_vol=trial.suggest_categorical("target_vol", [0.3, 0.4, 0.5, 0.6]),
        min_mult=trial.suggest_float("min_mult", 0.3, 0.7, step=0.1),
        max_mult=trial.suggest_float("max_mult", 1.2, 2.0, step=0.1),
        
        cooldown_minutes=trial.suggest_categorical("cooldown_minutes", [60, 120, 180, 240]),
        step_allocation=trial.suggest_categorical("step_allocation", [0.33, 0.5, 0.66, 1.0]),
        max_position=trial.suggest_categorical("max_position", [0.6, 0.8, 1.0]),
        
        # Dynamic Position Sizing (NEW!)
        position_sizing_mode=FORCE_FLAGS.get("position_sizing_mode") or trial.suggest_categorical("position_sizing_mode", ["static", "volatility"]),
        position_sizing_target_vol=trial.suggest_float("position_sizing_target_vol", 0.3, 0.7),
        position_sizing_min_step=trial.suggest_float("position_sizing_min_step", 0.1, 0.3),
        position_sizing_max_step=1.0,  # Keep at 1.0 (full allocation)
        
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
        long_only=FORCE_FLAGS.get("long_only") if "long_only" in FORCE_FLAGS else trial.suggest_categorical("long_only", long_only_choices),
    )

class Objective:
    def __init__(self, args, fee, train_close, test_close, train_bnb, test_bnb, train_funding, test_funding, train_df, test_df):
        self.args = args
        self.fee = fee
        self.train_close = train_close
        self.test_close = test_close
        self.train_bnb = train_bnb
        self.test_bnb = test_bnb
        self.train_funding = train_funding
        self.test_funding = test_funding
        self.train_df = train_df  # Full OHLC for MetaStrategy
        self.test_df = test_df    # Full OHLC for MetaStrategy
        self.bt = Backtester(fee)

    def __call__(self, trial):
        tid = trial.number  # Use Optuna's trial number (0-indexed, matches their output)
        t0 = time.time()
        
        try:
            # 1. Sample
            p = suggest_params(trial)
            
            # 2. Run Simulation (Train)
            bt = Backtester(self.fee)
            res_tr = bt.simulate(
                self.train_close, EthBtcStrategy(p),  # Wrap params in strategy
                funding_series=self.train_funding, bnb_price_series=self.train_bnb,
                full_df=self.train_df  # Pass full OHLC for MetaStrategy
            )
            # 3. Run Simulation (Test)
            res_te = bt.simulate(
                self.test_close, EthBtcStrategy(p),  # Wrap params in strategy
                funding_series=self.test_funding, bnb_price_series=self.test_bnb,
                full_df=self.test_df  # Pass full OHLC for MetaStrategy
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
            return -1e9


# =============================================
# Walk-Forward Optimization Helper (NEW!)
# =============================================
def run_slice_optimization_mr(args, fee, df, start_idx, end_idx, test_end_idx, funding_series, bnb_series):
    """
    Run Mean Reversion optimization for a single WFO window.
    
    This mirrors the logic in optimize_trend.py::run_slice_optimization
    but uses the MR strategy and EthBtcStrategy.
    
    Args:
        args: CLI arguments
        fee: FeeParams instance
        df: Full OHLC dataframe
        start_idx: Start index for training window
        end_idx: End index for training window
        test_end_idx: End index for test window
        funding_series: Optional funding rate series
        bnb_series: Optional BNB price series
    
    Returns:
        Dict with window results or None if window is too small.
    """
    train_close = df["close"].iloc[start_idx:end_idx]
    test_close = df["close"].iloc[end_idx:test_end_idx]
    
    if len(train_close) < 100 or len(test_close) < 10:
        return None
    
    # Align funding and BNB prices to window
    f_tr = f_te = None
    if funding_series is not None:
        f_tr = funding_series.reindex(train_close.index, method="ffill").fillna(0.0)
        f_te = funding_series.reindex(test_close.index, method="ffill").fillna(0.0)
    
    bnb_tr = bnb_te = None
    if bnb_series is not None:
        bnb_tr = bnb_series.reindex(train_close.index, method="ffill")
        bnb_te = bnb_series.reindex(test_close.index, method="ffill")
    
    # Unique study name for this window
    window_name = f"{args.study_name}_{train_close.index[-1].strftime('%Y%m%d')}"
    log.info(f"[WFO] Starting window: {window_name}")
    
    study = optuna.create_study(
        study_name=window_name, direction="maximize",
        storage=args.storage, load_if_exists=True
    )
    
    # Create dataframes for Objective
    train_df = df.iloc[start_idx:end_idx]
    test_df = df.iloc[end_idx:test_end_idx]
    
    obj = Objective(args, fee, train_close, test_close, bnb_tr, bnb_te, f_tr, f_te, train_df, test_df)
    study.optimize(obj, n_trials=args.n_trials, n_jobs=args.jobs)
    
    # Get best trial
    best_trial = study.best_trial
    oos_profit = best_trial.user_attrs.get("test_final_btc", best_trial.value)
    train_profit = best_trial.user_attrs.get("train_final_btc", best_trial.value)
    
    log.info(f"[WFO] Window {train_close.index[-1].date()}: Train={train_profit:.4f} | OOS={oos_profit:.4f}")
    
    return {
        "window_end": train_close.index[-1],
        "oos_start": test_close.index[0],
        "oos_end": test_close.index[-1],
        "oos_profit": oos_profit,
        "train_profit": train_profit,
        "best_params": json.dumps(best_trial.params)
    }


def main():
    ap = argparse.ArgumentParser(description="Bayesian Optimizer (Optuna) for Mean Reversion")
    ap.add_argument("--data", required=True)
    ap.add_argument("--funding-data", help="Path to funding rates CSV")
    ap.add_argument("--bnb-data")
    ap.add_argument("--config")
    
    # Date arguments (optional if --wfo is used)
    ap.add_argument("--train-start", default=None)
    ap.add_argument("--train-end", default=None)
    ap.add_argument("--test-start", default=None)
    ap.add_argument("--test-end", default=None)
    
    # Walk-Forward Optimization Mode (NEW!)
    ap.add_argument("--wfo", action="store_true", help="Enable Walk-Forward Optimization (rolling windows)")
    ap.add_argument("--window-days", type=int, default=180, help="Training window size in days")
    ap.add_argument("--step-days", type=int, default=30, help="Step size for re-optimization in days")
    
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

    # Long-only search mode
    ap.add_argument(
        "--long-only-mode",
        choices=["true", "false", "both"],
        default="both",
        help="Control search over long_only: 'true' (only long), 'false' (only short+long), 'both'.",
    )

    # Output
    ap.add_argument("--out", default="results/opt_results_smart.csv")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel jobs")
    ap.add_argument("--storage", default="sqlite:///data/db/optuna.db")
    ap.add_argument("--study-name", default="ethbtc_study")
    
    # === NEW: Exploration Control ===
    ap.add_argument("--n-startup-trials", type=int, default=50, help="Number of random exploration trials (default: 50)")
    
    # === NEW: Force Flags ===
    ap.add_argument("--force-trend-kind", choices=["sma", "roc"], help="Lock trend_kind to specific value")
    ap.add_argument("--force-sizing-mode", choices=["static", "volatility"], help="Lock position_sizing_mode")
    ap.add_argument("--force-long-only", type=lambda x: x.lower() == 'true', help="Lock long_only (true/false)")
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

    # Wire CLI flags into globals used by suggest_params
    global LONG_ONLY_MODE, FORCE_FLAGS
    LONG_ONLY_MODE = args.long_only_mode
    
    # Set force flags
    if args.force_trend_kind:
        FORCE_FLAGS["trend_kind"] = args.force_trend_kind
        log.info(f"🔒 Locked trend_kind = {args.force_trend_kind}")
    if args.force_sizing_mode:
        FORCE_FLAGS["position_sizing_mode"] = args.force_sizing_mode
        log.info(f"🔒 Locked position_sizing_mode = {args.force_sizing_mode}")
    if args.force_long_only is not None:
        FORCE_FLAGS["long_only"] = args.force_long_only
        log.info(f"🔒 Locked long_only = {args.force_long_only}")

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
    
    # Load optional funding series (global)
    funding_series = None
    if args.funding_data:
        log.info(f"Loading funding data from {args.funding_data}...")
        f_df = pd.read_csv(args.funding_data)
        f_df["time"] = pd.to_datetime(f_df["time"], format="mixed", utc=True)
        f_df = f_df.set_index("time").sort_index()
        funding_series = f_df["rate"]
    
    # Load optional BNB series (global)
    bnb_series = None
    if cfg.get("bnb_data", args.bnb_data):
        bnb_series = load_vision_csv(cfg.get("bnb_data", args.bnb_data))["close"]

    # =============================================
    # WFO MODE: Walk-Forward Optimization
    # =============================================
    if args.wfo:
        log.info(f"🚀 Starting Walk-Forward Optimization (Window={args.window_days}d, Step={args.step_days}d)")
        
        bars_per_day = 96  # 15m candles
        window_bars = args.window_days * bars_per_day
        step_bars = args.step_days * bars_per_day
        
        wfo_results = []
        
        for i in range(0, len(df) - window_bars - step_bars, step_bars):
            train_end = i + window_bars
            test_end = train_end + step_bars
            
            res = run_slice_optimization_mr(
                args, fee, df, i, train_end, test_end, funding_series, bnb_series
            )
            if res:
                wfo_results.append(res)
        
        if wfo_results:
            wfo_df = pd.DataFrame(wfo_results)
            wfo_df.to_csv(args.out, index=False)
            log.info(f"✅ WFO Complete. {len(wfo_results)} windows saved to {args.out}")
        else:
            log.error("❌ WFO failed: No valid windows found.")
            
    # =============================================
    # STATIC MODE: Traditional Train/Test Split
    # =============================================
    else:
        # Validate required date arguments for static mode
        if not all([args.train_start, args.train_end, args.test_start, args.test_end]):
            log.error("Static mode requires --train-start, --train-end, --test-start, --test-end")
            log.error("Use --wfo for Walk-Forward Optimization without explicit dates.")
            sys.exit(1)
        
        train_close = close.loc[args.train_start:args.train_end].dropna()
        test_close  = close.loc[args.test_start:args.test_end].dropna()

        bnb_train = bnb_test = None
        if bnb_series is not None:
            bnb_train = bnb_series.reindex(train_close.index, method="ffill")
            bnb_test  = bnb_series.reindex(test_close.index,  method="ffill")

        funding_train = funding_test = None
        if funding_series is not None:
            funding_train = funding_series.reindex(train_close.index, method="ffill").fillna(0.0)
            funding_test  = funding_series.reindex(test_close.index, method="ffill").fillna(0.0)

        log.info(f"Starting Optuna study '{args.study_name}' with {args.n_trials} trials...")
        log.info(f"🔍 Exploration: {args.n_startup_trials}/{args.n_trials} trials will explore randomly")
        
        # Create sampler with improved exploration
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=args.n_startup_trials,
            multivariate=True,
            seed=42
        )
        
        study = optuna.create_study(
            study_name=args.study_name, 
            direction="maximize",
            storage=args.storage,
            load_if_exists=True,
            sampler=sampler
        )
        
        train_df = df.loc[args.train_start:args.train_end]
        test_df = df.loc[args.test_start:args.test_end]
        
        obj = Objective(args, fee, train_close, test_close, bnb_train, bnb_test, funding_train, funding_test, train_df, test_df)
        
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
        
        if "robust_score" not in df_out.columns:
            if "value" in df_out.columns:
                df_out["robust_score"] = df_out["value"]
            elif "test_final_btc" in df_out.columns:
                df_out["robust_score"] = df_out["test_final_btc"]
            else:
                df_out["robust_score"] = 0.0

        if not df_out.empty:
            df_out = df_out.sort_values("robust_score", ascending=False)
            df_out.to_csv(args.out, index=False)
            log.info(f"Done. Best score: {study.best_value:.4f}")
            print(json.dumps(study.best_trial.params, indent=2))
        else:
            log.warning("No successful trials found.")

if __name__ == "__main__":
    main()