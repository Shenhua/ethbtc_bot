"""
Unified Optimizer CLI

The single entry point for all strategy optimizations.
Supports MR, Trend, and Meta strategies in both Static and WFO modes.

Usage:
    python3 -m tools.optimizer.cli --strategy mr --mode wfo --tag BTC
"""

import argparse
import sys
import os
import json
import logging
import pandas as pd
import optuna
from typing import Any, Dict

# Path discovery
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from core.ethbtc_accum_bot import load_vision_csv, FeeParams, StratParams, Backtester
from core.trend_strategy import TrendParams
# Core Framework
from .base import BaseOptimizer
from .progress import ProgressTracker, create_progress_queue
from .scoring import compute_robust_score
# Strategies
from .strategies.mean_reversion import MeanReversionOptimizer
from .strategies.trend import TrendOptimizer
from .strategies.meta import MetaOptimizer
# Modes
from .modes.wfo import run_wfo_optimization
from .modes.static import run_static_optimization


def main():
    parser = argparse.ArgumentParser(description="Unified Optimizer CLI")
    
    # Core settings
    parser.add_argument("--strategy", required=True, choices=["mr", "trend", "meta"])
    parser.add_argument("--mode", default="static", choices=["static", "wfo"])
    parser.add_argument("--tag", default="UNTAGGED")
    
    # Data settings
    parser.add_argument("--data", required=True, help="Path to price data CSV")
    parser.add_argument("--funding-data", help="Path to funding rates CSV")
    parser.add_argument("--bnb-data", help="Path to BNB price data CSV")
    
    # Time settings
    parser.add_argument("--train-start", default="2021-01-01")
    parser.add_argument("--train-end", default="2024-06-30")
    parser.add_argument("--test-start", default="2024-07-01")
    parser.add_argument("--test-end", default="2025-06-01")
    
    # WFO settings
    parser.add_argument("--window-days", type=int, default=365)
    parser.add_argument("--step-days", type=int, default=30)
    
    # Optimization settings
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--storage", help="Optuna storage URL (e.g. sqlite:///optuna.db)")
    parser.add_argument("--study-name", help="Optuna study name")
    
    # Strategy-specific overrides
    parser.add_argument("--force-trend-kind", help="MR: Lock trend_kind (sma/roc)")
    parser.add_argument("--force-sizing-mode", help="MR: Lock sizing_mode (static/volatility)")
    parser.add_argument("--force-long-only", help="MR/Trend: Lock long_only (true/false)")
    parser.add_argument("--allow-shorts", action="store_true", help="Trend: Allow shorting")
    
    # Meta specific
    parser.add_argument("--mr-config", help="Meta: Path to MR config")
    parser.add_argument("--trend-config", help="Meta: Path to Trend config")
    
    # Output
    parser.add_argument("--out", help="Path to output file")
    
    args = parser.parse_args()
    
    # 1. Setup Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger("OPTIMIZER")
    
    # 2. Load Data
    logger.info(f"Loading data from {args.data}...")
    df = load_vision_csv(args.data)
    
    funding_series = None
    if args.funding_data:
        f_df = pd.read_csv(args.funding_data)
        f_df["time"] = pd.to_datetime(f_df["time"], format="mixed", utc=True)
        f_df = f_df.set_index("time").sort_index()
        funding_series = f_df["rate"].reindex(df.index).ffill().fillna(0.0)
        
    bnb_series = None
    if args.bnb_data:
        df_bnb = load_vision_csv(args.bnb_data)
        bnb_series = df_bnb["close"].reindex(df.index, method="ffill")
        
    fee = FeeParams()
    bt = Backtester(fee)
    
    # 3. Setup Progress Tracking
    # If running as a subprocess of the orchestrator, we use stdout markers
    is_subprocess = os.environ.get("_OPTIMIZER_SUBPROCESS") == "1"
    tracker = ProgressTracker(use_stdout=is_subprocess)
    
    # 4. Instantiate Optimizer
    opt_instance = None
    suggest_kwargs = {}
    
    if args.strategy == "mr":
        opt_instance = MeanReversionOptimizer(fee, logger=logger)
        if args.force_trend_kind: suggest_kwargs["force_trend_kind"] = args.force_trend_kind
        if args.force_sizing_mode: suggest_kwargs["force_sizing_mode"] = args.force_sizing_mode
        if args.force_long_only: suggest_kwargs["force_long_only"] = args.force_long_only.lower() == "true"
        
    elif args.strategy == "trend":
        opt_instance = TrendOptimizer(fee, logger=logger)
        suggest_kwargs["allow_shorts"] = args.allow_shorts
        if args.force_long_only: suggest_kwargs["force_long_only"] = args.force_long_only.lower() == "true"
        
    elif args.strategy == "meta":
        if not args.mr_config or not args.trend_config:
            logger.error("Meta strategy requires --mr-config and --trend-config")
            sys.exit(1)
            
        with open(args.mr_config) as f: mr_raw = json.load(f)
        with open(args.trend_config) as f: tr_raw = json.load(f)
        
        # Simple cleanup helper (inline here for brevity, logic from optimize_meta.py)
        def clean(d, cls):
            valid = cls.__annotations__.keys()
            return {k: v for k, v in d.get("strategy", d).items() if k in valid}
            
        mr_params = StratParams(**clean(mr_raw, StratParams))
        tr_params = TrendParams(**clean(tr_raw, TrendParams))
        # FIX: Pass 'df' (full history) for signal caching
        opt_instance = MetaOptimizer(fee, mr_params, tr_params, full_df=df, logger=logger)
        
    # 5. Run Optimization
    if args.mode == "wfo":
        results = run_wfo_optimization(
            optimizer=opt_instance,
            df=df,
            window_days=args.window_days,
            step_days=args.step_days,
            n_trials=args.trials,
            backtester=bt,
            funding_series=funding_series,
            bnb_series=bnb_series,
            combo_name=args.tag,
            progress_tracker=tracker,
            storage=args.storage,
            study_name=args.study_name,
            **suggest_kwargs
        )
        
        # Save WFO results
        out_file = args.out or f"results/wfo_{args.strategy}_{args.tag}.csv"
        pd.DataFrame(results).to_csv(out_file, index=False)
        logger.info(f"WFO results saved to {out_file}")
        
    else:  # static
        # Split data
        train_df = df.loc[args.train_start:args.train_end]
        test_df = df.loc[args.test_start:args.test_end]
        
        f_train = funding_series.loc[args.train_start:args.train_end] if funding_series is not None else None
        f_test = funding_series.loc[args.test_start:args.test_end] if funding_series is not None else None
        
        # We need the study to extract all trials for legacy selection scripts
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            direction="maximize",
            load_if_exists=True
        )
        
        if not is_subprocess:
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
            
            with Progress(
                SpinnerColumn(),
                TextColumn(f"[bold blue]Optimizing {args.strategy.upper()}..."),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                transient=True
            ) as progress:
                task = progress.add_task("Optimizing", total=args.trials)
                
                def objective(trial):
                    val = opt_instance.run_trial(
                        trial,
                        train_data=train_df,
                        test_data=test_df,
                        backtester=bt,
                        funding_train=f_train,
                        funding_test=f_test,
                        **suggest_kwargs
                    )
                    progress.update(task, advance=1)
                    tracker.report_trial(args.tag, trial.number, val)
                    return val
                
                study.optimize(objective, n_trials=args.trials, n_jobs=args.jobs)
        else:
            # Subprocess mode: Just run optimizer, tracker handles signals
            def objective(trial):
                val = opt_instance.run_trial(
                    trial,
                    train_data=train_df,
                    test_data=test_df,
                    backtester=bt,
                    funding_train=f_train,
                    funding_test=f_test,
                    **suggest_kwargs
                )
                tracker.report_trial(args.tag, trial.number, val)
                return val
            
            study.optimize(objective, n_trials=args.trials, n_jobs=args.jobs)
        
        # 1. Save all trials as CSV (compatibility with wf_pick.py)
        trials_data = []
        for t in study.trials:
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            row = t.params.copy()
            row["value"] = t.value
            row.update(t.user_attrs) # This includes train_profit, test_profit, fees_btc, etc.
            trials_data.append(row)
        
        out_csv = args.out or f"results/opt_{args.strategy}_{args.tag}.csv"
        pd.DataFrame(trials_data).to_csv(out_csv, index=False)
        logger.info(f"Static trials saved to {out_csv}")
        
        # 2. Save best result as JSON (for reference)
        best_trial = study.best_trial
        best_results = {
            "best_params": opt_instance.params_to_json(best_trial.params),
            "score": best_trial.value,
            "detail": best_trial.user_attrs
        }
        out_json = out_csv.replace(".csv", ".json")
        with open(out_json, "w") as f:
            json.dump(best_results, f, indent=2)
        logger.info(f"Best static result saved to {out_json}")


if __name__ == "__main__":
    main()
