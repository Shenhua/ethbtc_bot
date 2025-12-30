"""
Walk-Forward Optimization Mode

Standardized WFO loop that works with any BaseOptimizer subclass.
Handles rolling windows, data slicing, and parallel execution.

Author: AI Audit System
Date: 2024-12-30
"""

import time
import json
import pandas as pd
from typing import Any, Dict, List, Optional
from datetime import timedelta
import optuna

from ..base import BaseOptimizer
from ..progress import ProgressTracker


def run_wfo_optimization(
    optimizer: BaseOptimizer,
    df: pd.DataFrame,
    window_days: int,
    step_days: int,
    n_trials: int,
    backtester: Any,
    funding_series: Optional[pd.Series] = None,
    bnb_series: Optional[pd.Series] = None,
    progress_tracker: Optional[ProgressTracker] = None,
    combo_name: str = "WFO",
    storage: Optional[str] = None,
    study_name: Optional[str] = None,
    **suggest_kwargs
) -> List[Dict[str, Any]]:
    """
    Standardized WFO loop.
    
    Args:
        optimizer: A BaseOptimizer subclass instance
        df: Full OHLC DataFrame
        window_days: Lookback window size in days
        step_days: Step size for rolling window in days
        n_trials: Number of trials per window
        backtester: Backtester instance
        funding_series: Optional funding rates
        bnb_series: Optional BNB prices
        progress_tracker: Optional tracker for remote reporting
        combo_name: Label for this combo
        storage: Optuna storage URL (optional)
        study_name: Base study name (optional)
        **suggest_kwargs: Passed to optimizer.suggest_params
        
    Returns:
        List of window result dicts.
    """
    bars_per_day = 96  # 15m candles
    window_bars = window_days * bars_per_day
    step_bars = step_days * bars_per_day
    
    total_iterations = (max(0, len(df) - window_bars - step_bars)) // step_bars
    if progress_tracker:
        progress_tracker.total_windows = max(1, total_iterations)
        progress_tracker.report_total_windows(combo_name, progress_tracker.total_windows)
    
    wfo_results = []
    window_count = 0
    
    for i in range(0, len(df) - window_bars - step_bars, step_bars):
        window_count += 1
        
        if progress_tracker:
            progress_tracker.report_window(combo_name, window_count)
            
        train_end = i + window_bars
        test_end = train_end + step_bars
        
        train_df = df.iloc[i:train_end]
        test_df = df.iloc[train_end:test_end]
        
        # Align series
        f_tr = f_te = None
        if funding_series is not None:
            f_tr = funding_series.reindex(train_df.index, method="ffill").fillna(0.0)
            f_te = funding_series.reindex(test_df.index, method="ffill").fillna(0.0)
            
        b_tr = b_te = None
        if bnb_series is not None:
            b_tr = bnb_series.reindex(train_df.index, method="ffill")
            b_te = bnb_series.reindex(test_df.index, method="ffill")
            
        # Run window optimization
        # Use persistent storage if provided
        current_study_name = None
        if storage and study_name:
            # Create a unique name for this window slice to allow resuming/compounding
            # Format: {base}_w{index}_{date}
            # e.g. "params_BTC_wfo_w1_20210101"
            start_date_str = train_df.index[0].strftime("%Y%m%d")
            current_study_name = f"{study_name}_w{window_count}_{start_date_str}"
            
            # STAGGER START: Add random jitter to prevent SQLite locking on parallel starts
            import random
            time.sleep(random.uniform(0.5, 3.0))

        # Retry logic for DB contention
        max_retries = 3
        study = None
        for attempt in range(max_retries):
            try:
                study = optuna.create_study(
                    direction="maximize",
                    storage=storage,
                    study_name=current_study_name,
                    load_if_exists=True
                )
                break
            except Exception as e:
                if attempt == max_retries - 1: raise e
                time.sleep(1 + attempt)  # Backoff
        
        def objective(trial):
            score = optimizer.run_trial(
                trial,
                train_data=train_df,
                test_data=test_df,
                backtester=backtester,
                funding_train=f_tr,
                funding_test=f_te,
                **suggest_kwargs
            )
            if progress_tracker:
                progress_tracker.report_trial(combo_name, trial.number, score)
            return score
            
        study.optimize(objective, n_trials=n_trials)
        best_trial = study.best_trial
        
        res = {
            "window_end": train_df.index[-1],
            "oos_start": test_df.index[0],
            "oos_end": test_df.index[-1],
            "oos_profit": best_trial.user_attrs.get("test_profit", best_trial.value),
            "train_profit": best_trial.user_attrs.get("train_profit", best_trial.value),
            "best_params": json.dumps(best_trial.params)
        }
        wfo_results.append(res)
            
    return wfo_results
