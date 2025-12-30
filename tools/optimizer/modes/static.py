"""
Static Optimization Mode

Standardized static train/test split optimization.
Useful for quick re-optimizations or small datasets.

Author: AI Audit System
Date: 2024-12-30
"""

from typing import Any, Dict, Optional
import optuna
import pandas as pd

from ..base import BaseOptimizer


def run_static_optimization(
    optimizer: BaseOptimizer,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_trials: int,
    backtester: Any,
    funding_train: Optional[pd.Series] = None,
    funding_test: Optional[pd.Series] = None,
    **suggest_kwargs
) -> Dict[str, Any]:
    """
    Single train/test split optimization.
    
    Returns:
        Dict with best params and performance metrics.
    """
    study = optuna.create_study(direction="maximize")
    
    def objective(trial):
        # Emit progress signal
        import json
        print(json.dumps({
            "signal": "OPTIMIZER_PROGRESS",
            "type": "trial",
            "data": trial.number + 1
        }), flush=True)
        
        return optimizer.run_trial(
            trial,
            train_data=train_df,
            test_data=test_df,
            backtester=backtester,
            funding_train=funding_train,
            funding_test=funding_test,
            **suggest_kwargs
        )
    
    study.optimize(objective, n_trials=n_trials)
    
    best_trial = study.best_trial
    best_params = optimizer.params_to_json(best_trial.params)
    
    return {
        "best_params": best_params,
        "score": best_trial.value,
        "detail": best_trial.user_attrs
    }
