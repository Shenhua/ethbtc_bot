"""
Base Optimizer Abstract Class

Provides the foundation for all strategy optimizers with:
- Abstract methods for strategy-specific logic (suggest_params, create_strategy)
- Shared methods for WFO and static optimization
- Unified trial execution and scoring

Author: AI Audit System
Date: 2024-12-30
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging
import optuna
import pandas as pd
import numpy as np

from .scoring import compute_robust_score


class BaseOptimizer(ABC):
    """
    Abstract base class for all strategy optimizers.
    
    Subclasses must implement:
    - suggest_params(trial) -> Dict: Define parameter search space
    - create_strategy(params) -> Any: Instantiate strategy from params
    - get_strategy_name() -> str: Human-readable strategy name
    
    The base class provides:
    - run_trial(): Unified trial execution with backtesting
    - score(): Apply robust scoring with penalties
    - Common data handling
    """
    
    def __init__(
        self,
        fee_params: Any,
        logger: Optional[logging.Logger] = None,
        min_trades: int = 10,
        penalty_gap: float = 0.25,
        penalty_trades: float = 0.001,
    ):
        """
        Initialize the base optimizer.
        
        Args:
            fee_params: FeeParams instance for backtesting
            logger: Optional logger (creates default if None)
            min_trades: Minimum trades required (prune if less)
            penalty_gap: Penalty weight for train/test gap
            penalty_trades: Penalty weight per excess trade
        """
        self.fee = fee_params
        self.log = logger or logging.getLogger(self.get_strategy_name())
        self.min_trades = min_trades
        self.penalty_gap = penalty_gap
        self.penalty_trades = penalty_trades
    
    # =========================================================================
    # ABSTRACT METHODS (must be implemented by subclasses)
    # =========================================================================
    
    @abstractmethod
    def suggest_params(self, trial: optuna.Trial, **kwargs) -> Dict[str, Any]:
        """
        Define the parameter search space for Optuna.
        
        Args:
            trial: Optuna trial object
            **kwargs: Additional context (e.g., force_flags)
        
        Returns:
            Dict of parameter name -> value
        """
        pass
    
    @abstractmethod
    def create_strategy(self, params: Dict[str, Any]) -> Any:
        """
        Instantiate a strategy object from parameters.
        
        Args:
            params: Parameter dict from suggest_params
        
        Returns:
            Strategy instance ready for backtesting
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the human-readable strategy name (e.g., 'Mean Reversion')."""
        pass
    
    # =========================================================================
    # SHARED METHODS (inherited by all subclasses)
    # =========================================================================
    
    def run_trial(
        self,
        trial: optuna.Trial,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        backtester: Any,
        funding_train: Optional[pd.Series] = None,
        funding_test: Optional[pd.Series] = None,
        **suggest_kwargs
    ) -> float:
        """
        Execute a single optimization trial.
        
        This is the main entry point called by Optuna's optimize().
        
        Args:
            trial: Optuna trial object
            train_data: Training period DataFrame (must have 'close' column)
            test_data: Test period DataFrame
            backtester: Backtester instance with simulate() method
            funding_train: Optional funding rates for training
            funding_test: Optional funding rates for testing
            **suggest_kwargs: Passed to suggest_params (e.g., force_trend_kind)
        
        Returns:
            Robust score (higher is better)
        """
        # 1. Get parameters
        params = self.suggest_params(trial, **suggest_kwargs)
        
        # 2. Create strategy
        try:
            strategy = self.create_strategy(params)
        except Exception as e:
            self.log.warning(f"Failed to create strategy: {e}")
            return -1e9  # Prune
        
        # 3. Run train backtest
        train_result = backtester.simulate(
            train_data["close"],
            strategy,
            funding_series=funding_train,
            full_df=train_data
        )
        
        # 4. Run test backtest
        test_result = backtester.simulate(
            test_data["close"],
            strategy,
            funding_series=funding_test,
            full_df=test_data
        )
        
        # 5. Calculate and return score
        return self.score(train_result, test_result, params, trial)
    
    def score(
        self,
        train_result: Dict,
        test_result: Dict,
        params: Dict,
        trial: Optional[optuna.Trial] = None
    ) -> float:
        """
        Calculate robust score and save metrics to trial.
        """
        score = compute_robust_score(
            train_result=train_result,
            test_result=test_result,
            params=params,
            penalty_gap=self.penalty_gap,
            penalty_trades=self.penalty_trades,
            min_trades=self.min_trades,
        )
        
        if trial:
            test_sum = test_result.get("summary", {})
            train_sum = train_result.get("summary", {})
            
            # Modern names
            trial.set_user_attr("train_profit", train_sum.get("final_btc", 1.0))
            trial.set_user_attr("test_profit", test_sum.get("final_btc", 1.0))
            trial.set_user_attr("drawdown", test_sum.get("max_drawdown_pct", 0.0))
            trial.set_user_attr("n_trades", test_sum.get("n_trades", 0))
            trial.set_user_attr("fees_btc", test_sum.get("fees_btc", 0.0))
            trial.set_user_attr("turnover", test_sum.get("turnover", 0.0))
            trial.set_user_attr("robust_score", score)
            
            # Legacy names (for wf_pick.py compatibility)
            trial.set_user_attr("train_final_btc", train_sum.get("final_btc", 1.0))
            trial.set_user_attr("test_final_btc", test_sum.get("final_btc", 1.0))
            trial.set_user_attr("final_btc", test_sum.get("final_btc", 1.0)) # Direct legacy name
            trial.set_user_attr("turns_test", test_sum.get("n_trades", 0))
            trial.set_user_attr("turnover_btc", test_sum.get("turnover", 0.0))
            
        return score
    
    def params_to_json(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert parameters to JSON-serializable format.
        
        Handles numpy types, etc.
        """
        result = {}
        for k, v in params.items():
            if isinstance(v, (np.integer, np.floating)):
                result[k] = v.item()
            elif isinstance(v, np.ndarray):
                result[k] = v.tolist()
            else:
                result[k] = v
        return result
