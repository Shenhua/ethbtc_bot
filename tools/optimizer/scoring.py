"""
Unified Scoring Module

Provides consistent scoring functions for all optimizers.
Extracted from the best patterns across existing scripts.

Author: AI Audit System
Date: 2024-12-30
"""

from typing import Any, Dict
import numpy as np


def compute_robust_score(
    train_result: Dict[str, Any],
    test_result: Dict[str, Any],
    params: Dict[str, Any],
    penalty_gap: float = 0.25,
    penalty_trades: float = 0.001,
    penalty_fees: float = 0.0,
    penalty_turnover: float = 0.0,
    min_trades: int = 10,
    max_drawdown_limit: float = 0.5,
) -> float:
    """
    Calculate a robust optimization score.
    
    This unified scoring function combines the best patterns from:
    - optimizer_cli.py: Gap penalty, trade count check
    - multi_interval_opt.py: Robust scoring with multiple penalties
    - optimize_trend.py: Drawdown consideration
    
    Components:
    1. Test profit (primary metric)
    2. Train/test gap penalty (prevents overfitting)
    3. Trade count normalization (prevents overtrading)
    4. Fee penalty (optional)
    5. Turnover penalty (optional)
    
    Args:
        train_result: Result dict from Backtester.simulate() for training
        test_result: Result dict from Backtester.simulate() for testing
        params: Parameter dict (for logging/debugging)
        penalty_gap: Weight for train/test gap penalty (default 0.25)
        penalty_trades: Weight per excess trade over 100 (default 0.001)
        penalty_fees: Weight for total fees (default 0)
        penalty_turnover: Weight for turnover (default 0)
        min_trades: Minimum trades required (returns -1e9 if below)
        max_drawdown_limit: Prune if drawdown exceeds this (default 0.5 = 50%)
    
    Returns:
        Score value. Higher is better. Negative values indicate pruning.
    """
    # Extract metrics
    train_summary = train_result.get("summary", {})
    test_summary = test_result.get("summary", {})
    
    train_profit = train_summary.get("final_btc", 1.0)
    test_profit = test_summary.get("final_btc", 1.0)
    n_trades = test_summary.get("n_trades", 0)
    test_drawdown = test_summary.get("max_drawdown_pct", 0.0)
    fees = test_summary.get("fees_btc", 0.0)
    turnover = test_summary.get("turnover_btc", 0.0)
    
    # === PRUNING CONDITIONS ===
    
    # Prune if insufficient trades
    if n_trades < min_trades:
        return -1e9
    
    # Prune if excessive drawdown
    if abs(test_drawdown) > max_drawdown_limit:
        return -1e8
    
    # Prune if negative test profit (losing strategy)
    if test_profit < 0.95:  # Lost more than 5% of capital
        return -1e7
    
    # === SCORE CALCULATION ===
    
    # Base score: test profit (centered around 1.0 BTC starting capital)
    score = test_profit
    
    # Penalty for train/test gap (overfitting indicator)
    # Only penalize if train >> test (overfitting), not if test > train (robust)
    gap = max(0.0, train_profit - test_profit)
    score -= penalty_gap * gap
    
    # Penalty for excessive trading
    excess_trades = max(0, n_trades - 100)
    score -= penalty_trades * excess_trades
    
    # Optional: Fee penalty
    if penalty_fees > 0:
        score -= penalty_fees * fees
    
    # Optional: Turnover penalty
    if penalty_turnover > 0:
        score -= penalty_turnover * turnover
    
    return score


def compute_consistency_ratio(train_profit: float, test_profit: float) -> float:
    """
    Calculate the train/test consistency ratio (WFE-like metric).
    
    A ratio close to 1.0 indicates good generalization.
    Ratio > 1.5 or < 0.7 is suspicious (overfitting or data leakage).
    
    Returns:
        Ratio of test_profit / train_profit
    """
    if train_profit <= 0:
        return 0.0
    return test_profit / train_profit


def is_suspicious_result(
    train_profit: float,
    test_profit: float,
    ratio_low: float = 0.7,
    ratio_high: float = 1.5
) -> bool:
    """
    Check if a result is suspicious (likely overfit or data leak).
    
    Args:
        train_profit: Training period final BTC
        test_profit: Test period final BTC
        ratio_low: Lower bound for acceptable ratio
        ratio_high: Upper bound for acceptable ratio
    
    Returns:
        True if result should be flagged as suspicious
    """
    ratio = compute_consistency_ratio(train_profit, test_profit)
    return ratio < ratio_low or ratio > ratio_high


def score_for_ranking(
    oos_profit: float,
    train_profit: float,
    recency_weight: float = 1.0,
    strategy: str = "weighted"
) -> float:
    """
    Score a WFO window for ranking/selection.
    
    Used by wfo_select_best.py to pick the best window.
    
    Strategies:
    - "best_oos": Just OOS profit
    - "weighted": Balanced OOS + consistency + recency
    - "consistent": Harmonic mean of train/test
    - "recent": OOS * recency
    
    Args:
        oos_profit: Out-of-sample profit
        train_profit: In-sample profit
        recency_weight: Weight for recency (normalized 0-1)
        strategy: Ranking strategy name
    
    Returns:
        Score for ranking (higher is better)
    """
    if strategy == "best_oos":
        return oos_profit
    
    elif strategy == "weighted":
        gap = abs(oos_profit - train_profit)
        avg = (oos_profit + train_profit) / 2
        return oos_profit * 0.6 + avg * 0.3 + recency_weight * 0.1 - gap * 0.2
    
    elif strategy == "consistent":
        if oos_profit + train_profit < 1e-9:
            return 0.0
        harmonic = 2 * oos_profit * train_profit / (oos_profit + train_profit)
        gap = abs(oos_profit - train_profit)
        return harmonic - gap * 0.5
    
    elif strategy == "recent":
        return oos_profit * recency_weight
    
    else:
        return oos_profit  # Fallback
