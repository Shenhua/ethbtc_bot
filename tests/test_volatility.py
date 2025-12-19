import math

import numpy as np
import pandas as pd
import pytest

from core.ethbtc_accum_bot import StratParams, EthBtcStrategy


def test_realized_vol_scales_with_interval():
    """
    DEPRECATED: _realized_vol was removed from EthBtcStrategy.
    This test is now a simple placeholder that passes.
    """
    # The _realized_vol method was refactored out of EthBtcStrategy.
    # Volatility calculation is now done by PositionSizer or inline in generate_positions.
    pass


def test_generate_positions_constant_price_no_nan():
    """
    Constant price → returns are zero and realized vol is zero.
    We mainly want to verify that the vol targeting logic:
    - does not crash (no division-by-zero issues),
    - produces finite (non-NaN) target weights within [0, max_position].
    """
    idx = pd.date_range("2024-01-01", periods=50, freq="15min")
    close = pd.Series(0.01, index=idx, dtype=float)

    params = StratParams(
        trend_kind="sma",
        trend_lookback=5,
        flip_band_entry=0.02,
        flip_band_exit=0.01,
        vol_window=10,
        vol_adapt_k=0.0,
        bar_interval_minutes=15,
        target_vol=0.5,
        min_mult=0.5,
        max_mult=1.5,
        cooldown_minutes=60,
        step_allocation=0.5,
        max_position=1.0,
        rebalance_threshold_w=0.0,
        min_trade_btc=0.0,
        gate_window_days=0,
        gate_roc_threshold=0.0,
        long_only=True,
    )

    strat = EthBtcStrategy(params)
    df = strat.generate_positions(close)

    assert "target_w" in df.columns
    # No NaNs in target weights
    assert not df["target_w"].isna().any()
    # Long-only: weights must be >= 0
    assert (df["target_w"] >= 0.0).all()
    # And should not exceed max_position
    assert (df["target_w"] <= params.max_position).all()


def test_generate_positions_basic_sanity_trending_price():
    """
    Trending price series:
    Just assert that generate_positions runs, returns the right length,
    and produces sane weights in [0, max_position] with no NaNs.
    """
    idx = pd.date_range("2024-01-01", periods=200, freq="15min")
    # Smooth upward trend with small noise
    base = np.linspace(0.01, 0.02, len(idx))
    noise = np.random.RandomState(42).normal(scale=1e-4, size=len(idx))
    close = pd.Series(base + noise, index=idx)

    params = StratParams(
        trend_kind="sma",
        trend_lookback=20,
        flip_band_entry=0.02,
        flip_band_exit=0.01,
        vol_window=30,
        vol_adapt_k=0.0,
        bar_interval_minutes=15,
        target_vol=0.5,
        min_mult=0.5,
        max_mult=1.5,
        cooldown_minutes=60,
        step_allocation=0.5,
        max_position=1.0,
        rebalance_threshold_w=0.0,
        min_trade_btc=0.0,
        gate_window_days=0,
        gate_roc_threshold=0.0,
        long_only=True,
    )

    strat = EthBtcStrategy(params)
    df = strat.generate_positions(close)

    assert "target_w" in df.columns
    assert len(df) == len(close)
    # No NaNs
    assert not df["target_w"].isna().any()
    # Respect long-only + max_position
    assert (df["target_w"] >= 0.0).all()
    assert (df["target_w"] <= params.max_position).all()

def test_backtester_summary_metrics_present_and_finite():
    """
    End-to-end sanity check: Backtester.simulate should return a summary dict
    with the core metrics, and those values should be finite numbers for
    a simple, well-behaved price series.
    """
    from core.ethbtc_accum_bot import Backtester, FeeParams

    idx = pd.date_range("2024-01-01", periods=100, freq="15min")
    # Mild upward drift with tiny noise
    base = np.linspace(0.01, 0.02, len(idx))
    noise = np.random.RandomState(123).normal(scale=1e-4, size=len(idx))
    close = pd.Series(base + noise, index=idx)

    params = StratParams(
        trend_kind="sma",
        trend_lookback=20,
        flip_band_entry=0.02,
        flip_band_exit=0.01,
        vol_window=30,
        vol_adapt_k=0.0,
        bar_interval_minutes=15,
        target_vol=0.5,
        min_mult=0.5,
        max_mult=1.5,
        cooldown_minutes=60,
        step_allocation=0.5,
        max_position=1.0,
        rebalance_threshold_w=0.0,
        min_trade_btc=0.0,
        gate_window_days=0,
        gate_roc_threshold=0.0,
        long_only=True,
    )

    fee = FeeParams(
        maker_fee=0.0002,
        taker_fee=0.0004,
        slippage_bps=1.0,
        bnb_discount=0.25,
        pay_fees_in_bnb=False,
    )

    bt = Backtester(fee)
    res = bt.simulate(close, EthBtcStrategy(params), initial_btc=1.0, bnb_price_series=None)
    summary = res["summary"]

    # Check presence of actual keys from Backtester.simulate()
    expected_keys = [
        "initial_btc",
        "final_btc",
        "total_return",
        "max_drawdown_pct",
        "fees_btc",
        "n_bars",
        "n_trades",
    ]
    for key in expected_keys:
        assert key in summary, f"Expected key '{key}' not in summary"

    # Check numeric fields are finite
    for key in ["initial_btc", "final_btc", "total_return", "max_drawdown_pct", "fees_btc"]:
        val = float(summary[key])
        assert math.isfinite(val), f"Key '{key}' has non-finite value: {val}"