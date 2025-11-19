import math

import numpy as np
import pandas as pd
import pytest

from ethbtc_accum_bot import StratParams, EthBtcStrategy


def test_realized_vol_scales_with_interval():
    """
    Check that _realized_vol scales correctly with bar_interval_minutes:
    same returns, different bar interval → ratio of vols matches
    sqrt(bars_per_year_1 / bars_per_year_2).
    """
    # Synthetic returns with some variation so rolling std is non-zero
    ret = pd.Series([0.0, 0.01, -0.01, 0.02, -0.02, 0.015, -0.015, 0.01, -0.01])

    p1 = StratParams(vol_window=3, bar_interval_minutes=1)
    s1 = EthBtcStrategy(p1)
    v1 = s1._realized_vol(ret)

    p2 = StratParams(vol_window=3, bar_interval_minutes=15)
    s2 = EthBtcStrategy(p2)
    v2 = s2._realized_vol(ret)

    # Use only points where both are defined
    mask = v1.notna() & v2.notna()
    v1_valid = v1[mask]
    v2_valid = v2[mask]

    # Both are rolling std * sqrt(bars_per_year); ratio should match sqrt of bars_per_year ratio
    bars_per_year_1 = 365 * 24 * 60 / 1
    bars_per_year_15 = 365 * 24 * 60 / 15
    expected_ratio = math.sqrt(bars_per_year_1 / bars_per_year_15)

    # Allow tiny numerical tolerance
    observed_ratio = (v1_valid / v2_valid).mean()
    assert observed_ratio == pytest.approx(expected_ratio, rel=1e-6)


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
    with the extended metrics, and those values should be finite numbers for
    a simple, well-behaved price series.
    """
    from ethbtc_accum_bot import Backtester, FeeParams

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

    # Check presence of new keys
    for key in [
        "initial_btc",
        "final_btc",
        "total_return",
        "ann_return",
        "ann_vol",
        "sharpe",
        "max_drawdown_btc",
        "max_drawdown_pct",
        "n_bars",
        "n_trades",
    ]:
        assert key in summary

    # Check numeric fields are finite
    numeric_keys = [
        "initial_btc",
        "final_btc",
        "total_return",
        "ann_return",
        "ann_vol",
        "sharpe",
        "max_drawdown_btc",
        "max_drawdown_pct",
    ]
    for key in numeric_keys:
        val = float(summary[key])
        assert math.isfinite(val)