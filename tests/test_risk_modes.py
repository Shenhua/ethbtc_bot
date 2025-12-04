# tests/test_risk_modes.py

import numpy as np
import pandas as pd

from core.ethbtc_accum_bot import Backtester, FeeParams, StratParams, EthBtcStrategy


def _make_trending_price_index():
    """
    Create a synthetic price series:
    - flat, then strong up-trend, then mild mean-reversion.
    Enough structure to trigger trades & risk logic.
    """
    idx = pd.date_range("2024-01-01", periods=300, freq="15min")

    # 0-99: flat around 0.01
    part1 = np.full(100, 0.01)

    # 100-199: linear up from 0.01 to 0.03
    part2 = np.linspace(0.01, 0.03, 100)

    # 200-299: small pullback / chop around 0.025
    rng = np.random.RandomState(123)
    part3 = 0.025 + rng.normal(scale=0.0005, size=100)

    prices = np.concatenate([part1, part2, part3])
    close = pd.Series(prices, index=idx)
    return close


def _default_params():
    """StratParams similar to your production config, but for tests."""
    return StratParams(
        trend_kind="sma",
        trend_lookback=50,
        flip_band_entry=0.02,
        flip_band_exit=0.01,
        vol_window=30,
        vol_adapt_k=0.0,
        bar_interval_minutes=15,
        target_vol=0.0,
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


def _default_fees():
    return FeeParams(
        maker_fee=0.0002,
        taker_fee=0.0004,
        slippage_bps=1.0,
        bnb_discount=0.25,
        pay_fees_in_bnb=False,
    )


def test_dynamic_vs_fixed_basis_give_different_final_btc():
    """
    On a synthetic trending series, dynamic risk mode should behave differently
    from fixed_basis mode when drawdown limits are active.
    We expect final_btc to differ between the two modes.
    """
    close = _make_trending_price_index()
    params = _default_params()
    fees = _default_fees()
    bt = Backtester(fees)

    # Fixed basis: 20% max DD in ABSOLUTE BTC
    res_fixed = bt.simulate(
        close,
        EthBtcStrategy(params),
        initial_btc=1.0,
        bnb_price_series=None,
        max_daily_loss_btc=0.0,
        max_dd_btc=0.20,        # 20% of initial capital, fixed
        max_daily_loss_frac=0.0,
        max_dd_frac=0.0,
        risk_mode="fixed_basis",
    )
    summary_fixed = res_fixed["summary"]

    # Dynamic: 20% max DD as FRACTION of peak equity
    res_dyn = bt.simulate(
        close,
        EthBtcStrategy(params),
        initial_btc=1.0,
        bnb_price_series=None,
        max_daily_loss_btc=0.0,
        max_dd_btc=0.0,
        max_daily_loss_frac=0.0,
        max_dd_frac=0.20,       # 20% of equity_high
        risk_mode="dynamic",
    )
    summary_dyn = res_dyn["summary"]

    final_fixed = summary_fixed["final_btc"]
    final_dyn = summary_dyn["final_btc"]

    # They should differ meaningfully if the risk modes behave differently.
    assert final_dyn != final_fixed
    # As the account grows, dynamic mode typically allows more risk,
    # so final_dyn is often >= final_fixed. Not a hard guarantee, but a sanity check:
    assert final_dyn > final_fixed or summary_dyn["max_drawdown_btc"] > summary_fixed["max_drawdown_btc"]


def test_dynamic_dd_threshold_scales_with_equity():
    """
    In dynamic mode with max_dd_frac > 0, the drawdown allowed in BTC should
    scale with equity_high, and in practice we expect larger max_drawdown_btc
    than in a fixed absolute-DD regime with the same starting notional.
    """
    close = _make_trending_price_index()
    params = _default_params()
    fees = _default_fees()
    bt = Backtester(fees)

    res_fixed = bt.simulate(
        close,
        EthBtcStrategy(params),
        initial_btc=1.0,
        bnb_price_series=None,
        max_daily_loss_btc=0.0,
        max_dd_btc=0.20,        # absolute 0.20 BTC max DD
        max_daily_loss_frac=0.0,
        max_dd_frac=0.0,
        risk_mode="fixed_basis",
    )
    max_dd_fixed = res_fixed["summary"]["max_drawdown_btc"]

    res_dyn = bt.simulate(
        close,
        EthBtcStrategy(params),
        initial_btc=1.0,
        bnb_price_series=None,
        max_daily_loss_btc=0.0,
        max_dd_btc=0.0,
        max_daily_loss_frac=0.0,
        max_dd_frac=0.20,       # 20% of peak equity
        risk_mode="dynamic",
    )
    max_dd_dyn = res_dyn["summary"]["max_drawdown_btc"]

    # In a trending series where equity grows, dynamic mode should permit a
    # larger drawdown in BTC terms than a fixed 0.20 cap.
    assert max_dd_dyn >= max_dd_fixed