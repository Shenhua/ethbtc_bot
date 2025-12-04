#!/usr/bin/env python3
"""
sanity_check_config.py

Run a full backtest over a long dataset with a given config and print summary stats.

Usage example:

    python sanity_check_config.py \
        --data data/ETHBTC_15m_full.csv \
        --config configs/selected_params.json \
        --funding-data data/ETHBTC_funding.csv
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from typing import Optional

import pandas as pd

# --- MAGIC PATH FIX ---
# Allow importing 'core' even if running from tools/ folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ----------------------


from core.ethbtc_accum_bot import (
    load_vision_csv,
    Backtester,
    StratParams,
    EthBtcStrategy,
    FeeParams,
)

from core.trend_strategy import TrendParams, TrendStrategy  # type: ignore
from core.meta_strategy import MetaStrategy  # type: ignore
from core.config_schema import load_config  # type: ignore


def _interval_to_minutes(interval: str) -> int:
    mapping = {
        "1m": 1,
        "3m": 3,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "2h": 120,
        "4h": 240,
        "6h": 360,
        "8h": 480,
        "12h": 720,
        "1d": 1440,
    }
    return mapping.get(interval, 15)


def build_strategy_from_config(app_cfg, df: pd.DataFrame):
    """
    app_cfg: config_schema.AppConfig
    df: OHLC dataframe (with 'close' column and datetime index)
    """
    strat_cfg = app_cfg.strategy
    exec_cfg = app_cfg.execution

    interval_str = exec_cfg.interval
    bar_minutes = _interval_to_minutes(interval_str)

    # --- Mean Reversion params (StratParams) ---
    common_kwargs = dict(
        trend_kind=strat_cfg.trend_kind,
        trend_lookback=strat_cfg.trend_lookback,
        flip_band_entry=strat_cfg.flip_band_entry,
        flip_band_exit=strat_cfg.flip_band_exit,
        vol_window=strat_cfg.vol_window,
        vol_adapt_k=strat_cfg.vol_adapt_k,
        bar_interval_minutes=bar_minutes,
        target_vol=strat_cfg.target_vol,
        min_mult=strat_cfg.min_mult,
        max_mult=strat_cfg.max_mult,
        cooldown_minutes=strat_cfg.cooldown_minutes,
        step_allocation=strat_cfg.step_allocation,
        max_position=strat_cfg.max_position,
        long_only=strat_cfg.long_only,
        rebalance_threshold_w=strat_cfg.rebalance_threshold_w,
        min_trade_btc=exec_cfg.min_trade_btc or 0.0,
        gate_window_days=strat_cfg.gate_window_days,
        gate_roc_threshold=strat_cfg.gate_roc_threshold,
        profit_lock_dd=strat_cfg.profit_lock_dd,
        vol_scaled_step=strat_cfg.vol_scaled_step,
        funding_limit_long=strat_cfg.funding_limit_long,
        funding_limit_short=strat_cfg.funding_limit_short,
        fast_period=strat_cfg.fast_period,
        slow_period=strat_cfg.slow_period,
        ma_type=strat_cfg.ma_type,
        adx_threshold=strat_cfg.adx_threshold,
        strategy_type=strat_cfg.strategy_type,
    )

    s_type = strat_cfg.strategy_type

    if s_type == "trend":
        # Build a pure TrendStrategy
        tp = TrendParams(
            fast_period=strat_cfg.fast_period,
            slow_period=strat_cfg.slow_period,
            ma_type=strat_cfg.ma_type,
            cooldown_minutes=strat_cfg.cooldown_minutes,
            step_allocation=strat_cfg.step_allocation,
            max_position=strat_cfg.max_position,
            long_only=strat_cfg.long_only,
            funding_limit_long=strat_cfg.funding_limit_long,
            funding_limit_short=strat_cfg.funding_limit_short,
        )
        strat = TrendStrategy(tp)
        return strat

    if s_type == "meta":
        # Meta uses both MR and Trend
        mr_p = StratParams(**common_kwargs)

        tp = TrendParams(
            fast_period=strat_cfg.fast_period,
            slow_period=strat_cfg.slow_period,
            ma_type=strat_cfg.ma_type,
            cooldown_minutes=strat_cfg.cooldown_minutes,
            step_allocation=strat_cfg.step_allocation,
            max_position=strat_cfg.max_position,
            long_only=strat_cfg.long_only,
            funding_limit_long=strat_cfg.funding_limit_long,
            funding_limit_short=strat_cfg.funding_limit_short,
        )
        tr_p = TrendParams(**tp.__dict__)
        strat = MetaStrategy(mr_p, tr_p, adx_threshold=strat_cfg.adx_threshold)
        return strat

    # Default: mean_reversion
    mr_p = StratParams(**common_kwargs)
    strat = EthBtcStrategy(mr_p)
    return strat


def load_funding_series(path: Optional[str], ref_index: pd.DatetimeIndex) -> Optional[pd.Series]:
    if not path:
        return None
    f_df = pd.read_csv(path)
    if "time" not in f_df.columns:
        raise ValueError("Funding CSV must have a 'time' column.")
    f_df["time"] = pd.to_datetime(f_df["time"], utc=True, format="mixed")
    f_df = f_df.set_index("time").sort_index()
    if "rate" not in f_df.columns:
        raise ValueError("Funding CSV must have a 'rate' column.")
    funding = f_df["rate"].reindex(ref_index).ffill().fillna(0.0)
    return funding


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to OHLC CSV (vision format).")
    ap.add_argument("--config", required=True, help="JSON config file (nested or legacy flat).")
    ap.add_argument("--funding-data", help="Optional funding CSV with 'time' and 'rate' columns.")
    ap.add_argument("--start-bnb", type=float, default=0.05, help="Initial BNB balance for fees.")
    args = ap.parse_args()

    # 1. Load candles
    df = load_vision_csv(args.data)
    if "close" not in df.columns:
        raise ValueError("Input data must have a 'close' column.")
    df = df.sort_index()

    # 2. Load config via pydantic (handles nested + legacy)
    app_cfg = load_config(args.config)

    # 3. Build strategy from config
    strat = build_strategy_from_config(app_cfg, df)

    # 4. Build fees and risk from config
    fees_cfg = app_cfg.fees
    fee = FeeParams(
        maker_fee=fees_cfg.maker_fee,
        taker_fee=fees_cfg.taker_fee,
        slippage_bps=fees_cfg.slippage_bps,
        bnb_discount=fees_cfg.bnb_discount,
        pay_fees_in_bnb=fees_cfg.pay_fees_in_bnb,
    )

    risk_cfg = app_cfg.risk
    # If basis_btc is 0.0 (default), fall back to 1.0 to match your CLI behaviour
    basis = risk_cfg.basis_btc if risk_cfg.basis_btc > 0 else 1.0
    max_daily_loss_btc = risk_cfg.max_daily_loss_btc
    max_dd_btc = risk_cfg.max_dd_btc
    max_daily_loss_frac = risk_cfg.max_daily_loss_frac
    max_dd_frac = risk_cfg.max_dd_frac
    risk_mode = risk_cfg.risk_mode

    # 5. Funding (optional)
    funding_series = load_funding_series(args.funding_data, df.index)

    # 6. Run backtest
    bt = Backtester(fee)
    res = bt.simulate(
        df["close"],
        strat,
        funding_series=funding_series,
        full_df=df,
        initial_btc=basis,
        max_daily_loss_btc=max_daily_loss_btc,
        max_dd_btc=max_dd_btc,
        max_daily_loss_frac=max_daily_loss_frac,
        max_dd_frac=max_dd_frac,
        risk_mode=risk_mode,
    )

    summary = res.get("summary", {})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()