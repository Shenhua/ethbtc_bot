#!/usr/bin/env python3
"""
ethbtc_accum_bot.py — v5.3 (Optimized & Fixed)
"""

from __future__ import annotations
import sys, os

# --- MAGIC PATH FIX ---
# Allows importing 'core' modules even if running this script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ----------------------

import math
import argparse
import json
import random
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Any, Union
import pandas as pd
import numpy as np

log = logging.getLogger("ethbtc_accum_bot")

# --- Strategy Imports (Safe) ---
try:
    from core.alert_manager import AlertManager
    from core.story_writer import StoryWriter
except ImportError:
    pass

from core.config_schema import load_config, AppConfig

# ------------------ Loaders ------------------

def load_json_config(path: Optional[str]) -> Dict:
    """
    Legacy helper: Loads a JSON file and flattens the structure for easy access.
    Required by optimizer_cli.py and other tools.
    """
    if not path: return {}
    with open(path, "r") as f: data = json.load(f)
    if not isinstance(data, dict): return {}
    flat: Dict[str, Any] = dict(data)
    # Flatten specific blocks for legacy script compatibility
    for block in ("params", "fees", "strategy", "execution", "risk"):
        block_dict = data.get(block)
        if isinstance(block_dict, dict): flat.update(block_dict)
    return flat

def load_vision_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    alias = {"opentime":"open_time","closetime":"close_time"}
    for k,v in alias.items():
        if k in df.columns and v not in df.columns: df.rename(columns={k:v}, inplace=True)
    
    if "close_time" not in df.columns:
        if "date" in df.columns: df.rename(columns={"date":"close_time"}, inplace=True)
        else: raise ValueError("close_time column not found")

    def _parse_dt(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s):
            vmax = float(pd.to_numeric(s, errors="coerce").dropna().head(1).max() or 0)
            unit = "ms" if vmax > 1e11 else "s"
            return pd.to_datetime(s, unit=unit, utc=True)
        return pd.to_datetime(s, utc=True, errors="coerce")

    if "open_time" in df.columns: df["open_time"] = _parse_dt(df["open_time"])
    df["close_time"] = _parse_dt(df["close_time"])

    for c in ["open","high","low","close","volume"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    
    df = df.dropna(subset=["close"]).set_index("close_time")
    
    # Hardened Index Cleaning
    df = df[df.index.notna()]
    df = df[~df.index.duplicated(keep='first')]
    
    return df.sort_index()

def _write_excel(path: str, sheets: dict):
    """Helper for multi-interval summary tools."""
    import pandas as _pd
    with _pd.ExcelWriter(path, engine="xlsxwriter") as w:
        for name, obj in sheets.items():
            if isinstance(obj, _pd.DataFrame):
                df = obj.copy()
                if isinstance(df.index, _pd.DatetimeIndex):
                    df.index = df.index.tz_convert(None)
                for c in df.columns:
                    if _pd.api.types.is_datetime64_any_dtype(df[c]):
                        df[c] = df[c].dt.tz_convert(None)
                df.to_excel(w, sheet_name=str(name)[:31])
            else:
                _pd.DataFrame([obj]).to_excel(w, sheet_name=str(name)[:31], index=False)

# ------------------ Params & Strategy ------------------

@dataclass
class StratParams:
    # Mean Reversion
    trend_kind: str = "sma"
    trend_lookback: int = 200
    flip_band_entry: float = 0.025
    flip_band_exit: float = 0.015
    vol_window: int = 60
    vol_adapt_k: float = 0.0
    bar_interval_minutes: int = 15
    target_vol: float = 0.5
    min_mult: float = 0.5
    max_mult: float = 1.5
    cooldown_minutes: int = 180
    step_allocation: float = 0.5
    max_position: float = 1.0
    long_only: bool = True        
    rebalance_threshold_w: float = 0.0
    min_trade_btc: float = 0.0
    gate_window_days: int = 60
    gate_roc_threshold: float = 0.0
    profit_lock_dd: float = 0.0
    vol_scaled_step: bool = False
    
    # Dynamic Position Sizing
    position_sizing_mode: str = "static"  # "static", "volatility", "kelly"
    position_sizing_target_vol: float = 0.5
    position_sizing_min_step: float = 0.1
    position_sizing_max_step: float = 1.0
    # Kelly Criterion
    kelly_win_rate: float = 0.55
    kelly_avg_win: float = 0.02
    kelly_avg_loss: float = 0.01

    # Funding & Trend (New)
    funding_limit_long: float = 0.05
    funding_limit_short: float = -0.05
    fast_period: int = 50
    slow_period: int = 200
    ma_type: str = "ema"
    adx_threshold: float = 25.0
    strategy_type: str = "mean_reversion"

@dataclass
class FeeParams:
    maker_fee: float = 0.0002
    taker_fee: float = 0.0004
    slippage_bps: float = 1.0
    bnb_discount: float = 0.25
    pay_fees_in_bnb: bool = True

class EthBtcStrategy:
    def __init__(self, p: StratParams): self.p = p

    def generate_positions(self, close: pd.Series, funding: Optional[pd.Series] = None) -> pd.DataFrame:
        # --- FIX ITEM 4: Optimized Vectorized Calculation ---
        
        # 1. Indicators
        if self.p.trend_kind == "sma":
            ma = close.rolling(self.p.trend_lookback).mean()
            ratio = close / ma - 1.0
        else:
            ratio = (close / close.shift(self.p.trend_lookback)) - 1.0
        
        log.debug(f"[STRATEGY] Current ratio: {ratio.iloc[-1]:.4f}")

        ret = close.pct_change(fill_method=None).fillna(0)
        bars_per_year = (365 * 24 * 60) / float(self.p.bar_interval_minutes)
        vol = ret.rolling(self.p.vol_window).std() * math.sqrt(bars_per_year)
        log.debug(f"[STRATEGY] Current volatility: {vol.iloc[-1]:.4f}")
        
        adj = self.p.vol_adapt_k * (vol.fillna(vol.median()))
        band_entry = self.p.flip_band_entry + adj
        band_exit  = self.p.flip_band_exit + adj

        # 2. Gate Logic
        gate_buy_mask = pd.Series(True, index=close.index)
        gate_sell_mask = pd.Series(True, index=close.index)

        if self.p.gate_window_days > 0:
            daily = close.resample("1D").last()
            roc_daily = daily.pct_change(self.p.gate_window_days, fill_method=None)            
            roc = roc_daily.reindex(close.index).ffill().fillna(0.0)
            
            gate_buy_mask = roc <= -self.p.gate_roc_threshold
            gate_sell_mask = roc >= self.p.gate_roc_threshold

        # 3. Funding Logic
        allow_buy = pd.Series(True, index=close.index)
        allow_sell = pd.Series(True, index=close.index)
        
        if funding is not None:
            f_aligned = funding.reindex(close.index).ffill().fillna(0.0)
            allow_buy = f_aligned <= self.p.funding_limit_long
            allow_sell = f_aligned >= self.p.funding_limit_short

        # 4. State Machine (Vectorized Loop Optimization)
        # Instead of iterating .loc/iloc (slow), we use numpy arrays.
        
        # Prepare arrays
        r_arr = ratio.values
        be_arr = band_entry.values
        bx_arr = band_exit.values
        gb_arr = gate_buy_mask.values
        gs_arr = gate_sell_mask.values
        ab_arr = allow_buy.values
        as_arr = allow_sell.values
        idx_arr = close.index
        
        out_sig = np.zeros(len(close))
        state = 0.0 # Default Start State
        
        # Cooldown logic requires timestamps, convert to int nanoseconds for speed comparison
        ts_arr = idx_arr.astype(np.int64)
        cooldown_ns = self.p.cooldown_minutes * 60 * 1_000_000_000
        last_flip_ts = ts_arr[0]

        for i in range(len(close)):
            t = ts_arr[i]
            
            if (t - last_flip_ts) < cooldown_ns:
                out_sig[i] = state
                continue

            r = r_arr[i]
            be = be_arr[i]
            bx = bx_arr[i]
            
            desired = state
            
            # Buy Logic
            if state <= 0:
                if r < -be and gb_arr[i] and ab_arr[i]:
                    desired = 1.0
            
            # Sell Logic
            if state >= 0:
                if r > be and gs_arr[i] and as_arr[i]:
                     desired = -1.0 
                elif r > -bx and state > 0:
                     desired = -1.0

            if desired != state:
                state = desired
                last_flip_ts = t
            
            out_sig[i] = state

        # 5. Volatility Scaling
        mult = pd.Series(1.0, index=close.index)
        if self.p.target_vol > 0:
            vol_adj = vol.replace(0, np.nan)
            mult = (self.p.target_vol / vol_adj).clip(self.p.min_mult, self.p.max_mult).fillna(self.p.min_mult)

        # 6. Final Allocation
        sig_series = pd.Series(out_sig, index=close.index)
        lo = 0.0 if self.p.long_only else -self.p.max_position
        target_w = (sig_series * mult).clip(lo, self.p.max_position)
        
        return pd.DataFrame({"target_w": target_w})

class Backtester:
    def __init__(self, fee: FeeParams): self.fee = fee

    def simulate(self, close: pd.Series, 
                 strategy, 
                 funding_series: Optional[pd.Series] = None,
                 bnb_price_series: Optional[pd.Series] = None,
                 full_df: Optional[pd.DataFrame] = None,
                 story_writer=None,
                 initial_btc: float = 1.0, 
                 start_bnb: float = 0.05,
                 max_daily_loss_btc=0.0, 
                 max_dd_btc=0.0,
                 max_daily_loss_frac=0.0, 
                 max_dd_frac=0.0, 
                 risk_mode="fixed_basis",
                 drawdown_reset_days=0.0, 
                 drawdown_reset_score=0.0,
                 base_asset="ETH",
                 quote_asset="BTC"):
        """
        Simulate trading strategy on historical price data.
        
        Args:
            close: Series of closing prices
            strategy: Strategy instance  
            funding_series: Optional funding rate series
            bnb_price_series: Optional BNB price series
            full_df: Optional full OHLC dataframe
            story_writer: Optional StoryWriter for real-time narrative
            initial_btc: Starting BTC balance
            start_bnb: Starting BNB balance
            max_daily_loss_btc: Daily loss limit in BTC
            max_dd_btc: Max drawdown in BTC
            max_daily_loss_frac: Daily loss limit as fraction
            max_dd_frac: Max drawdown as fraction
            risk_mode: Risk mode (fixed_basis or dynamic)
            drawdown_reset_days: Days to wait before phoenix
            drawdown_reset_score: ADX score needed for phoenix
            base_asset: Name of base asset (e.g. ETH)
            quote_asset: Name of quote asset (e.g. BTC)
        """
        px = close.astype(float).copy()
        
        # Align Funding
        aligned_funding = None
        if funding_series is not None:
            aligned_funding = funding_series.reindex(close.index).ffill().fillna(0.0)

        # Generate Positions
        if hasattr(strategy, 'adx_threshold'): # MetaStrategy
            if full_df is None: raise ValueError("MetaStrategy requires full OHLC dataframe (full_df).")
            plan = strategy.generate_positions(full_df, funding=aligned_funding)
        elif hasattr(strategy, 'generate_positions'):
            if isinstance(strategy, EthBtcStrategy):
                plan = strategy.generate_positions(px, funding=aligned_funding)
            else:
                input_data = full_df if full_df is not None else px
                plan = strategy.generate_positions(input_data, funding=aligned_funding)
        
        target_w = plan["target_w"]

        # === DYNAMIC POSITION SIZING (matching live_executor.py) ===
        from core.position_sizer import PositionSizer, PositionSizerConfig
        
        step_mr = 1.0
        thresh_mr = 0.0
        step_trend = 1.0
        thresh_trend = 0.0
        adx_cutoff = 25.0
        is_meta = False

        if hasattr(strategy, 'adx_threshold'): # MetaStrategy
            is_meta = True
            adx_cutoff = strategy.adx_threshold
            
            # Get thresholds (static)
            thresh_mr = getattr(strategy.mr.p, 'rebalance_threshold_w', 0.0)
            thresh_trend = 0.0
            
            step_mr =  getattr(strategy.mr.p, 'step_allocation', 1.0)
            step_trend = getattr(strategy.trend.p, 'step_allocation', 1.0)
            # Dynamic step sizing will be calculated per-bar below
            # (we initialize to base values here, but will override in loop)
            
        elif hasattr(strategy, 'p'):
             thresh_mr = getattr(strategy.p, 'rebalance_threshold_w', 0.0)
             thresh_trend = thresh_mr


        # --- Execution Loop ---
        btc = np.zeros(len(px))
        eth = np.zeros(len(px))
        bnb = np.zeros(len(px))
        
        btc[0] = initial_btc 
        bnb[0] = start_bnb
        
        taker_fee = self.fee.taker_fee
        fee_disc = (1.0 - self.fee.bnb_discount) if self.fee.pay_fees_in_bnb else 1.0
        
        total_fees_btc = 0.0
        total_turnover = 0.0
        trades = []

        cur_w = 0.0
        
        # Risk tracking variables
        equity_high = initial_btc
        maxdd_hit = False
        maxdd_hit_ts = None
        
        # Story tracking
        last_regime = None
        
        # === Story Logging Setup ===
        if story_writer:
            strategy_name = type(strategy).__name__ if hasattr(strategy, '__class__') else "Strategy"
            mode = f"BACKTEST-{strategy_name}"
            story_writer.log_startup(px.index[0], initial_btc, mode, quote_asset)
        
        
        # --- FIX ITEM 15: Daily Risk Tracking Variables ---
        current_day = px.index[0].date()
        day_start_wealth = initial_btc
        daily_limit_hit = False
        
        for i in range(1, len(px)):
            price = float(px.iat[i])
            timestamp = px.index[i]
            
            # Carry forward balances
            btc[i] = btc[i-1]
            eth[i] = eth[i-1]
            bnb[i] = bnb[i-1]
            
            # Funding Fees
            if aligned_funding is not None and abs(eth[i-1]) > 0:
                if timestamp.hour % 8 == 0 and timestamp.minute == 0:
                    rate = float(aligned_funding.iat[i])
                    funding_cost = eth[i-1] * price * rate
                    btc[i] -= funding_cost

            # Wealth Calculation
            wealth = btc[i] + eth[i] * price
            
            # Story: Check for ATH
            if story_writer:
                story_writer.check_ath(timestamp, wealth, quote_asset)
            
            # --- FIX ITEM 15: Daily Loss Logic ---
            if timestamp.date() != current_day:
                # Story: Log daily summary
                if story_writer:
                    daily_pnl = wealth - day_start_wealth
                    story_writer.check_and_log_daily(timestamp, daily_pnl, wealth, quote_asset)
                    story_writer.check_and_log_weekly(timestamp, wealth, quote_asset)
                    story_writer.check_and_log_monthly(timestamp, wealth, quote_asset)
                    story_writer.check_and_log_annual(timestamp, wealth, quote_asset)
                
                current_day = timestamp.date()
                day_start_wealth = wealth # Reset for new day
                daily_limit_hit = False
            
            if not daily_limit_hit and max_daily_loss_btc > 0:
                day_loss = day_start_wealth - wealth
                if day_loss >= max_daily_loss_btc:
                    daily_limit_hit = True

            # Max Drawdown Logic
            if not maxdd_hit:
                if wealth > equity_high: equity_high = wealth
            
            if not maxdd_hit and max_dd_frac > 0.0 and equity_high > 0:
                dd = (equity_high - wealth) / equity_high
                if dd >= max_dd_frac: 
                    maxdd_hit = True
                    maxdd_hit_ts = timestamp
                    # Story: Log safety breaker
                    if story_writer:
                        story_writer.log_safety_breaker(timestamp, dd)
            
            # Phoenix Reset
            if maxdd_hit and drawdown_reset_days > 0:
                time_passed = timestamp - maxdd_hit_ts
                current_score = 0.0
                if "regime_score" in plan.columns:
                    current_score = float(plan["regime_score"].iat[i])
                
                if time_passed.days >= drawdown_reset_days and current_score >= drawdown_reset_score:
                    maxdd_hit = False
                    equity_high = wealth
                    # Story: Log phoenix activation
                    if story_writer:
                        story_writer.log_phoenix_activation(timestamp, current_score, drawdown_reset_days)
            
            # --- Rebalancing Logic ---
            # Target weight at this bar
            tw = float(target_w.iat[i]) if not np.isnan(float(target_w.iat[i])) else 0.0
            
            # === DYNAMIC STEP SIZING (per bar, matching live_executor.py) ===
            # Get realized volatility at this bar (if available)
            rv_at_i = float(plan["vol"].iat[i]) if "vol" in plan.columns and not np.isnan(float(plan["vol"].iat[i])) else 0.5
            
            # Calculate dynamic step based on strategy and volatility
            active_step = 1.0
            active_thresh = 0.0

            if is_meta:
                # MetaStrategy: use MR or Trend params based on regime
                regime_val = float(plan["regime_state"].iat[i]) if "regime_state" in plan.columns else -1.0
                
                if regime_val < 0:  # Mean Reversion mode
                    sizer_config = PositionSizerConfig(
                        mode=getattr(strategy.mr.p, 'position_sizing_mode', 'static'),
                        base_step=getattr(strategy.mr.p, 'step_allocation', 1.0),
                        target_vol=getattr(strategy.mr.p, 'position_sizing_target_vol', 0.5),
                        min_step=getattr(strategy.mr.p, 'position_sizing_min_step', 0.1),
                        max_step=getattr(strategy.mr.p, 'position_sizing_max_step', 1.0),
                    )
                    active_step = PositionSizer(sizer_config).calculate_step(rv_at_i)
                    active_thresh = thresh_mr
                else:  # Trend mode
                    tr_config = PositionSizerConfig(
                        mode=getattr(strategy.trend.p, 'position_sizing_mode', 'static'),
                        base_step=getattr(strategy.trend.p, 'step_allocation', 1.0),
                        target_vol=getattr(strategy.trend.p, 'position_sizing_target_vol', 0.5),
                        min_step=getattr(strategy.trend.p, 'position_sizing_min_step', 0.1),
                        max_step=getattr(strategy.trend.p, 'position_sizing_max_step', 1.0),
                    )
                    active_step = PositionSizer(tr_config).calculate_step(rv_at_i)
                    active_thresh = thresh_trend
            else:
                # Single strategy
                single_config = PositionSizerConfig(
                    mode=getattr(strategy.p, 'position_sizing_mode', 'static'),
                    base_step=getattr(strategy.p, 'step_allocation', 1.0),
                    target_vol=getattr(strategy.p, 'position_sizing_target_vol', 0.5),
                    min_step=getattr(strategy.p, 'position_sizing_min_step', 0.1),
                    max_step=getattr(strategy.p, 'position_sizing_max_step', 1.0),
                )
                active_step = PositionSizer(single_config).calculate_step(rv_at_i)
                active_thresh = thresh_mr
          
            # --- DYNAMIC PARAM SELECTION ---
            step = active_step
            thresh = active_thresh
            
            if is_meta and "regime_score" in plan.columns:
                score = float(plan["regime_score"].iat[i])
                
                # Story: Log regime switch
                if story_writer:
                    story_writer.check_regime_switch(timestamp, score, adx_cutoff, "meta")
                
                if score > adx_cutoff:
                    step = step_trend
                    thresh = thresh_trend
            # -------------------------------

            # Rebalance Logic
            new_w = cur_w + step * (tw - cur_w)
            
            # === SNAP-TO-ZERO (matching live_executor.py) ===
            # Force full exit for small positions to clean up dust
            if tw == 0.0 and abs(eth[i]) > 0:
                eth_val_quote = abs(eth[i]) * price
                min_trade_quote = 0.0001  # Minimum trade size in quote currency
                # If position is small but not tiny, force full exit
                if eth_val_quote > min_trade_quote and eth_val_quote < (3.0 * min_trade_quote):
                    new_w = 0.0  # Force full exit (overrides smoothing)
            
            # Threshold check
            if abs(new_w - cur_w) < thresh:
                new_w = cur_w

            target_eth = new_w * wealth / price
            delta = target_eth - eth[i]
            
            # === MIN TRADE SIZE CHECK (matching live_executor.py) ===
            min_trade_btc = 0.0001  # Configurable minimum trade size
            trade_value_btc = abs(delta * price)
            
            # Execution (only if trade meets minimum size)
            if trade_value_btc >= min_trade_btc:
                notional = trade_value_btc
                f_rate = taker_fee * fee_disc 
                fee_val = notional * f_rate
                
                if self.fee.pay_fees_in_bnb and bnb_price_series is not None:
                    bnb_px = bnb_price_series.iat[i]
                    if bnb_px > 0:
                        bnb_cost = fee_val / bnb_px
                        if bnb[i] >= bnb_cost:
                            bnb[i] -= bnb_cost
                        else:
                            # Not enough BNB, pay in BTC
                            btc[i] -= fee_val
                else:
                    btc[i] -= fee_val
                
                # Apply trade
                eth[i] += delta
                btc[i] -= delta * price
                if delta > 0: # Buy
                    trades.append({"time": px.index[i], "side":"BUY", "price":price, "qty":delta, "fee":fee_val})
                    # Story: Log trade
                    if story_writer:
                        story_writer.log_trade(timestamp, "BUY", delta, price, base_asset, quote_asset)
                else: # Sell
                    trades.append({"time": px.index[i], "side":"SELL", "price":price, "qty":delta, "fee":fee_val})
                    # Story: Log trade
                    if story_writer:
                        story_writer.log_trade(timestamp, "SELL", delta, price, base_asset, quote_asset)
                
                total_fees_btc += fee_val
                total_turnover += notional

            cur_w = (eth[i] * price) / max(wealth, 1e-12)

        final_btc = btc[-1] + eth[-1] * float(px.iat[-1])
        
        port_df = pd.DataFrame({"wealth_btc": btc + eth*px}, index=px.index)
        
        summary = {
            "initial_btc": initial_btc,
            "final_btc": final_btc,
            "total_return": (final_btc / initial_btc) - 1.0,
            "max_drawdown_pct": (equity_high - final_btc)/equity_high if equity_high > 0 else 0.0,
            "fees_btc": total_fees_btc,
            "turnover_btc": total_turnover,
            "n_trades": len(trades),
            "n_bars": len(px)
        }
        
        # Return
        return {
            "summary": summary,
            "portfolio": port_df,
            "balances": pd.DataFrame({"btc": btc, "eth": eth, "bnb": bnb}, index=px.index),
            "trades": pd.DataFrame(trades),
            "diagnostics": plan if hasattr(strategy, 'generate_positions') else None
        }

# ------------------ CLI ------------------
def _interval_to_minutes(interval: str) -> int:
    mapping = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480,
        "12h": 720, "1d": 1440,
    }
    return mapping.get(interval, 15)

def load_funding_series(path: Optional[str], ref_index: pd.DatetimeIndex) -> Optional[pd.Series]:
    if not path: return None
    f_df = pd.read_csv(path)
    if "time" not in f_df.columns: raise ValueError("Funding CSV must have 'time'")
    f_df["time"] = pd.to_datetime(f_df["time"], utc=True, format="mixed")
    f_df = f_df.set_index("time").sort_index()
    if "rate" not in f_df.columns: raise ValueError("Funding CSV must have 'rate'")
    funding = f_df["rate"].reindex(ref_index).ffill().fillna(0.0)
    return funding

def build_strategy_from_config(app_cfg, df: pd.DataFrame):
    strat_cfg = app_cfg.strategy
    exec_cfg = app_cfg.execution
    interval_str = str(strat_cfg.strategy_type and exec_cfg.interval)
    bar_minutes = _interval_to_minutes(interval_str)
    
    # Local imports to avoid circular dependency
    from core.trend_strategy import TrendStrategy, TrendParams
    from core.meta_strategy import MetaStrategy
    
    # Common parameters shared across MR and Trend
    common_kwargs = dict(
        trend_kind=strat_cfg.trend_kind, trend_lookback=strat_cfg.trend_lookback,
        flip_band_entry=strat_cfg.flip_band_entry, flip_band_exit=strat_cfg.flip_band_exit,
        vol_window=strat_cfg.vol_window, vol_adapt_k=strat_cfg.vol_adapt_k,
        bar_interval_minutes=bar_minutes, target_vol=strat_cfg.target_vol,
        min_mult=strat_cfg.min_mult, max_mult=strat_cfg.max_mult,
        cooldown_minutes=strat_cfg.cooldown_minutes, step_allocation=strat_cfg.step_allocation,
        max_position=strat_cfg.max_position, long_only=strat_cfg.long_only,
        rebalance_threshold_w=strat_cfg.rebalance_threshold_w,
        min_trade_btc=exec_cfg.min_trade_btc or 0.0,
        gate_window_days=strat_cfg.gate_window_days, gate_roc_threshold=strat_cfg.gate_roc_threshold,
        profit_lock_dd=strat_cfg.profit_lock_dd, vol_scaled_step=strat_cfg.vol_scaled_step,
        funding_limit_long=strat_cfg.funding_limit_long, funding_limit_short=strat_cfg.funding_limit_short,
        fast_period=strat_cfg.fast_period, slow_period=strat_cfg.slow_period,
        ma_type=strat_cfg.ma_type, adx_threshold=strat_cfg.adx_threshold,
        strategy_type=strat_cfg.strategy_type,
    )
    
    if strat_cfg.strategy_type == "trend":
        tp = TrendParams(
            fast_period=strat_cfg.fast_period, slow_period=strat_cfg.slow_period,
            ma_type=strat_cfg.ma_type, cooldown_minutes=strat_cfg.cooldown_minutes,
            step_allocation=strat_cfg.step_allocation, max_position=strat_cfg.max_position,
            long_only=strat_cfg.long_only, funding_limit_long=strat_cfg.funding_limit_long,
            funding_limit_short=strat_cfg.funding_limit_short,
            rebalance_threshold_w=strat_cfg.rebalance_threshold_w
        )
        return TrendStrategy(tp)

    if strat_cfg.strategy_type == "meta":
        # Extract override configs
        mr_opts = strat_cfg.mean_reversion_overrides
        tr_opts = strat_cfg.trend_overrides
        
        # Merge overrides into base params
        # Pydantic models to dict
        base_dict = strat_cfg.model_dump()
        
        mr_dict = {**base_dict, **mr_opts}
        tr_dict = {**base_dict, **tr_opts}
        
        # Filter keys that are valid for StratParams/TrendParams
        valid_mr = {k: v for k, v in mr_dict.items() if k in StratParams.__annotations__}
        valid_tr = {k: v for k, v in tr_dict.items() if k in TrendParams.__annotations__}
        
        mr_p = StratParams(**valid_mr)
        tr_p = TrendParams(**valid_tr)
        
        return MetaStrategy(mr_p, tr_p, adx_threshold=strat_cfg.adx_threshold)

    return EthBtcStrategy(StratParams(**common_kwargs))

def cmd_backtest(args):
    df = load_vision_csv(args.data)
    
    # Date Slicing
    if args.start or args.end:
        s = args.start if args.start else df.index[0]
        e = args.end if args.end else df.index[-1]
        df = df.loc[s:e]
    df = df.sort_index()

    app_cfg = load_config(args.config)
    strategy = build_strategy_from_config(app_cfg, df)
    
    fees_cfg = app_cfg.fees
    fee = FeeParams(
        maker_fee=fees_cfg.maker_fee, taker_fee=fees_cfg.taker_fee,
        slippage_bps=fees_cfg.slippage_bps, bnb_discount=fees_cfg.bnb_discount,
        pay_fees_in_bnb=fees_cfg.pay_fees_in_bnb,
    )

    risk_cfg = app_cfg.risk
    basis = args.basis_btc if args.basis_btc is not None else (risk_cfg.basis_btc if risk_cfg.basis_btc > 0 else 1.0)
    
    reset_days = getattr(risk_cfg, 'drawdown_reset_days', 0.0)
    reset_score = getattr(risk_cfg, 'drawdown_reset_score', 30.0)

    funding_series = load_funding_series(args.funding_data, df.index)
    bnb_series = None
    if args.bnb_data:
        bnb_df = load_vision_csv(args.bnb_data)
        bnb_series = bnb_df["close"].reindex(df.index, method="ffill")

    story_writer = None
    
    # Infer symbols from args or default
    base_asset = args.base if args.base else "ETH"
    quote_asset = args.quote if args.quote else "BTC"
    symbol_str = f"{base_asset}{quote_asset}"

    if args.story:
        try:
            story_writer = StoryWriter(args.story, symbol=symbol_str, alerter=None)
            log.info(f"Story logging enabled: {args.story}")
        except Exception as e:
            log.warning(f"Failed to initialize StoryWriter: {e}")

    bt = Backtester(fee)
    res = bt.simulate(
        df["close"], strategy, funding_series=funding_series, full_df=df,
        initial_btc=basis, bnb_price_series=bnb_series,
        story_writer=story_writer,
        max_daily_loss_btc=risk_cfg.max_daily_loss_btc,
        max_dd_btc=risk_cfg.max_dd_btc,
        max_daily_loss_frac=risk_cfg.max_daily_loss_frac,
        max_dd_frac=risk_cfg.max_dd_frac,
        risk_mode=risk_cfg.risk_mode,
        drawdown_reset_days=reset_days,
        drawdown_reset_score=reset_score,
        base_asset=base_asset,
        quote_asset=quote_asset
    )
    
    print(json.dumps(res["summary"], indent=2))
    if args.out:
        df_out = res["portfolio"]
        if "diagnostics" in res: df_out = df_out.join(res["diagnostics"], how="left")
        df_out.to_csv(args.out)
        print(f"Saved detailed diagnostics to {args.out}")

# --- FIX ITEM 11: Remove Dummy Logic ---
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    
    bt = sub.add_parser("backtest")
    bt.add_argument("--data", required=True)
    bt.add_argument("--funding-data")
    bt.add_argument("--config", required=True)
    bt.add_argument("--out")
    bt.add_argument("--start")
    bt.add_argument("--end")
    bt.add_argument("--bnb-data")
    bt.add_argument("--basis-btc", type=float)
    bt.add_argument("--story", help="Path to output story log file")
    bt.add_argument("--base", help="Base asset name (default: ETH)")
    bt.add_argument("--quote", help="Quote asset name (default: BTC)")
    
    bt.set_defaults(func=cmd_backtest)
    
    # Removed "optimize" parser to avoid confusion with tools/optimize_cli.py
    
    args = ap.parse_args()
    if hasattr(args, "func"): args.func(args)