#!/usr/bin/env python3
"""
ethbtc_accum_bot.py — v5.3 (Optimized & Fixed)
"""

from __future__ import annotations
import sys
import os

# --- MAGIC PATH FIX ---
# Allows importing 'core' modules even if running this script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ----------------------

import math
import argparse
import json
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np

log = logging.getLogger("ethbtc_accum_bot")

# --- Strategy Imports (Safe) ---
try:
    from core.alert_manager import AlertManager
    from core.story_writer import StoryWriter
except ImportError:
    pass

from core.config_schema import load_config

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
    
    # RSI Filter (for MR entries)
    rsi_filter_enabled: bool = False
    rsi_period: int = 14
    rsi_oversold: float = 30.0  # Enter long only if RSI < 30
    rsi_overbought: float = 70.0  # Enter short only if RSI > 70

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
            daily = close.resample("1D").last().shift(1)
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

        # 3b. RSI Filter (NEW)
        # Only allow MR entries when RSI confirms oversold/overbought
        rsi_allow_long = pd.Series(True, index=close.index)
        rsi_allow_short = pd.Series(True, index=close.index)
        
        if self.p.rsi_filter_enabled:
            # Calculate RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)
            avg_gain = gain.ewm(span=self.p.rsi_period, adjust=False).mean()
            avg_loss = loss.ewm(span=self.p.rsi_period, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100.0 - (100.0 / (1.0 + rs))
            rsi = rsi.fillna(50.0)  # Default to neutral
            
            rsi_allow_long = rsi < self.p.rsi_oversold
            rsi_allow_short = rsi > self.p.rsi_overbought
            log.debug(f"[STRATEGY] RSI: {rsi.iloc[-1]:.2f}")

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
        rl_arr = rsi_allow_long.values  # RSI filter for longs
        rs_arr = rsi_allow_short.values  # RSI filter for shorts
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
            
            # --- 3-STATE LOGIC (Long / Neutral / Short) ---
            
            # 1. Check Exits (Mean Reversion)
            # If we are in a position, check if we should exit to Neutral (0.0)
            if state == 1.0:        # Currently Long
                if r > -bx:         # Price rose back to "Exit Band" -> Take Profit/Close
                    desired = 0.0
            elif state == -1.0:     # Currently Short
                if r < bx:          # Price fell back to "Exit Band" -> Take Profit/Close
                    desired = 0.0

            # 2. Check Entries (Strong Signals + RSI Filter)
            if r < -be and gb_arr[i] and ab_arr[i] and rl_arr[i]:
                desired = 1.0       # Strong Dip + RSI oversold -> Buy Long
            elif r > be and gs_arr[i] and as_arr[i] and rs_arr[i]:
                desired = -1.0      # Strong Rally + RSI overbought -> Sell Short

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
                 quote_asset="BTC",
                 leverage: int = 1):
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
            leverage: Leverage multiplier for futures (clamps target_w to [-leverage, +leverage])
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
        
        # === VECTORIZED DYNAMIC SIZING (Pre-Loop Optimization) ===
        # Instead of instantiating PositionSizer 100k times inside the loop,
        # we calculate the step/thresh arrays for the entire series at once.
        # EXCEPTION: Kelly mode uses dynamic step calculation inside the loop.
        
        # 1. Prepare Inputs
        vol_arr = plan["vol"].fillna(0.5).values if "vol" in plan.columns else np.full(len(px), 0.5)
        
        step_arr = np.ones(len(px))
        thresh_arr = np.zeros(len(px))
        
        # Check if any strategy uses Kelly mode (needs dynamic step)
        uses_kelly = False
        kelly_sizer_mr = None
        kelly_sizer_tr = None
        
        if is_meta:
            mr_mode = getattr(strategy.mr.p, 'position_sizing_mode', 'static')
            tr_mode = getattr(strategy.trend.p, 'position_sizing_mode', 'static')
            uses_kelly = (mr_mode == 'kelly' or tr_mode == 'kelly')
            
            if mr_mode == 'kelly':
                from core.position_sizer import PositionSizer, PositionSizerConfig
                kelly_sizer_mr = PositionSizer(PositionSizerConfig(
                    mode='kelly',
                    base_step=getattr(strategy.mr.p, 'step_allocation', 0.5),
                    target_vol=getattr(strategy.mr.p, 'position_sizing_target_vol', 0.5),
                    min_step=getattr(strategy.mr.p, 'position_sizing_min_step', 0.1),
                    max_step=getattr(strategy.mr.p, 'position_sizing_max_step', 1.0),
                    kelly_win_rate=getattr(strategy.mr.p, 'kelly_win_rate', 0.55),
                    kelly_avg_win=getattr(strategy.mr.p, 'kelly_avg_win', 0.02),
                    kelly_avg_loss=getattr(strategy.mr.p, 'kelly_avg_loss', 0.015),
                    kelly_fraction=getattr(strategy.mr.p, 'kelly_fraction', 0.5),
                ))
            if tr_mode == 'kelly':
                from core.position_sizer import PositionSizer, PositionSizerConfig
                kelly_sizer_tr = PositionSizer(PositionSizerConfig(
                    mode='kelly',
                    base_step=getattr(strategy.trend.p, 'step_allocation', 0.5),
                    target_vol=getattr(strategy.trend.p, 'position_sizing_target_vol', 0.5),
                    min_step=getattr(strategy.trend.p, 'position_sizing_min_step', 0.1),
                    max_step=getattr(strategy.trend.p, 'position_sizing_max_step', 1.0),
                    kelly_win_rate=getattr(strategy.trend.p, 'kelly_win_rate', 0.55),
                    kelly_avg_win=getattr(strategy.trend.p, 'kelly_avg_win', 0.02),
                    kelly_avg_loss=getattr(strategy.trend.p, 'kelly_avg_loss', 0.015),
                    kelly_fraction=getattr(strategy.trend.p, 'kelly_fraction', 0.5),
                ))
        elif hasattr(strategy, 'p'):
            single_mode = getattr(strategy.p, 'position_sizing_mode', 'static')
            uses_kelly = (single_mode == 'kelly')
            if uses_kelly:
                from core.position_sizer import PositionSizer, PositionSizerConfig
                kelly_sizer_mr = PositionSizer(PositionSizerConfig(
                    mode='kelly',
                    base_step=getattr(strategy.p, 'step_allocation', 0.5),
                    target_vol=getattr(strategy.p, 'position_sizing_target_vol', 0.5),
                    min_step=getattr(strategy.p, 'position_sizing_min_step', 0.1),
                    max_step=getattr(strategy.p, 'position_sizing_max_step', 1.0),
                    kelly_win_rate=getattr(strategy.p, 'kelly_win_rate', 0.55),
                    kelly_avg_win=getattr(strategy.p, 'kelly_avg_win', 0.02),
                    kelly_avg_loss=getattr(strategy.p, 'kelly_avg_loss', 0.015),
                    kelly_fraction=getattr(strategy.p, 'kelly_fraction', 0.5),
                ))
        
        # Trade tracking for Kelly dynamic updates
        last_trade_entry_price = 0.0
        last_trade_entry_wealth = initial_btc
        
        # 2. Vectorized Sizing Function (for non-Kelly modes)
        def calc_step_vectorized(p_obj, vol_array):
             # Extract params
             mode = getattr(p_obj, 'position_sizing_mode', 'static')
             base = getattr(p_obj, 'step_allocation', 1.0)
             t_vol = getattr(p_obj, 'position_sizing_target_vol', 0.5)
             min_s = getattr(p_obj, 'position_sizing_min_step', 0.1)
             max_s = getattr(p_obj, 'position_sizing_max_step', 1.0)
             
             if mode == 'volatility':
                 # Vectorized: base * (target_vol / vol)
                 # Avoid div by zero
                 safe_vol = np.where(vol_array < 1e-6, 0.5, vol_array) 
                 raw_step = base * (t_vol / safe_vol)
                 return np.clip(raw_step, min_s, max_s)
             elif mode == 'kelly':
                 # Will be calculated dynamically inside loop
                 return np.full(len(vol_array), base) 
             else:
                 return np.full(len(vol_array), base)

        # 3. Calculate Arrays
        if is_meta and "regime_state" in plan.columns:
            # Meta Strategy: Mixed Regime
            regime_arr = plan["regime_state"].fillna(-1.0).values
            regime_score_arr = plan["regime_score"].fillna(0.0).values
            
            # MR Config
            mr_obj = strategy.mr.p
            step_mr_arr = calc_step_vectorized(mr_obj, vol_arr)
            thresh_mr_val = getattr(mr_obj, 'rebalance_threshold_w', 0.0)
            
            # Trend Config
            tr_obj = strategy.trend.p
            step_tr_arr = calc_step_vectorized(tr_obj, vol_arr)
            thresh_tr_val = 0.0 # Trend typically has 0 threshold
            
            # Mask: Where regime > adx_cutoff is Trend (Override based on score for switching logic)
            # Actually, the loop used `score > adx_cutoff` to switch parameters.
            # Let's align with the loop logic:
            # if score > adx_cutoff: use Trend params
            # else: use MR params (derived from regime_state inside loop, but simplified here)
            
            # The loop had complex logic:
            # active_step based on regime_state (MR or Trend mode)
            # THEN overridden by `score > adx_cutoff` check.
            
            # Let's Vectorize the final decision:
            # Condition A: regime_state < 0 (MR Mode) -> Use MR Sizing
            # Condition B: regime_state > 0 (Trend Mode) -> Use Trend Sizing
            
            # Base Sizing
            step_arr = np.where(regime_arr < 0, step_mr_arr, step_tr_arr)
            thresh_arr = np.where(regime_arr < 0, thresh_mr_val, thresh_tr_val)
            
            # Override Condition (Regime Switch Logic from Loop)
            # "if score > adx_cutoff: step = step_trend, thresh = thresh_trend"
            # This override happened at the END of the param selection.
            # It implies that even if we are in MR state, if score spikes, we use Trend sizing?
            # Or is it just setting the *next* state? 
            # Reviewing original code: it changes `step` and `thresh` for the *current* rebalance.
            mask_override = regime_score_arr > adx_cutoff
            step_arr[mask_override] = step_tr_arr[mask_override]
            thresh_arr[mask_override] = thresh_tr_val
            
        else:
            # Single Strategy
            p_obj = getattr(strategy, 'p', None)
            if p_obj:
                step_arr = calc_step_vectorized(p_obj, vol_arr)
                thresh_arr = np.full(len(px), getattr(p_obj, 'rebalance_threshold_w', 0.0))

        # 4. Prepare Numpy Execution Arrays
        tw_arr = target_w.fillna(0.0).values
        eth_arr = np.zeros(len(px))
        btc_arr = np.zeros(len(px)) # Shadow for fast indexing
        btc_arr[0] = initial_btc
        
        # Pre-calc constants to avoid property access in loop
        min_trade_btc = 0.0001
        if hasattr(strategy, 'p'): min_trade_btc = getattr(strategy.p, 'min_trade_btc', 0.0001) or 0.0001
        elif hasattr(strategy, 'mr'): min_trade_btc = getattr(strategy.mr.p, 'min_trade_btc', 0.0001) or 0.0001

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
                    story_writer.check_and_log_daily(timestamp, daily_pnl, wealth, quote_asset, price)
                    story_writer.check_and_log_weekly(timestamp, wealth, quote_asset, price)
                    story_writer.check_and_log_monthly(timestamp, wealth, quote_asset, price)
                    story_writer.check_and_log_annual(timestamp, wealth, quote_asset, price)
                
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
            
            # Leverage clamping has been moved down
            # tw is now fetched from pre-calculated array
            


            # === Loop Slimming (Phase 9) ===
            # Use pre-calculated arrays, OR calculate dynamically for Kelly mode
            step = step_arr[i]
            thresh = thresh_arr[i]
            tw = tw_arr[i]
            
            # Dynamic Kelly: Recalculate step based on current trade stats
            if uses_kelly:
                rv = vol_arr[i] if i < len(vol_arr) else 0.5
                # Determine which sizer to use based on regime
                if is_meta and "regime_state" in plan.columns:
                    regime_state = plan["regime_state"].iat[i] if i < len(plan) else -1
                    if regime_state > 0 and kelly_sizer_tr is not None:
                        step = kelly_sizer_tr.calculate_step(rv)
                    elif kelly_sizer_mr is not None:
                        step = kelly_sizer_mr.calculate_step(rv)
                elif kelly_sizer_mr is not None:
                    step = kelly_sizer_mr.calculate_step(rv)
            
            if leverage > 1:
                tw = max(-leverage, min(leverage, tw))
            else:
                tw = max(-1.0, min(1.0, tw))
                
            # Min Trade Check (Moved to optimized path above)
            # min_trade_btc is already set

            # Rebalance Logic
            new_w = cur_w + step * (tw - cur_w)
            
            # === SNAP-TO-ZERO (Anti-Zeno) ===
            force_exit = False
            if tw == 0.0 and abs(eth[i]) > 0:
                # 1. Dust Cleanup
                eth_val_quote = abs(eth[i]) * price
                if eth_val_quote > min_trade_btc and eth_val_quote < (3.0 * min_trade_btc):
                    new_w = 0.0
                    force_exit = True
                
                # 2. Anti-Zeno: Force Exit if partial step is too small
                implied_target = new_w * wealth / price
                implied_delta = implied_target - eth[i]
                implied_trade_val = abs(implied_delta * price)
                
                if implied_trade_val < min_trade_btc:
                    new_w = 0.0
                    force_exit = True
            
            # Threshold check
            if not force_exit:
                if abs(new_w - cur_w) < thresh:
                    new_w = cur_w

            # Apply leverage scaling: with leverage=2, target_w=0.5 => 100% notional
            # This matches real futures behavior where margin × leverage = position notional
            leveraged_w = new_w * leverage
            target_eth = leveraged_w * wealth / price
            delta = target_eth - eth[i]
            
            # === MIN TRADE SIZE CHECK (matching live_executor.py) ===
            # min_trade_btc defined above
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
                
                # ------------------------------------------------------------
                # SLIPPAGE MODELING (Hardening Fix)
                # Applies slippage penalty to realistic fill price:
                # - Buys: Pay slightly MORE than the bar's close.
                # - Sells: Receive slightly LESS than the bar's close.
                # This makes backtest results more pessimistic and realistic.
                # ------------------------------------------------------------
                slippage_penalty_bps = getattr(self.fee, 'slippage_bps', 0.0)
                slippage_multiplier = slippage_penalty_bps / 10000.0  # Convert bps to decimal
                
                if delta > 0:  # Buy
                    fill_price = price * (1.0 + slippage_multiplier)  # Penalize: pay more
                else:  # Sell
                    fill_price = price * (1.0 - slippage_multiplier)  # Penalize: receive less
                
                # Apply trade using the slippage-adjusted fill price
                eth[i] += delta
                btc[i] -= delta * fill_price
                
                if delta > 0: # Buy
                    trades.append({"time": px.index[i], "side":"BUY", "price":fill_price, "qty":delta, "fee":fee_val})
                    # Story: Log trade
                    if story_writer:
                        story_writer.log_trade(timestamp, "BUY", delta, fill_price, base_asset, quote_asset)
                    
                    # Dynamic Kelly: Track entry for later P&L calculation
                    if uses_kelly:
                        last_trade_entry_price = fill_price
                        last_trade_entry_wealth = wealth
                        
                else: # Sell
                    trades.append({"time": px.index[i], "side":"SELL", "price":fill_price, "qty":delta, "fee":fee_val})
                    # Story: Log trade
                    if story_writer:
                        story_writer.log_trade(timestamp, "SELL", delta, fill_price, base_asset, quote_asset)
                    
                    # Dynamic Kelly: Calculate P&L and record trade
                    if uses_kelly and last_trade_entry_price > 0:
                        # Calculate P&L as fraction of wealth at entry
                        trade_pnl = (fill_price - last_trade_entry_price) / last_trade_entry_price
                        trade_pnl_adjusted = trade_pnl * abs(delta) * last_trade_entry_price / max(last_trade_entry_wealth, 1e-12)
                        
                        # Record to appropriate sizer based on regime
                        if is_meta and "regime_state" in plan.columns:
                            regime_state = plan["regime_state"].iat[i] if i < len(plan) else -1
                            if regime_state > 0 and kelly_sizer_tr is not None:
                                kelly_sizer_tr.record_trade(trade_pnl_adjusted)
                            elif kelly_sizer_mr is not None:
                                kelly_sizer_mr.record_trade(trade_pnl_adjusted)
                        elif kelly_sizer_mr is not None:
                            kelly_sizer_mr.record_trade(trade_pnl_adjusted)
                        
                        last_trade_entry_price = 0.0  # Reset for next trade
                
                total_fees_btc += fee_val
                total_turnover += notional


            # cur_w tracks signal weight (unleveraged), not actual position
            # With leverage=2, if position is 100% notional, cur_w should be 0.5
            cur_w = (eth[i] * price) / max(wealth, 1e-12) / leverage

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
    """
    Build strategy from config - wrapper for backwards compatibility.
    
    IMPORTANT: This function now uses the shared strategy_factory module
    to ensure parity between backtest and live executor.
    """
    # Import from shared factory to ensure parity
    from core.strategy_factory import build_strategy
    
    strategy, _ = build_strategy(app_cfg, df)
    return strategy


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
    base_asset = args.base
    quote_asset = args.quote
    
    # Smart Inference from Data Filename if not provided
    if (not base_asset or not quote_asset) and args.data:
        import os
        filename = os.path.basename(args.data)
        # Attempt to match common patterns like "BTCUSDT_..." or "ETHBTC_..."
        upper_name = filename.upper()
        
        # Common Quote Assets to check for at the end of the symbol part
        known_quotes = ["USDT", "USDC", "BTC", "ETH", "BNB"]
        
        detected_base = None
        detected_quote = None
        
        for q in known_quotes:
            # Check if filename starts with SOMETHING + QUOTE + optional separator
            # e.g. BTCUSDT_15m... -> starts with BTCUSDT
            # We look for the quote asset in the first chunk
            parts = upper_name.split('_')
            first_part = parts[0] # e.g. "BTCUSDT" or "ETHBTC"
            
            if first_part.endswith(q) and len(first_part) > len(q):
                detected_quote = q
                detected_base = first_part[:-len(q)]
                break
        
        if detected_base and detected_quote:
            if not base_asset: base_asset = detected_base
            if not quote_asset: quote_asset = detected_quote
            print(f"ℹ️  Inferred Asset: {base_asset}/{quote_asset} from filename '{filename}'")

    # Final Fallbacks
    base_asset = base_asset if base_asset else "ETH"
    quote_asset = quote_asset if quote_asset else "BTC"
    symbol_str = f"{base_asset}{quote_asset}"

    if args.story:
        try:
            story_writer = StoryWriter(args.story, symbol=symbol_str, base_asset=base_asset, alerter=None)
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
        quote_asset=quote_asset,
        leverage=app_cfg.execution.leverage,
    )
    
    print(json.dumps(res["summary"], indent=2))
    
    # Enhanced Report Generation
    if args.report:
        from core.backtest_report import BacktestReport
        from datetime import datetime
        
        strategy_name = type(strategy).__name__ if hasattr(strategy, '__class__') else "Strategy"
        
        report = BacktestReport.from_backtest_result(
            result=res,
            price_series=df["close"],
            strategy_name=strategy_name,
            symbol=symbol_str,
            base_asset=base_asset,
            quote_asset=quote_asset,
        )
        
        # Print to terminal
        report.print_report()
        
        # Save to Markdown file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"results/backtest_report_{symbol_str}_{timestamp}.md"
        report.to_markdown(report_path)
        print(f"\n📄 Report saved to: {report_path}")
    
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
    bt.add_argument("--report", action="store_true", help="Generate enhanced report with HODL comparison and risk metrics")
    
    bt.set_defaults(func=cmd_backtest)
    
    # Removed "optimize" parser to avoid confusion with tools/optimize_cli.py
    
    args = ap.parse_args()
    if hasattr(args, "func"): args.func(args)