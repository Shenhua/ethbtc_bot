#!/usr/bin/env python3
"""
ethbtc_accum_bot.py — v5.2 (Definitive Multi-Strategy Backtester)
"""

from __future__ import annotations
import sys, os

# --- MAGIC PATH FIX ---
# Allows importing 'core' modules even if running this script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ----------------------

import math, argparse, json, random
from dataclasses import dataclass
from typing import Dict, Optional, Any, Union
import pandas as pd
import numpy as np

# --- Strategy Imports (Safe) ---
try:
    from core.trend_strategy import TrendStrategy, TrendParams
    from core.meta_strategy import MetaStrategy
    from core.alert_manager import AlertManager
except ImportError:
    pass

# ------------------ Loaders ------------------
def load_json_config(path: Optional[str]) -> Dict:
    if not path: return {}
    with open(path, "r") as f: data = json.load(f)
    if not isinstance(data, dict): return {}
    flat: Dict[str, Any] = dict(data)
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
    
    return df.dropna(subset=["close"]).set_index("close_time").sort_index()

def _write_excel(path: str, sheets: dict):
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
        # 1. Indicators
        if self.p.trend_kind == "sma":
            ma = close.rolling(self.p.trend_lookback).mean()
            ratio = close / ma - 1.0
        else:
            ratio = (close / close.shift(self.p.trend_lookback)) - 1.0
            
        ret = close.pct_change(fill_method=None).fillna(0)
        bars_per_year = (365 * 24 * 60) / float(self.p.bar_interval_minutes)
        vol = ret.rolling(self.p.vol_window).std() * math.sqrt(bars_per_year)
        
        adj = self.p.vol_adapt_k * (vol.fillna(vol.median()))
        band_entry = self.p.flip_band_entry + adj
        band_exit  = self.p.flip_band_exit + adj

        # 2. Gate Logic (Pre-calculated Vectorized)
        if self.p.gate_window_days > 0:
            daily = close.resample("1D").last()
            roc_daily = daily.pct_change(self.p.gate_window_days, fill_method=None)            
            # Reindex to match close (15m), ffill to propagate daily value forward
            roc = roc_daily.reindex(close.index).ffill().fillna(0.0)
            
            gate_buy_mask = roc <= -self.p.gate_roc_threshold
            gate_sell_mask = roc >= self.p.gate_roc_threshold
        else:
            # Gate Disabled: Always True
            gate_buy_mask = pd.Series(True, index=close.index)
            gate_sell_mask = pd.Series(True, index=close.index)

        # 3. Funding Logic (Vectorized)
        allow_buy = pd.Series(True, index=close.index)
        allow_sell = pd.Series(True, index=close.index)
        
        if funding is not None:
            # Align funding to close index
            f_aligned = funding.reindex(close.index).ffill().fillna(0.0)
            allow_buy = f_aligned <= self.p.funding_limit_long
            allow_sell = f_aligned >= self.p.funding_limit_short

        # 4. Loop State Machine
        sig = pd.Series(0.0, index=close.index)
        state = -1.0
        last_flip_ts = close.index[0]
        min_delta = pd.Timedelta(minutes=self.p.cooldown_minutes)

        # Pre-convert to numpy for speed & safety
        idx_arr = close.index
        r_arr = ratio.values
        be_arr = band_entry.values
        bx_arr = band_exit.values
        gb_arr = gate_buy_mask.values
        gs_arr = gate_sell_mask.values
        ab_arr = allow_buy.values
        as_arr = allow_sell.values
        
        out_sig = np.zeros(len(close))

        for i in range(len(close)):
            t = idx_arr[i]
            
            # Cooldown check
            if (t - last_flip_ts) < min_delta:
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
        if self.p.target_vol > 0:
            vol_adj = vol.replace(0, np.nan)
            mult = (self.p.target_vol / vol_adj).clip(self.p.min_mult, self.p.max_mult).fillna(self.p.min_mult)
        else:
            mult = 1.0

        # 6. Final Allocation
        # Convert numpy array back to Series for alignment
        sig_series = pd.Series(out_sig, index=close.index)
        
        lo = 0.0 if self.p.long_only else -self.p.max_position
        target_w = (sig_series * mult).clip(lo, self.p.max_position)
        
        return pd.DataFrame({"target_w": target_w})

class Backtester:
    def __init__(self, fee: FeeParams): self.fee = fee

    def simulate(self, close: pd.Series, 
                 strat: Union[EthBtcStrategy, TrendStrategy, MetaStrategy],
                 funding_series: Optional[pd.Series] = None,
                 initial_btc: float = 1.0, start_bnb: float = 0.05,
                 bnb_price_series: Optional[pd.Series] = None,
                 max_daily_loss_btc=0.0, max_dd_btc=0.0,
                 max_daily_loss_frac=0.0, max_dd_frac=0.0, risk_mode="fixed_basis",
                 full_df: Optional[pd.DataFrame] = None):
        
        px = close.astype(float).copy()
        
        aligned_funding = None
        if funding_series is not None:
            aligned_funding = funding_series.reindex(close.index).ffill().fillna(0.0)

        # Generate Positions based on Strategy Type
        if hasattr(strat, 'adx_threshold'): # MetaStrategy
            if full_df is None: raise ValueError("MetaStrategy requires full OHLC dataframe (full_df).")
            plan = strat.generate_positions(full_df, funding=aligned_funding)
        elif hasattr(strat, 'generate_positions'):
            # Trend or Mean Reversion
            # Trend handles Series or DF.
            if isinstance(strat, EthBtcStrategy):
                plan = strat.generate_positions(px, funding=aligned_funding)
            else:
                # TrendStrategy or others that might use OHLC if available
                input_data = full_df if full_df is not None else px
                plan = strat.generate_positions(input_data, funding=aligned_funding)
        
        target_w = plan["target_w"]

        # --- Execution Loop ---
        btc = np.zeros(len(px)); eth = np.zeros(len(px)); bnb = np.zeros(len(px))
        btc[0] = initial_btc; bnb[0] = start_bnb
        cur_w = 0.0

        equity_high = initial_btc
        maxdd_hit = False
        
        taker_fee = self.fee.taker_fee
        fee_disc = (1.0 - self.fee.bnb_discount) if self.fee.pay_fees_in_bnb else 1.0
        
        total_fees_btc = 0.0
        total_turnover = 0.0
        trades = []

        step = 1.0
        thresh = 0.0
        
        # Try to extract step/thresh
        if hasattr(strat, 'p'):
            step = getattr(strat.p, 'step_allocation', 1.0)
            thresh = getattr(strat.p, 'rebalance_threshold_w', 0.0)
        elif hasattr(strat, 'mr'):
            step = getattr(strat.mr.p, 'step_allocation', 1.0)
            thresh = getattr(strat.mr.p, 'rebalance_threshold_w', 0.0)

        for i in range(1, len(px)):
            price = float(px.iat[i])
            btc[i] = btc[i-1]; eth[i] = eth[i-1]; bnb[i] = bnb[i-1]
            
            wealth = btc[i] + eth[i] * price
            if wealth > equity_high: equity_high = wealth
            
            if max_dd_frac > 0.0 and equity_high > 0:
                dd = (equity_high - wealth) / equity_high
                if dd >= max_dd_frac: maxdd_hit = True
            
            tw = 0.0 if maxdd_hit else float(target_w.iat[i])
            
            new_w = cur_w + step * (tw - cur_w)
            if abs(new_w - cur_w) < thresh:
                new_w = cur_w

            target_eth = new_w * wealth / price
            delta = target_eth - eth[i]
            
            if abs(delta * price) > 0.0001:
                notional = abs(delta * price)
                f_rate = taker_fee * fee_disc 
                fee_val = notional * f_rate
                
                if self.fee.pay_fees_in_bnb and bnb_price_series is not None:
                    bnb_px = bnb_price_series.iat[i]
                    if bnb_px > 0:
                        bnb_cost = fee_val / bnb_px
                        if bnb[i] >= bnb_cost:
                            bnb[i] -= bnb_cost
                        else:
                            btc[i] -= (fee_val - (bnb[i]*bnb_px))
                            bnb[i] = 0
                else:
                    btc[i] -= fee_val
                
                if delta > 0: # Buy
                    btc[i] -= notional
                    eth[i] += delta
                    trades.append({"time": px.index[i], "side":"BUY", "price":price, "qty":delta, "fee":fee_val})
                else: # Sell
                    btc[i] += notional
                    eth[i] += delta
                    trades.append({"time": px.index[i], "side":"SELL", "price":price, "qty":delta, "fee":fee_val})
                
                total_fees_btc += fee_val
                total_turnover += notional

            cur_w = (eth[i] * price) / max(btc[i] + eth[i] * price, 1e-12)

        final_btc = btc[-1] + eth[-1] * float(px.iat[-1])
        
        summary = {
            "final_btc": final_btc,
            "total_return": (final_btc / initial_btc) - 1.0,
            "max_drawdown_pct": (equity_high - final_btc)/equity_high if equity_high > 0 else 0.0,
            "fees_btc": total_fees_btc,
            "turnover_btc": total_turnover,
            "n_trades": len(trades)
        }
        
        # CRITICAL FIX: Return 'plan' (diagnostics) alongside portfolio for analysis
        port_df = pd.DataFrame({"wealth_btc": btc + eth*px}, index=px.index)
        return {
            "summary": summary, 
            "portfolio": port_df, 
            "trades": pd.DataFrame(trades),
            "diagnostics": plan
        }

# ------------------ CLI ------------------
def cmd_backtest(args):
    df = load_vision_csv(args.data)
    
    funding_series = None
    if args.funding_data:
        f_df = pd.read_csv(args.funding_data)
        f_df["time"] = pd.to_datetime(f_df["time"], format="mixed", utc=True)
        f_df = f_df.set_index("time").sort_index()
        funding_series = f_df["rate"]

    cfg = load_json_config(args.config)
    exec_cfg = cfg.get("execution", {})
    interval_str = exec_cfg.get("interval", cfg.get("interval", "15m"))
    bar_minutes = _interval_to_minutes(str(interval_str))

    risk_cfg = cfg.get("risk") or cfg

    basis = float(risk_cfg.get("basis_btc", 1.0))
    max_daily_loss_btc = float(risk_cfg.get("max_daily_loss_btc", 0.0))
    max_dd_btc = float(risk_cfg.get("max_dd_btc", 0.0))
    max_daily_loss_frac = float(risk_cfg.get("max_daily_loss_frac", 0.0))
    max_dd_frac = float(risk_cfg.get("max_dd_frac", 0.0))
    risk_mode = risk_cfg.get("risk_mode", "fixed_basis")

    # --- Fees: nested "fees" block if present, otherwise flat keys ---
    fee_cfg = cfg.get("fees") or cfg
    fee = FeeParams(
        maker_fee=float(fee_cfg.get("maker_fee", 0.0002)),
        taker_fee=float(fee_cfg.get("taker_fee", 0.0004)),
        slippage_bps=float(fee_cfg.get("slippage_bps", 1.0)),
        bnb_discount=float(fee_cfg.get("bnb_discount", 0.25)),
        pay_fees_in_bnb=bool(fee_cfg.get("pay_fees_in_bnb", True)),
    )

    strat_params = cfg.get("strategy", {})
    clean_params = {k: v for k, v in strat_params.items() if not k.startswith("_")}
    clean_params.setdefault("bar_interval_minutes", bar_minutes)
    s_type = clean_params.get("strategy_type", "mean_reversion")
    
    if s_type == "meta":
        try:
            from core.meta_strategy import MetaStrategy
            from core.trend_strategy import TrendParams
            
            # 1. Create a clean base by removing the container keys
            # We use .copy() to avoid affecting other logic
            base = clean_params.copy()
            mr_opts = base.pop("mean_reversion_overrides", {})
            tr_opts = base.pop("trend_overrides", {})
            
            # 2. Clean out metadata like "comment" which might cause errors too
            base.pop("comment", None)
            mr_opts.pop("comment", None)
            tr_opts.pop("comment", None)
            
            # 3. Merge: Base + Overrides
            mr_merged = {**base, **mr_opts}
            tr_merged = {**base, **tr_opts}

            # 4. Filter against StratParams fields to be 100% safe
            # (This prevents crashes if you add comments or new keys later)
            valid_mr_keys = StratParams.__annotations__.keys()
            mr_final = {k: v for k, v in mr_merged.items() if k in valid_mr_keys}

            valid_tr_keys = TrendParams.__annotations__.keys()
            tr_final = {k: v for k, v in tr_merged.items() if k in valid_tr_keys}

            # 5. Initialize
            mr_p = StratParams(**mr_final)
            tr_p = TrendParams(**tr_final)
            
            strat = MetaStrategy(mr_p, tr_p, adx_threshold=float(clean_params.get("adx_threshold", 25.0)))
        except ImportError:
            print("Error: core.meta_strategy not found.")
            return

    elif s_type == "trend":
        try:
            from core.trend_strategy import TrendStrategy, TrendParams
            tr_p = TrendParams(
                fast_period=int(clean_params.get("fast_period", 50)),
                slow_period=int(clean_params.get("slow_period", 200)),
                ma_type=clean_params.get("ma_type", "ema"),
                cooldown_minutes=int(clean_params.get("cooldown_minutes", 60)),
                step_allocation=float(clean_params.get("step_allocation", 1.0)),
                max_position=float(clean_params.get("max_position", 1.0)),
                long_only=bool(clean_params.get("long_only", True)),
                funding_limit_long=float(clean_params.get("funding_limit_long", 0.05)),
                funding_limit_short=float(clean_params.get("funding_limit_short", -0.05))
            )
            strat = TrendStrategy(tr_p)
        except ImportError:
            print("Error: core.trend_strategy not found.")
            return
        
    else:
        p = StratParams(**clean_params)
        strat = EthBtcStrategy(p)

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
        risk_mode=risk_mode
    )
    print(json.dumps(res["summary"], indent=2))
    if args.out:
        # Merge Wealth with Diagnostics for Analysis
        df_out = res["portfolio"]
        if "diagnostics" in res:
            df_out = df_out.join(res["diagnostics"], how="left")
            
        df_out.to_csv(args.out)
        print(f"Saved detailed diagnostics to {args.out}")

def _interval_to_minutes(interval: str) -> int:
    mapping = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480,
        "12h": 720, "1d": 1440,
    }
    return mapping.get(interval, 15)

def cmd_dummy(args):
    print("Optimize/Selftest moved to tools/")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    
    bt = sub.add_parser("backtest")
    bt.add_argument("--data", required=True)
    bt.add_argument("--funding-data", help="Path to funding rates CSV")
    bt.add_argument("--config", required=True)
    bt.add_argument("--out")
    
    # Overrides
    bt.add_argument("--rebalance-threshold-w", type=float, default=None)
    bt.add_argument("--strategy", choices=["mean_reversion", "trend", "meta"], default=None)
    
    bt.set_defaults(func=cmd_backtest)
    
    opt = sub.add_parser("optimize")
    opt.set_defaults(func=cmd_dummy)
    
    args = ap.parse_args()
    if hasattr(args, "func"): args.func(args)