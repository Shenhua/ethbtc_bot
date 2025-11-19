#!/usr/bin/env python3
"""
ethbtc_accum_bot.py — Rich Mode + Failsafe + CLI override + anti-churn + date slicing
Version: 3.0.0 (2025-11-18)
"""

from __future__ import annotations
import math, argparse, json, random
from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd
import numpy as np

__version__ = "3.0.0-rich-cli-guards-daterange"

# ------------------ Loaders ------------------
# -------------- Loaders ------------------
def load_json_config(path: Optional[str]) -> Dict:
    """
    Load config JSON and flatten nested blocks so the backtester can accept:

    - Legacy style:
        {
          "params": { ... },
          "fees": { ... }
        }

    - New AppConfig style:
        {
          "fees": { ... },
          "strategy": { ... },
          "execution": { ... },
          "risk": { ... }
        }

    The returned dict is a flat dict with keys like trend_lookback, maker_fee,
    basis_btc, long_only, etc.
    """
    if not path:
        return {}

    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return {}

    # Start with top-level keys (basis_btc, etc.)
    flat: Dict[str, Any] = dict(data)

    # Flatten any nested blocks we care about
    for block in ("params", "fees", "strategy", "execution", "risk"):
        block_dict = data.get(block)
        if isinstance(block_dict, dict):
            flat.update(block_dict)

    return flat

def load_vision_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    alias = {"opentime":"open_time","closetime":"close_time"}
    for k,v in alias.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k:v}, inplace=True)
    if "close_time" not in df.columns:
        for c in list(df.columns):
            if c.replace(" ","_") == "close_time":
                df.rename(columns={c:"close_time"}, inplace=True)
                break
        if "close_time" not in df.columns:
            raise ValueError("close_time column not found")

    def _parse_dt(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s):
            vmax = float(pd.to_numeric(s, errors="coerce").dropna().head(1).max() or 0)
            unit = "ms" if vmax > 1e11 else "s"
            return pd.to_datetime(s, unit=unit, utc=True)
        try:
            return pd.to_datetime(s, format="ISO8601", utc=True)
        except Exception:
            return pd.to_datetime(s, utc=True, format="mixed")

    if "open_time" in df.columns:
        df["open_time"] = _parse_dt(df["open_time"])
    df["close_time"] = _parse_dt(df["close_time"])

    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        elif c.capitalize() in df.columns:
            df[c] = pd.to_numeric(df.pop(c.capitalize()), errors="coerce")
        else:
            raise ValueError(f"Missing column {c}")
    df = df.dropna(subset=["close"]).set_index("close_time").sort_index()
    return df

# ------------------ Excel writer (tz-safe + chart) ------------------
def _write_excel(path: str, sheets: dict):
    import pandas as _pd
    from pandas.api.types import is_object_dtype, is_datetime64_any_dtype

    def _excel_safe_df(obj):
        df = obj.copy()
        if isinstance(df.index, _pd.DatetimeIndex):
            if df.index.tz is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)
        for c in df.columns:
            if is_datetime64_any_dtype(df[c]) and getattr(df[c].dtype, "tz", None) is not None:
                df[c] = df[c].dt.tz_convert("UTC").dt.tz_localize(None)
            elif is_object_dtype(df[c]):
                parsed = _pd.to_datetime(df[c], errors="coerce", utc=True)
                if str(parsed.dtype).startswith("datetime64[ns,"):
                    if parsed.notna().mean() > 0.8:
                        df[c] = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
                    else:
                        df[c] = df[c].astype(str)
                else:
                    df[c] = df[c].astype(str)
        return df

    with _pd.ExcelWriter(path, engine="xlsxwriter") as w:
        book = w.book
        for name, obj in sheets.items():
            n = str(name)[:31]
            if isinstance(obj, _pd.DataFrame):
                safe = _excel_safe_df(obj)
                safe.to_excel(w, sheet_name=n, index=True)
            elif isinstance(obj, dict):
                _pd.DataFrame([obj]).to_excel(w, sheet_name=n, index=False)
            else:
                _pd.DataFrame({"value":[str(obj)]}).to_excel(w, sheet_name=n, index=False)

        if "Equity" in sheets and isinstance(sheets["Equity"], _pd.DataFrame):
            eq_df = _excel_safe_df(sheets["Equity"]).reset_index()
            nrows = len(eq_df)
            if nrows > 1 and "wealth_btc" in eq_df.columns:
                chart = book.add_chart({"type":"line"})
                chart.set_title({"name":"Equity Curve"})
                chart.set_x_axis({"name":"Time"}); chart.set_y_axis({"name":"Wealth (BTC)"})
                wealth_col = list(eq_df.columns).index("wealth_btc")
                chart.add_series({
                    "name":"Wealth BTC",
                    "categories":["Equity", 1, 0, nrows, 0],
                    "values":["Equity", 1, wealth_col, nrows, wealth_col],
                })
                w.book.add_worksheet("Chart").insert_chart("B2", chart)

# ------------------ Strategy & Backtest ------------------
@dataclass
class StratParams:
    # signal
    trend_kind: str = "sma"           # 'sma' or 'roc'
    trend_lookback: int = 200
    flip_band_entry: float = 0.025
    flip_band_exit: float = 0.015
    # vol-adaptive banding
    vol_window: int = 60
    vol_adapt_k: float = 0.0
    bar_interval_minutes: int = 15
    target_vol: float = 0.5
    min_mult: float = 0.5
    max_mult: float = 1.5
    # execution control
    cooldown_minutes: int = 180
    step_allocation: float = 0.5
    max_position: float = 1.0
    long_only: bool = True       
    # anti-churn
    rebalance_threshold_w: float = 0.0
    min_trade_btc: float = 0.0
    # regime gate (daily)
    gate_window_days: int = 60
    gate_roc_threshold: float = 0.0

@dataclass
class FeeParams:
    maker_fee: float = 0.0002
    taker_fee: float = 0.0004
    slippage_bps: float = 1.0
    bnb_discount: float = 0.25
    pay_fees_in_bnb: bool = True

class EthBtcStrategy:
    def __init__(self, p: StratParams): self.p = p

    def _trend_ratio(self, close: pd.Series) -> pd.Series:
        if self.p.trend_kind == "sma":
            ma = close.rolling(self.p.trend_lookback).mean()
            return close / ma - 1.0
        elif self.p.trend_kind == "roc":
            shifted = close.shift(self.p.trend_lookback)
            return (close / shifted) - 1.0
        else:
            raise ValueError("trend_kind must be 'sma' or 'roc'")

    def _realized_vol(self, ret: pd.Series) -> pd.Series:
        # Annualized realized volatility, aware of bar interval
        bars_per_year = (365 * 24 * 60) / float(self.p.bar_interval_minutes)
        return ret.rolling(self.p.vol_window).std() * math.sqrt(bars_per_year)

    def _gate_series(self, close: pd.Series) -> pd.Series:
        daily = close.resample("1D").last()
        roc = daily.pct_change(self.p.gate_window_days)
        return roc.reindex(close.index, method="ffill")

    def generate_positions(self, close: pd.Series) -> pd.DataFrame:
        ratio = self._trend_ratio(close).fillna(0)
        ret = close.pct_change().fillna(0)
        vol = self._realized_vol(ret)

        # Adaptive bands
        base_entry = self.p.flip_band_entry
        base_exit  = self.p.flip_band_exit
        if self.p.vol_adapt_k > 0:
            adj = self.p.vol_adapt_k * (vol.fillna(vol.median()))
            band_entry = base_entry + adj
            band_exit  = base_exit  + adj
        else:
            band_entry = pd.Series(base_entry, index=close.index)
            band_exit  = pd.Series(base_exit,  index=close.index)

        # Regime gate (daily ROC)
        gate = self._gate_series(close) if self.p.gate_window_days > 0 else pd.Series(0.0, index=close.index)

        # Hysteresis with cooldown
        sig = pd.Series(0.0, index=close.index)
        state = -1.0  # start BTC
        last_flip_ts = close.index[0]
        min_delta = pd.Timedelta(minutes=self.p.cooldown_minutes)

        for t in close.index:
            if (t - last_flip_ts) < min_delta:
                sig.loc[t] = state
                continue

            r = ratio.loc[t]
            be = float(band_entry.loc[t]); bx = float(band_exit.loc[t])
            g = gate.loc[t] if gate is not None else 0.0
            desired = state

            if state <= 0:
                if r > be and g >= self.p.gate_roc_threshold:
                    desired = +1.0
            if state >= 0:
                if r < -be and g <= -self.p.gate_roc_threshold:
                    desired = -1.0
                if state > 0 and r < bx:
                    desired = -1.0
                if state < 0 and r > -bx:
                    desired = +1.0

            if desired != state:
                state = desired
                last_flip_ts = t
            sig.loc[t] = state

        # Vol targeting multiplier
        if self.p.target_vol > 0:
            vol_adj = vol.replace(0, np.nan)
            mult = (self.p.target_vol / vol_adj).clip(
                self.p.min_mult, self.p.max_mult
            ).fillna(self.p.min_mult)
        else:
            mult = pd.Series(1.0, index=close.index)

        # Target ETH weight; smoothed allocation handled in execution
        # Target ETH weight (spot: optional long-only clip)
        lo = 0.0 if getattr(self.p, "long_only", True) else -self.p.max_position
        target_w = (sig * mult).clip(lo, self.p.max_position)
        return pd.DataFrame({"target_w": target_w})

class Backtester:
    def __init__(self, fee: FeeParams): self.fee = fee

    def simulate(self, close: pd.Series, strat: EthBtcStrategy,
                 initial_btc: float = 1.0, start_bnb: float = 0.05,
                 bnb_price_series: Optional[pd.Series] = None,
                 max_daily_loss_btc: float = 0.0,
                 max_dd_btc: float = 0.0,
                 max_daily_loss_frac: float = 0.0,
                 max_dd_frac: float = 0.0,
                 risk_mode: str = "fixed_basis"):
        px = close.astype(float).copy()
        plan = strat.generate_positions(px)
        target_w = plan["target_w"]

        btc = np.zeros(len(px)); eth = np.zeros(len(px)); bnb = np.zeros(len(px))
        btc[0] = initial_btc; bnb[0] = start_bnb
        cur_w = 0.0  # start BTC

        # ----- Risk state (drawdown + daily loss) -----
        first_ts = px.index[0]
        first_price = float(px.iat[0])
        first_wealth = float(btc[0] + eth[0] * first_price)
        equity_high_btc = float(first_wealth)
        current_date = first_ts.normalize()
        daily_start_wealth = first_wealth
        daily_limit_hit = False
        maxdd_hit = False

        taker_fee = self.fee.taker_fee
        maker_fee = self.fee.maker_fee
        fee_disc = (1.0 - self.fee.bnb_discount) if self.fee.pay_fees_in_bnb else 1.0
        slip = self.fee.slippage_bps / 1e4

        def _charge_fee_bnb(i:int, fee_btc: float):
            nonlocal btc, bnb
            if not (self.fee.pay_fees_in_bnb and bnb_price_series is not None): return
            bpx = max(float(bnb_price_series.iat[i]), 1e-12)
            need_bnb = fee_btc / bpx
            if bnb[i] >= need_bnb: bnb[i] -= need_bnb
            else:
                short = need_bnb - bnb[i]; cost_btc = short * bpx
                if btc[i] > cost_btc: btc[i] -= cost_btc; bnb[i] = 0.0
                else: bnb[i] = 0.0; btc[i] = 0.0

        trades = []; total_fees_btc = 0.0; total_turnover_btc = 0.0

        for i in range(1, len(px)):
            ts = px.index[i]
            price = float(px.iat[i])
            btc[i] = btc[i-1]; eth[i] = eth[i-1]; bnb[i] = bnb[i-1]

            wealth = float(btc[i] + eth[i] * price)

            # ----- Update risk state (drawdown + daily loss) -----

            # Max drawdown tracking
            if max_dd_btc > 0.0 or max_dd_frac > 0.0:
                if wealth > equity_high_btc:
                    equity_high_btc = wealth
                dd_now = equity_high_btc - wealth

                if risk_mode == "dynamic":
                    # Dynamic: fraction of peak equity, or fallback to absolute BTS if provided
                    if max_dd_frac > 0.0:
                        threshold_dd = equity_high_btc * max_dd_frac
                    else:
                        threshold_dd = max_dd_btc
                else:
                    # fixed_basis (existing behaviour): absolute BTC limit already derived from basis_btc
                    threshold_dd = max_dd_btc

                if threshold_dd > 0.0 and dd_now >= threshold_dd:
                    maxdd_hit = True

            # Daily PnL tracking
            if max_daily_loss_btc > 0.0 or max_daily_loss_frac > 0.0:
                cur_date = ts.normalize()
                if cur_date != current_date:
                    current_date = cur_date
                    daily_start_wealth = wealth
                    daily_limit_hit = False
                daily_pnl = wealth - daily_start_wealth

                if risk_mode == "dynamic":
                    if max_daily_loss_frac > 0.0:
                        threshold_loss = daily_start_wealth * max_daily_loss_frac
                    else:
                        threshold_loss = max_daily_loss_btc
                else:
                    threshold_loss = max_daily_loss_btc

                if threshold_loss > 0.0 and daily_pnl <= -threshold_loss:
                    daily_limit_hit = True

            # If we've hit max DD, force BTC-only target (w = 0)
            if maxdd_hit:
                tw = 0.0
            else:
                tw = float(target_w.iat[i])

            # Smooth step toward target
            new_w = cur_w + strat.p.step_allocation * (tw - cur_w)
            new_w = max(-strat.p.max_position, min(strat.p.max_position, new_w))

            # If daily loss limit is hit, freeze trading (no new rebalances)
            if daily_limit_hit:
                # Do not change cur_w or holdings, just carry forward
                continue

            # Anti-churn #1: rebalance deadband
            if abs(new_w - cur_w) < strat.p.rebalance_threshold_w:
                cur_w = new_w  # update desired weight without trading
                continue

            target_eth = max(new_w, 0.0) * wealth / price  # long-only ETH weight
            delta_eth = target_eth - eth[i]

            # Skip dust
            if wealth <= 0 or abs(delta_eth) <= 1e-12:
                cur_w = new_w
                continue

            if delta_eth > 0:
                # Buy ETH (taker)
                eff = price*(1+slip)
                btc_notional = delta_eth * eff
                if btc_notional < strat.p.min_trade_btc:
                    cur_w = new_w; continue
                fee = btc_notional * taker_fee * fee_disc
                if self.fee.pay_fees_in_bnb and bnb_price_series is not None:
                    eth[i] += delta_eth; _charge_fee_bnb(i, fee); btc[i] -= btc_notional
                else:
                    btc[i] -= (btc_notional + fee); eth[i] += delta_eth
                trades.append({"time": ts, "side":"BUY_ETH","px":float(price),"px_eff":float(eff),
                               "eth_delta": float(delta_eth), "fee_btc_equiv": float(fee)})
                total_fees_btc += float(fee); total_turnover_btc += float(btc_notional)
            else:
                # Sell ETH (maker if possible)
                eff = price*(1-slip)
                eth_out = -delta_eth
                btc_gross = eth_out * eff
                if btc_gross < strat.p.min_trade_btc:
                    cur_w = new_w; continue
                fee = btc_gross * maker_fee * fee_disc
                if self.fee.pay_fees_in_bnb and bnb_price_series is not None:
                    btc[i] += btc_gross; _charge_fee_bnb(i, fee); eth[i] -= eth_out
                else:
                    btc[i] += (btc_gross - fee); eth[i] -= eth_out
                trades.append({"time": ts, "side":"SELL_ETH","px":float(price),"px_eff":float(eff),
                               "eth_delta": -float(eth_out), "fee_btc_equiv": float(fee)})
                total_fees_btc += float(fee); total_turnover_btc += float(btc_gross)

            cur_w = (eth[i]*price) / max(btc[i] + eth[i]*price, 1e-12)

        wealth_btc = btc + eth * px.values
        df = pd.DataFrame({
            "close": px,
            "btc": btc,
            "eth": eth,
            "bnb": bnb,
            "wealth_btc": wealth_btc,
            "target_w": target_w,
        }, index=px.index)

        trades_df = pd.DataFrame(trades)

        # ---------- Aggregated performance metrics ----------
        initial_btc = float(df["wealth_btc"].iloc[0])
        final_btc = float(df["wealth_btc"].iloc[-1])
        n_bars = max(len(df) - 1, 1)

        # Simple per-bar returns on total wealth
        ret = df["wealth_btc"].pct_change().fillna(0.0)

        # Annualization based on the same bar interval convention as the strategy
        try:
            bar_minutes = float(getattr(strat.p, "bar_interval_minutes", 1))
        except Exception:
            bar_minutes = 1.0
        bars_per_year = (365.0 * 24.0 * 60.0) / max(bar_minutes, 1.0)

        if initial_btc > 0 and final_btc > 0:
            total_return = (final_btc / initial_btc) - 1.0
            ann_return = (final_btc / initial_btc) ** (bars_per_year / n_bars) - 1.0
        else:
            total_return = 0.0
            ann_return = 0.0

        # Annualized volatility of wealth returns
        ret_std = float(ret.std(ddof=0)) if len(ret) > 1 else 0.0
        ann_vol = ret_std * math.sqrt(bars_per_year)

        # Max drawdown in BTC and pct terms (relative to running high watermark)
        wealth_series = df["wealth_btc"].astype(float)
        roll_max = wealth_series.cummax()
        dd_btc = (roll_max - wealth_series).max()
        if float(roll_max.max()) > 0:
            dd_pct = float(((wealth_series / roll_max) - 1.0).min())
        else:
            dd_pct = 0.0

        # Sharpe-like metric (rf = 0)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

        turns = int((np.sign(df["target_w"].diff().fillna(0)) != 0).sum())
        n_trades = int(len(trades_df)) if not trades_df.empty else 0

        summary = {
            # existing fields (backwards compatible)
            "final_btc": final_btc,
            "turns": turns,
            "fees_btc": float(total_fees_btc),
            "turnover_btc": float(total_turnover_btc),
            "start": str(df.index[0]),
            "end": str(df.index[-1]),
            # new fields
            "initial_btc": initial_btc,
            "total_return": total_return,
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown_btc": float(dd_btc),
            "max_drawdown_pct": dd_pct,
            "n_bars": int(len(df)),
            "n_trades": n_trades,
        }

        return {
            "portfolio": df,
            "trades": trades_df,
            "summary": summary,
        }

# ------------------ Helpers ------------------
def _align_series(base_index, other: pd.Series) -> pd.Series:
    s = other.copy().sort_index()
    return s.reindex(base_index, method="ffill")

def _resolve(prefer_cli: bool, arg_val, cfg: Dict, key: str, fallback):
    if prefer_cli and arg_val is not None:
        return arg_val
    if key in cfg and cfg[key] is not None:
        return cfg[key]
    return fallback

def _load_params_from_cfg_args(cfg: Dict, args) -> StratParams:
    cfg_trend_kind = cfg.get("trend_kind", cfg.get("strategy"))
    trend_kind = _resolve(args.prefer_cli, args.strategy, cfg, "trend_kind", cfg_trend_kind or "sma")
    trend_lookback = _resolve(args.prefer_cli, args.trend, cfg, "trend_lookback", cfg.get("trend", 200))
    return StratParams(
        trend_kind = "roc" if str(trend_kind).lower() == "roc" else "sma",
        trend_lookback = int(trend_lookback),
        flip_band_entry = float(_resolve(args.prefer_cli, args.flip_band_entry, cfg, "flip_band_entry", cfg.get("flip_band", 0.025))),
        flip_band_exit  = float(_resolve(args.prefer_cli, args.flip_band_exit,  cfg, "flip_band_exit",  cfg.get("flip_band", 0.015))),
        vol_window      = int(_resolve(args.prefer_cli, args.vol_window,       cfg, "vol_window", 60)),
        vol_adapt_k     = float(_resolve(args.prefer_cli, args.vol_adapt_k,    cfg, "vol_adapt_k", 0.0)),
        target_vol      = float(_resolve(args.prefer_cli, args.target_vol,     cfg, "target_vol", 0.5)),
        min_mult        = float(_resolve(args.prefer_cli, args.min_mult,       cfg, "min_mult", 0.5)),
        max_mult        = float(_resolve(args.prefer_cli, args.max_mult,       cfg, "max_mult", 1.5)),
        cooldown_minutes= int(_resolve(args.prefer_cli, args.cooldown_minutes, cfg, "cooldown_minutes", 180)),
        step_allocation = float(_resolve(args.prefer_cli, args.step_allocation,cfg, "step_allocation", 0.5)),
        max_position    = float(_resolve(args.prefer_cli, args.max_position,   cfg, "max_position", 1.0)),
        rebalance_threshold_w = float(_resolve(args.prefer_cli, args.rebalance_threshold_w, cfg, "rebalance_threshold_w", 0.0)),
        min_trade_btc         = float(_resolve(args.prefer_cli, args.min_trade_btc,         cfg, "min_trade_btc", 0.0)),
        gate_window_days = int(_resolve(args.prefer_cli, args.gate_window_days, cfg, "gate_window_days", 60)),
        gate_roc_threshold = float(_resolve(args.prefer_cli, args.gate_roc_threshold, cfg, "gate_roc_threshold", 0.0)),
        bar_interval_minutes = int(_resolve(args.prefer_cli,getattr(args, "bar_interval_minutes", None),cfg, "bar_interval_minutes", cfg.get("bar_interval_minutes", 15))
        ),
        long_only = bool(_resolve(args.prefer_cli, getattr(args, "long_only", None), cfg, "long_only", True)),

    )

def _parse_utc(ts: Optional[str]) -> Optional[pd.Timestamp]:
    if not ts: return None
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    if t is pd.NaT: return None
    return t

# ------------------ Commands ------------------
def cmd_backtest(args):
    df = load_vision_csv(args.data); close = df["close"]

    # Optional date slicing for apples-to-apples comparisons
    start_ts = _parse_utc(args.start)
    end_ts   = _parse_utc(args.end)
    if start_ts is not None or end_ts is not None:
        close = close.loc[
            (close.index >= (start_ts or close.index.min())) &
            (close.index <= (end_ts   or close.index.max()))
        ]
        if len(close) < 2:
            raise ValueError("After applying --start/--end, not enough data points to run.")

    cfg = load_json_config(getattr(args, "config", None))

    # Optional BNB price series
    bnb_series = None
    bnb_path = cfg.get("bnb_data", args.bnb_data)
    if bnb_path:
        df_bnb = load_vision_csv(bnb_path)
        bnb_series = _align_series(close.index, df_bnb["close"])

    # --- resolve fees and basis first ---
    fee = FeeParams(
        maker_fee=float(_resolve(args.prefer_cli, args.maker_fee, cfg, "maker_fee", 0.0002)),
        taker_fee=float(_resolve(args.prefer_cli, args.taker_fee, cfg, "taker_fee", 0.0004)),
        slippage_bps=float(_resolve(args.prefer_cli, args.slippage_bps, cfg, "slippage_bps", 1.0)),
        bnb_discount=float(_resolve(args.prefer_cli, args.bnb_discount, cfg, "bnb_discount", 0.25)),
        pay_fees_in_bnb=bool(_resolve(args.prefer_cli, (not args.no_bnb), cfg, "pay_fees_in_bnb", True)),
    )
    basis = float(_resolve(args.prefer_cli, args.basis_btc, cfg, "basis_btc", 1.0))

    # Risk mode: how to interpret fractional limits
    risk_mode = _resolve(
        args.prefer_cli,
        getattr(args, "risk_mode", None),
        cfg,
        "risk_mode",
        "fixed_basis",
    )

    # Absolute BTC limits (optional)
    max_daily_loss_btc_raw = float(_resolve(
        args.prefer_cli,
        getattr(args, "max_daily_loss_btc", None),
        cfg,
        "max_daily_loss_btc",
        0.0,
    ))
    max_dd_btc_raw = float(_resolve(
        args.prefer_cli,
        getattr(args, "max_dd_btc", None),
        cfg,
        "max_dd_btc",
        0.0,
    ))

    # Fractional limits (optional)
    max_daily_loss_frac = float(_resolve(
        args.prefer_cli,
        getattr(args, "max_daily_loss_frac", None),
        cfg,
        "max_daily_loss_frac",
        0.0,
    ))
    max_dd_frac = float(_resolve(
        args.prefer_cli,
        getattr(args, "max_dd_frac", None),
        cfg,
        "max_dd_frac",
        0.0,
    ))

    # Effective BTC limits:
    # - fixed_basis: fractions are relative to basis_btc
    # - dynamic: fractions are applied inside Backtester based on current equity
    if risk_mode == "dynamic":
        # In dynamic mode, use raw BTC limits only if explicitly set.
        max_daily_loss = max_daily_loss_btc_raw
        max_dd = max_dd_btc_raw
    else:
        # fixed_basis (default / legacy behaviour)
        if max_daily_loss_frac > 0.0 and max_daily_loss_btc_raw == 0.0:
            max_daily_loss = basis * max_daily_loss_frac
        else:
            max_daily_loss = max_daily_loss_btc_raw

        if max_dd_frac > 0.0 and max_dd_btc_raw == 0.0:
            max_dd = basis * max_dd_frac
        else:
            max_dd = max_dd_btc_raw

    risk_info = {
        "risk_mode": risk_mode,
        "basis_btc": basis,
        "max_daily_loss_btc": max_daily_loss,
        "max_dd_btc": max_dd,
        "max_daily_loss_frac": max_daily_loss_frac,
        "max_dd_frac": max_dd_frac,
    }
    # --- compute min_trade_btc from fraction BEFORE building StratParams ---
    # CLI fraction wins if absolute not given on CLI
    if getattr(args, "min_trade_frac", None) is not None and args.min_trade_btc is None:
        cfg = dict(cfg)
        cfg["min_trade_btc"] = float(args.min_trade_frac) * basis

    # JSON fraction used if absolute missing/None/≤0 and no CLI absolute
    if args.min_trade_btc is None:
        cfg_abs = cfg.get("min_trade_btc")
        needs_compute = (cfg_abs is None) or (isinstance(cfg_abs, (int, float)) and cfg_abs <= 0)
        if needs_compute:
            mtf = cfg.get("min_trade_frac")
            try:
                if mtf is not None and float(mtf) > 0:
                    cfg = dict(cfg)
                    cfg["min_trade_btc"] = float(mtf) * basis
            except (TypeError, ValueError):
                pass

    # --- NOW build params so p.min_trade_btc reflects the computed value ---
    p = _load_params_from_cfg_args(cfg, args)

    # Print effective params, fees, and window for audit
    print(json.dumps({
        "effective_params": p.__dict__,
        "fees": fee.__dict__,
        "risk": risk_info,
        "window": {"start": str(close.index.min()), "end": str(close.index.max()), "rows": int(len(close))}
    }, indent=2, default=str))

    bt = Backtester(fee)
    res = bt.simulate(
        close,
        EthBtcStrategy(p),
        initial_btc=basis,
        bnb_price_series=bnb_series,
        max_daily_loss_btc=max_daily_loss,
        max_dd_btc=max_dd,
        max_daily_loss_frac=max_daily_loss_frac,
        max_dd_frac=max_dd_frac,
        risk_mode=risk_mode,
    )
    port = res["portfolio"]
    print(json.dumps(res["summary"], indent=2))
    if args.out:
        port.to_csv(args.out)
        print(f"Saved equity → {args.out}")
    if args.excel_out:
        sheets = {"Summary": res["summary"], "Equity": port[["wealth_btc","target_w"]]}
        if not res["trades"].empty: sheets["Trades"] = res["trades"]
        _write_excel(args.excel_out, sheets)
        print(f"Wrote Excel report → {args.excel_out}")

def cmd_optimize(args):
    df = load_vision_csv(args.data); close = df["close"]
    bnb_series = None
    if args.bnb_data:
        df_bnb = load_vision_csv(args.bnb_data); bnb_series = _align_series(close.index, df_bnb["close"])
    fee = FeeParams(maker_fee=args.maker_fee, taker_fee=args.taker_fee,
                    slippage_bps=args.slippage_bps, bnb_discount=args.bnb_discount,
                    pay_fees_in_bnb=not args.no_bnb)

    rows = []
    for _ in range(args.n_random):
        p = StratParams(
            trend_kind=random.choice(["sma","roc"]),
            trend_lookback=random.choice([120,160,200,240,300]),
            flip_band_entry=random.uniform(0.01, 0.05),
            flip_band_exit=random.uniform(0.005,0.03),
            vol_window=random.choice([45,60,90]),
            vol_adapt_k=random.choice([0.0, 0.0025, 0.005, 0.0075]),
            target_vol=random.choice([0.3,0.4,0.5,0.6]),
            min_mult=0.5, max_mult=1.5,
            cooldown_minutes=random.choice([60,120,180,240]),
            step_allocation=random.choice([0.33,0.5,0.66,1.0]),
            max_position=random.choice([0.6,0.8,1.0]),
            rebalance_threshold_w=random.choice([0.0,0.01,0.02]),
            min_trade_btc=random.choice([0.0, 0.00025, 0.0005]),
            gate_window_days=random.choice([30,60,90]),
            gate_roc_threshold=random.choice([0.0, 0.01, 0.02]),
        )
        train = close.loc[args.train_start:args.train_end].dropna()
        test  = close.loc[args.test_start:args.test_end].dropna()
        bt = Backtester(fee)
        res_tr = bt.simulate(train, EthBtcStrategy(p), bnb_price_series=bnb_series)
        res_te = bt.simulate(test,  EthBtcStrategy(p), bnb_price_series=bnb_series)
        rows.append({
            "params": p.__dict__,
            "train_final_btc": res_tr["summary"]["final_btc"],
            "test_final_btc":  res_te["summary"]["final_btc"],
            "turns_test":      res_te["summary"]["turns"],
            "fees_btc":        res_te["summary"]["fees_btc"],
            "turnover_btc":    res_te["summary"]["turnover_btc"],
            "score":           res_te["summary"]["final_btc"],
        })
    out = pd.DataFrame(rows).sort_values("score", ascending=False)
    if args.out:
        out.to_csv(args.out, index=False); print(f"Saved optimization results → {args.out}")
    if args.excel_out:
        _write_excel(args.excel_out, {"Optimization": out}); print(f"Wrote Excel report → {args.excel_out}")

# ------------------ Self-test (quick) ------------------
def cmd_selftest(args):
    idx = pd.date_range("2024-01-01", periods=500, freq="15min", tz="UTC")
    rnd = np.random.RandomState(42)
    ret = rnd.normal(0, 0.0008, size=len(idx)) + 0.00002
    price = 0.05 * (1 + pd.Series(ret, index=idx)).cumprod()
    df = pd.DataFrame({"close": price}, index=idx)
    p = StratParams()
    fee = FeeParams()
    bt = Backtester(fee)
    res = bt.simulate(df["close"], EthBtcStrategy(p))
    port = res["portfolio"]
    print(json.dumps(res["summary"], indent=2))
    if args.out:
        port.to_csv(args.out)
        print(f"Saved equity → {args.out}")
    if args.excel_out:
        _write_excel(args.excel_out, {"Summary": res["summary"], "Equity": port[["wealth_btc","target_w"]]})
        print(f"Wrote Excel report → {args.excel_out}")

# ------------------ CLI ------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ETHBTC BTC-accumulation bot — Rich Mode (failsafe)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # -------- backtest --------
    a = sub.add_parser("backtest")
    a.add_argument("--data", required=True)
    a.add_argument("--config")
    a.add_argument("--bnb-data")
    a.add_argument("--basis-btc", type=float, default=None)
    a.add_argument("--max-daily-loss-btc", type=float, default=None,
                   help="Max daily loss in BTC before trading halts for the day (backtest only).")
    a.add_argument("--max-daily-loss-frac", type=float, default=None,
                   help="Max daily loss as a fraction of basis_btc (e.g. 0.02 = 2%% of basis).")
    a.add_argument("--max-dd-btc", type=float, default=None,
                   help="Max peak-to-trough drawdown in BTC before switching to BTC-only (backtest only).")
    a.add_argument("--max-dd-frac", type=float, default=None,
                   help="Max peak-to-trough drawdown as a fraction of basis_btc or current equity (e.g. 0.15 = 15%%).")
    a.add_argument(
        "--risk-mode",
        choices=["fixed_basis", "dynamic"],
        default=None,
        help=(
            "Risk sizing mode: 'fixed_basis' (limits are fractions of basis_btc) or "
            "'dynamic' (limits are fractions of current equity)."
        ),
    )
    a.add_argument("--min-trade-frac", type=float, default=None)

    # date slicing
    a.add_argument("--start", help="Start date (UTC), e.g. 2024-01-01")
    a.add_argument("--end",   help="End date (UTC), e.g. 2024-09-30")

    # prefer-cli switch
    a.add_argument("--prefer-cli", action="store_true", default=True,
                   help="If set, CLI flags override config values when provided (default: ON).")

    # strategy knobs (defaults None so we can detect explicit flags)
    a.add_argument("--strategy", choices=["sma","roc"], default=None)
    a.add_argument("--trend", type=int, default=None)
    a.add_argument("--flip-band-entry", type=float, default=None)
    a.add_argument("--flip-band-exit", type=float, default=None)
    a.add_argument("--vol-window", type=int, default=None)
    a.add_argument("--bar-interval-minutes", type=int, default=None)
    a.add_argument("--vol-adapt-k", type=float, default=None)
    a.add_argument("--target-vol", type=float, default=None)
    a.add_argument("--min-mult", type=float, default=None)
    a.add_argument("--max-mult", type=float, default=None)
    a.add_argument("--cooldown-minutes", type=int, default=None)
    a.add_argument("--step-allocation", type=float, default=None)
    a.add_argument("--max-position", type=float, default=None)
    a.add_argument("--rebalance-threshold-w", type=float, default=None)
    a.add_argument("--min-trade-btc", type=float, default=None)
    a.add_argument("--gate-window-days", type=int, default=None)
    a.add_argument("--gate-roc-threshold", type=float, default=None)

    # fees/slippage
    a.add_argument("--maker-fee", type=float, default=None)
    a.add_argument("--taker-fee", type=float, default=None)
    a.add_argument("--bnb-discount", type=float, default=None)
    a.add_argument("--no-bnb", action="store_true")
    a.add_argument("--slippage-bps", type=float, default=None)

    a.add_argument("--out")
    a.add_argument("--excel-out")
    a.set_defaults(func=cmd_backtest)

    # -------- optimize --------
    b = sub.add_parser("optimize")
    b.add_argument("--data", required=True)
    b.add_argument("--bnb-data")
    b.add_argument("--train-start", required=True)
    b.add_argument("--train-end", required=True)
    b.add_argument("--test-start", required=True)
    b.add_argument("--test-end", required=True)
    b.add_argument("--n-random", type=int, default=100)
    b.add_argument("--maker-fee", type=float, default=0.0002)
    b.add_argument("--taker-fee", type=float, default=0.0004)
    b.add_argument("--bnb-discount", type=float, default=0.25)
    b.add_argument("--no-bnb", action="store_true")
    b.add_argument("--slippage-bps", type=float, default=1.0)
    b.add_argument("--out")
    b.add_argument("--excel-out")
    b.set_defaults(func=cmd_optimize)

    # -------- selftest --------
    c = sub.add_parser("selftest")
    c.add_argument("--out")
    c.add_argument("--excel-out")
    c.set_defaults(func=cmd_selftest)

    args = ap.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        ap.print_help()
