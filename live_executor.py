# live_executor.py
from __future__ import annotations
import os, json, time, logging, argparse, math, threading
from dotenv import load_dotenv
load_dotenv()
from typing import Dict, Any
import pandas as pd
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Lock
from binance.spot import Spot
from core.config_schema import load_config
from core.binance_adapter import BinanceSpotAdapter
from pathlib import Path

# --- METRICS ---
from core.metrics import (
    ORDERS_SUBMITTED, FILLS, REJECTIONS, PNL_QUOTE, EXPOSURE_W, SPREAD_BPS, BAR_LATENCY, 
    GATE_STATE, SIGNAL_ZONE, TRADE_DECISION, DELTA_W, DELTA_BASE, WEALTH_TOTAL, 
    PRICE_MID, BAL_FREE, SKIPS, PRICE_ASSET_USD, SIGNAL_RATIO, 
    DIST_TO_BUY_BPS, DIST_TO_SELL_BPS, FUNDING_RATE,
    start_metrics_server, mark_gate, mark_zone, mark_decision, mark_signal_metrics, 
    snapshot_wealth_balances, set_delta_metrics, mark_risk_mode, mark_risk_flags, 
    mark_trade_readiness, mark_funding_rate, mark_asset_price_usd
)
from core.ascii_levelbar import dist_to_buy_sell_bps, ascii_level_bar

# --- STRATEGIES (Safe Import) ---
from core.ethbtc_accum_bot import EthBtcStrategy, StratParams
from core.trend_strategy import TrendStrategy, TrendParams
from core.meta_strategy import MetaStrategy

# --- MAKER LOGIC ---
try:
    from core.twap_maker import maker_chase
except ImportError:
    maker_chase = None
    print("WARNING: core.twap_maker not found. Maker/Post-Only logic disabled.")

# --- Simple JSON /status on :9110 ------------------------------------------------

DECISION_KEYS = (
    "exec_buy", "exec_sell", "skip_threshold", "skip_balance", "skip_min_notional", 
    "skip_cooldown", "skip_gate_closed", "skip_delta_zero","skip_order_error"
)

def reset_trade_decision():
    for k in DECISION_KEYS:
        try:
            TRADE_DECISION.labels(k).set(0)
        except Exception:
            pass

def start_status_server(port: int = 9110):
    log = logging.getLogger("live_enhanced")
    _status_lock = Lock()
    _STATUS = {}

    def update_status(payload: dict = None, **kwargs) -> None:
        if payload is None:
            payload = kwargs
        with _status_lock:
            _STATUS.clear()
            _STATUS.update(payload)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/status":
                with _status_lock:
                    body = json.dumps(_STATUS, separators=(",", ":")).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *args, **kwargs):
            return

    server = HTTPServer(("0.0.0.0", port), Handler)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    log.info("Human status on :%d/status", port)

    return update_status

def inc_rejection(reason: str = "error") -> None:
    try:
        REJECTIONS.labels(reason=reason).inc()
    except Exception:
        REJECTIONS.inc()

logging.basicConfig(level=os.getenv("LOGLEVEL","INFO"))
log = logging.getLogger("live_enhanced")

def last_closed_bar_ts(now_s: int, interval: str) -> int:
    units = {"m":60, "h":3600, "d":86400}
    sec = int(interval[:-1]) * units[interval[-1]]
    return now_s - (now_s % sec) - 1

def load_state(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(path: str, st: Dict[str, Any]) -> None:
    import json, os, errno
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(p) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(st, f, indent=2, sort_keys=True)
        os.replace(tmp, str(p))
    except OSError as e:
        if e.errno in (errno.EROFS,):
            fallback = Path.cwd() / "run_state" / p.name
            fallback.parent.mkdir(parents=True, exist_ok=True)
            tmp = str(fallback) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(st, f, indent=2, sort_keys=True)
            os.replace(tmp, str(fallback))
        else:
            raise

def _ensure_risk_state(state: Dict[str, Any], wealth: float, ts: pd.Timestamp) -> None:
    if "risk_equity_high" not in state:
        state["risk_equity_high"] = wealth
    if "risk_current_date" not in state:
        state["risk_current_date"] = ts.normalize().date().isoformat()
    if "risk_daily_start_wealth" not in state:
        state["risk_daily_start_wealth"] = wealth
    if "risk_daily_limit_hit" not in state:
        state["risk_daily_limit_hit"] = False
    if "risk_maxdd_hit" not in state:
        state["risk_maxdd_hit"] = False

def _update_risk_state(state: Dict[str, Any], wealth: float, ts: pd.Timestamp, cfg) -> None:
    risk = cfg.risk
    risk_mode = getattr(risk, "risk_mode", "fixed_basis")
    max_dd_frac = float(getattr(risk, "max_dd_frac", 0.0) or 0.0)
    max_daily_loss_frac = float(getattr(risk, "max_daily_loss_frac", 0.0) or 0.0)

    equity_high = float(state.get("risk_equity_high", wealth))
    current_date_str = state.get("risk_current_date")
    if current_date_str:
        try:
            current_date = pd.to_datetime(current_date_str).date()
        except Exception:
            current_date = ts.normalize().date()
    else:
        current_date = ts.normalize().date()

    daily_start = float(state.get("risk_daily_start_wealth", wealth))
    daily_limit_hit = bool(state.get("risk_daily_limit_hit", False))
    maxdd_hit = bool(state.get("risk_maxdd_hit", False))

    if wealth > equity_high:
        equity_high = wealth
    dd_now = equity_high - wealth

    if risk_mode == "dynamic":
        if max_dd_frac > 0.0:
            threshold_dd = equity_high * max_dd_frac
        else:
            threshold_dd = float(risk.max_dd_btc)
    else:
        threshold_dd = float(risk.max_dd_btc)

    if threshold_dd > 0.0 and dd_now >= threshold_dd:
        maxdd_hit = True

    cur_date = ts.normalize().date()
    if cur_date != current_date:
        current_date = cur_date
        daily_start = wealth
        daily_limit_hit = False

    daily_pnl = wealth - daily_start

    if risk_mode == "dynamic":
        if max_daily_loss_frac > 0.0:
            threshold_loss = daily_start * max_daily_loss_frac
        else:
            threshold_loss = float(risk.max_daily_loss_btc)
    else:
        threshold_loss = float(risk.max_daily_loss_btc)

    if threshold_loss > 0.0 and daily_pnl <= -threshold_loss:
        daily_limit_hit = True

    state["risk_equity_high"] = equity_high
    state["risk_current_date"] = current_date.isoformat()
    state["risk_daily_start_wealth"] = daily_start
    state["risk_daily_limit_hit"] = daily_limit_hit
    state["risk_maxdd_hit"] = maxdd_hit

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--mode", choices=["dry","testnet","live"], default="dry")
    ap.add_argument("--symbol", default="ETHBTC")
    
    _default_state = os.getenv("STATE_FILE") or (
        "/data/state.json" if Path("/.dockerenv").exists()
        else str(Path.cwd() / "run_state" / "state.json")
    )
    ap.add_argument("--state", default=_default_state, help="state file path")
    ap.add_argument("--once", action="store_true", help="Run one logic loop and exit (for Cron)")

    args = ap.parse_args()

    state_file_name = "state.json"
    if args.state.endswith(state_file_name):
        p = Path(args.state)
        new_name = f"{p.stem}_{args.mode}{p.suffix}"
        args.state = str(p.parent / new_name)
        
    log.name = args.symbol
    log.info("State file: %s", args.state)

    cfg = load_config(args.params)

    env_base = (os.getenv("BINANCE_BASE_URL") or "").strip()
    if env_base:
        base_url = env_base
    elif args.mode == "testnet":
        base_url = "https://testnet.binance.vision"
    else:
        base_url = "https://api.binance.com"

    log.info("Using Binance base_url=%s (mode=%s)", base_url, args.mode)

    client = Spot(
        api_key=os.getenv("BINANCE_KEY", ""),
        api_secret=os.getenv("BINANCE_SECRET", ""),
        base_url=base_url,
    )

    try:
        info = client.exchange_info(symbol=args.symbol)
        s_info = info["symbols"][0]
        base_asset = s_info["baseAsset"]
        quote_asset = s_info["quoteAsset"]
        log.info("Configuration: Trading %s | Base=%s | Quote=%s", args.symbol, base_asset, quote_asset)
    except Exception as e:
        log.error("Could not fetch exchange info for %s: %s", args.symbol, e)
        if args.mode == "dry":
            base_asset = args.symbol.replace("BTC", "").replace("USDT", "")
            quote_asset = "BTC" if "BTC" in args.symbol else "USDT"
            log.warning("Dry run fallback parsing: Base=%s, Quote=%s", base_asset, quote_asset)
        else:
            raise e
        
    adapter = BinanceSpotAdapter(client)
    metrics_port = int(os.getenv("METRICS_PORT", "9109"))
    status_port  = int(os.getenv("STATUS_PORT", "9110"))
    
    start_metrics_server(metrics_port)
    update_status = start_status_server(status_port)

    state = load_state(args.state)
    if "session_start_W" not in state:
        state["session_start_W"] = 0.0
    last_seen_bar = 0

    while True:
        t0 = time.time()
        now_s = int(t0)
        bar_ts = last_closed_bar_ts(now_s, cfg.execution.interval)
        if last_seen_bar == bar_ts:
            if args.once:
                log.info("Run-once mode: Bar not closed yet, but logic complete. Exiting.")
                break
            time.sleep(cfg.execution.poll_sec)
            continue

        with BAR_LATENCY.time():
            try:
                ks = adapter.get_klines(args.symbol, cfg.execution.interval, limit=600)
                df = pd.DataFrame(ks)
                if "close_time" in df.columns:
                    df.index = pd.to_datetime(df["close_time"], unit="ms", utc=True)
                price = float(df["close"].iloc[-1])
            except Exception as e:
                log.error("Failed to fetch klines: %s", e)
                time.sleep(5)
                continue

            try:
                acct = client.account()
                bal_map = {
                    x["asset"]: float(x["free"]) + float(x["locked"])
                    for x in acct["balances"]
                }
                quote_bal = bal_map.get(quote_asset, 0.0) 
                base_bal  = bal_map.get(base_asset,  0.0) 
                
                if state.get("last_balance_log_ts", 0) < now_s - 300:
                    log.info("[BALANCE] %s=%.6f, %s=%.6f", quote_asset, quote_bal, base_asset, base_bal)
                    state["last_balance_log_ts"] = now_s
            except Exception as e:
                log.error("CRITICAL: Failed to fetch %s account balance. Reason: %s", args.mode.upper(), e)
                quote_bal, base_bal = cfg.risk.basis_btc, 0.0 

                if "last_known_quote" in state:
                    quote_bal = state["last_known_quote"]
                    base_bal = state.get("last_known_base", 0.0)
                    log.warning("Using LAST KNOWN balance: %s=%.6f", quote_asset, quote_bal)
                else:
                    log.warning("Using CONFIG BASIS fallback (Dangerous for Live!): %.6f", quote_bal)

            state["last_known_quote"] = quote_bal
            state["last_known_base"]  = base_bal

            f = adapter.get_filters(args.symbol)
            min_base_val = f.min_notional / max(price, 1e-12)
            
            effective_base = base_bal
            if base_bal > 0 and base_bal < min_base_val:
                effective_base = 0.0
                if state.get("last_dust_log_ts", 0) < now_s - 3600:
                    log.info("[DUST] Masking %.8f %s (Too small).", base_bal, base_asset)
                    state["last_dust_log_ts"] = now_s

            W = quote_bal + base_bal * price
            cur_w = 0.0 if W <= 0 else (effective_base * price) / W            

            if state.get("session_start_W", 0.0) == 0.0 and W > 0:
                state["session_start_W"] = W

            bar_dt = pd.to_datetime(bar_ts, unit="s", utc=True)
            _ensure_risk_state(state, W, bar_dt)
            _update_risk_state(state, W, bar_dt, cfg)

            risk_mode_str = getattr(cfg.risk, "risk_mode", "fixed_basis")
            mark_risk_mode(risk_mode_str)

            daily_limit_hit = bool(state.get("risk_daily_limit_hit", False))
            maxdd_hit = bool(state.get("risk_maxdd_hit", False))
            mark_risk_flags(daily_limit_hit=daily_limit_hit, maxdd_hit=maxdd_hit)

            WEALTH_TOTAL.set(W) 
            PRICE_MID.set(price)
            BAL_FREE.labels(quote_asset.lower()).set(float(quote_bal))
            BAL_FREE.labels(base_asset.lower()).set(float(base_bal))

            q_usd, b_usd = 0.0, 0.0
            try:
                if quote_asset == "USDT":
                    mark_asset_price_usd("usdt", 1.0)
                    q_usd = 1.0
                else:
                    q_usd = adapter.get_usd_price(f"{quote_asset}USDT")
                    mark_asset_price_usd(quote_asset, q_usd)

                b_usd = adapter.get_usd_price(f"{base_asset}USDT")
                mark_asset_price_usd(base_asset, b_usd)
            except Exception:
                pass
            
            log.debug("Wallet: %s=%.8f, %s=%.8f, W=%.8f, cur_w=%.4f", 
                      quote_asset, quote_bal, base_asset, base_bal, W, cur_w)

            # --- INDICATORS (Common) ---
            L = int(cfg.strategy.trend_lookback)
            ser_close = df["close"].astype(float)
            if cfg.strategy.trend_kind == "sma":
                sma = ser_close.rolling(L).mean()
                cur_ratio = float(ser_close.iloc[-1] / max(sma.iloc[-1], 1e-12) - 1.0)
            else:
                prev = ser_close.shift(L).iloc[-1]
                cur_ratio = float(ser_close.iloc[-1] / max(prev, 1e-12) - 1.0)
            rv = float(ser_close.pct_change().rolling(cfg.strategy.vol_window).std().iloc[-1])
            entry = cfg.strategy.flip_band_entry + cfg.strategy.vol_adapt_k * (rv if rv == rv else 0.0)
            exitb = cfg.strategy.flip_band_exit + cfg.strategy.vol_adapt_k * (rv if rv == rv else 0.0)

            # --- GATE & FUNDING CHECKS ---
            gate_ok = True
            gate_reason = "open"
            funding_rate = 0.0
            
            # 1. Trend Gate (Legacy)
            if cfg.strategy.gate_window_days and cfg.strategy.gate_roc_threshold:
                day_close = ser_close.resample("1D").last()
                if len(day_close) > cfg.strategy.gate_window_days:
                    droc = float(day_close.iloc[-1] / max(day_close.shift(cfg.strategy.gate_window_days).iloc[-1], 1e-12) - 1.0)
                    if abs(droc) < cfg.strategy.gate_roc_threshold:
                        gate_ok = False
                        gate_reason = "trend_weak"

            # 2. Funding Gate
            funding_ticker = f"{base_asset}USDT"
            try:
                funding_rate = adapter.get_funding_rate(funding_ticker)  
                mark_funding_rate(funding_rate)
                
                if funding_rate > cfg.strategy.funding_limit_long:
                    # Don't block here, just mark the reason. Strategy logic will handle direction.
                    # But for safety, we flag it.
                    if cur_ratio <= -entry: # Only care if we might buy
                        gate_ok = False
                        gate_reason = f"funding_high ({funding_rate:.4f}%)"
                        log.warning("Gate CLOSE: Market Euphoria! Funding=%.4f%%", funding_rate)

                if funding_rate < cfg.strategy.funding_limit_short:
                    if cur_ratio >= entry: # Only care if we might sell
                        gate_ok = False
                        gate_reason = f"funding_low ({funding_rate:.4f}%)"
                        log.warning("Gate CLOSE: Market Panic! Funding=%.4f%%", funding_rate)
            except Exception as e:
                log.warning("Funding check warning: %s", e)


            # --- STRATEGY EXECUTION ---
            target_w = None
            strat_type = getattr(cfg.strategy, "strategy_type", "mean_reversion")

            try:
                if strat_type == "trend":
                    # TREND Strategy
                    tp = TrendParams(
                        fast_period=cfg.strategy.fast_period,
                        slow_period=cfg.strategy.slow_period,
                        ma_type=cfg.strategy.ma_type,
                        cooldown_minutes=cfg.strategy.cooldown_minutes,
                        step_allocation=cfg.strategy.step_allocation,
                        max_position=cfg.strategy.max_position,
                        long_only=cfg.strategy.long_only,
                        funding_limit_long=cfg.strategy.funding_limit_long,
                        funding_limit_short=cfg.strategy.funding_limit_short
                    )
                    strat = TrendStrategy(tp)
                    plan = strat.generate_positions(df) # Requires OHLC
                    target_w = float(plan["target_w"].iloc[-1])
                    
                elif strat_type == "meta":
                    # META Strategy
                    # 1. Extract Mean Reversion Params
                    mr_p = StratParams(
                        trend_kind=cfg.strategy.trend_kind,
                        trend_lookback=cfg.strategy.trend_lookback,
                        flip_band_entry=cfg.strategy.flip_band_entry,
                        flip_band_exit=cfg.strategy.flip_band_exit,
                        vol_window=cfg.strategy.vol_window,
                        vol_adapt_k=cfg.strategy.vol_adapt_k,
                        target_vol=getattr(cfg.strategy, "target_vol", 0.5),
                        cooldown_minutes=cfg.strategy.cooldown_minutes,
                        step_allocation=cfg.strategy.step_allocation,
                        max_position=cfg.strategy.max_position,
                        rebalance_threshold_w=cfg.strategy.rebalance_threshold_w,
                        min_trade_btc=getattr(cfg.strategy, "min_trade_btc", 0.0),
                        gate_window_days=cfg.strategy.gate_window_days,
                        gate_roc_threshold=cfg.strategy.gate_roc_threshold,
                        long_only=getattr(cfg.strategy, "long_only", True),
                        funding_limit_long=cfg.strategy.funding_limit_long,
                        funding_limit_short=cfg.strategy.funding_limit_short
                    )
                    # 2. Extract Trend Params
                    tr_p = TrendParams(
                        fast_period=cfg.strategy.fast_period,
                        slow_period=cfg.strategy.slow_period,
                        ma_type=cfg.strategy.ma_type,
                        cooldown_minutes=cfg.strategy.cooldown_minutes,
                        step_allocation=cfg.strategy.step_allocation,
                        max_position=cfg.strategy.max_position,
                        long_only=cfg.strategy.long_only,
                        funding_limit_long=cfg.strategy.funding_limit_long,
                        funding_limit_short=cfg.strategy.funding_limit_short
                    )
                    strat = MetaStrategy(mr_p, tr_p, adx_threshold=cfg.strategy.adx_threshold)
                    plan = strat.generate_positions(df) # OHLC
                    target_w = float(plan["target_w"].iloc[-1])

                else:
                    # DEFAULT: Mean Reversion
                    sp = StratParams(
                        trend_kind=cfg.strategy.trend_kind,
                        trend_lookback=cfg.strategy.trend_lookback,
                        flip_band_entry=cfg.strategy.flip_band_entry,
                        flip_band_exit=cfg.strategy.flip_band_exit,
                        vol_window=cfg.strategy.vol_window,
                        vol_adapt_k=cfg.strategy.vol_adapt_k,
                        target_vol=getattr(cfg.strategy, "target_vol", 0.5),
                        cooldown_minutes=cfg.strategy.cooldown_minutes,
                        step_allocation=cfg.strategy.step_allocation,
                        max_position=cfg.strategy.max_position,
                        rebalance_threshold_w=cfg.strategy.rebalance_threshold_w,
                        min_trade_btc=getattr(cfg.strategy, "min_trade_btc", 0.0),
                        gate_window_days=cfg.strategy.gate_window_days,
                        gate_roc_threshold=cfg.strategy.gate_roc_threshold,
                        long_only=getattr(cfg.strategy, "long_only", True),
                        funding_limit_long=cfg.strategy.funding_limit_long,
                        funding_limit_short=cfg.strategy.funding_limit_short
                    )
                    strat = EthBtcStrategy(sp)
                    # Note: EthBtcStrategy usually takes 'close' series. 
                    # The new one we updated takes (close, funding).
                    # Live execution logic for funding is handled via gate_ok mostly, 
                    # but let's pass None to generate_positions and rely on safety override below.
                    plan = strat.generate_positions(ser_close) 
                    target_w = float(plan["target_w"].iloc[-1])

            except Exception as e:
                log.error("Strategy calculation failed (%s): %s", strat_type, e)
                target_w = cur_w # Hold position on error

            # --- SAFETY OVERRIDE ---
            # If the "Global Gate" is closed (due to funding or trend check in main loop),
            # we must prevent BUYING.
            if not gate_ok:
                # If strategy wants to increase exposure, block it.
                if target_w > cur_w:
                    log.warning("Safety Override: Strategy wants %.2f, but Gate is CLOSED. Holding %.2f.", target_w, cur_w)
                    target_w = cur_w
            # -----------------------

            target_w = max(0.0, min(1.0, target_w))

            reset_trade_decision()

            delta_eth = 0.0
            delta_w = 0.0
            side = "HOLD"

            meter = ascii_level_bar(cur_ratio, entry, exitb, width=64)
            dist_to_buy_bps, dist_to_sell_bps = dist_to_buy_sell_bps(cur_ratio, entry, exitb)
            mark_signal_metrics(cur_ratio, dist_to_buy_bps, dist_to_sell_bps)

            snapshot_wealth_balances(W, price, quote_bal, base_bal, quote_asset, base_asset)
            
            gate_display = "OPEN" if gate_ok else f"CLOSED ({gate_reason})"
            log.info("[SIG] ratio=%+0.4f  bands: -entry=%0.4f  -exit=%0.4f  +exit=%0.4f  +entry=%0.4f  gate=%s  %s",
                    cur_ratio, -entry, -exitb, exitb, entry, gate_display, meter)

            mark_gate(gate_ok)

            if cur_ratio <= -entry:
                zone = "buy_band"
            elif cur_ratio >= entry:
                zone = "sell_band"
            else:
                zone = "neutral"
            mark_zone(zone)

            action_side = ('BUY' if (target_w > cur_w) else ('SELL' if (target_w < cur_w) else 'HOLD'))
            step = cfg.strategy.step_allocation

            # --- SNAP-TO-ZERO (Generic) ---
            if target_w == 0.0 and base_bal > 0:
                total_val_quote = base_bal * price
                min_trade_quote = max(cfg.execution.min_trade_floor_btc,
                                cfg.execution.min_trade_btc or (cfg.execution.min_trade_frac * cfg.risk.basis_btc))
                
                if total_val_quote > min_trade_quote and total_val_quote < (3.0 * min_trade_quote):
                    step = 1.0
                    log.info("[EXEC] Snap-to-Zero: %s position small (%.6f %s). Forcing exit.", base_asset, total_val_quote, quote_asset)

            if getattr(cfg.strategy, "vol_scaled_step", False):
                z = cur_ratio / max(rv, 1e-6)
                step = min(1.0, max(0.1, step * min(2.0, abs(z))))

            # Final target calculation with smoothing
            new_w_ideal = cur_w + step * (target_w - cur_w)
            new_w = max(0.0, min(1.0, new_w_ideal))

            if maxdd_hit:
                if new_w > 0.0:
                    log.warning("Risk: max drawdown hit → forcing Exit (new_w=0.0).")
                target_w = 0.0
                new_w = 0.0
            elif daily_limit_hit:
                if abs(new_w - cur_w) > 1e-12:
                    log.warning("Risk: daily loss limit hit → freezing position at w=%.4f.", cur_w)
                new_w = cur_w

            delta_w = new_w - cur_w
            delta_eth = delta_w * W / max(price, 1e-12)
            set_delta_metrics(delta_w, delta_eth)

            # Spread Probe
            sp_bps = 0.0
            try:
                book = adapter.get_book(args.symbol)
                sp_bps = 1e4 * (book.best_ask - book.best_bid) / max(price, 1e-12)
                SPREAD_BPS.set(sp_bps)
            except Exception as e:
                log.warning("Spread probe failed. Reason: %s", e)

            action_side = ('BUY' if (target_w > cur_w) else ('SELL' if (target_w < cur_w) else 'HOLD'))
            
            update_status({
                "mode": args.mode,
                "symbol": args.symbol,
                "price": price,
                "wealth_btc": W,
                "cur_w": cur_w,
                "target_w": target_w,
                "step": step,
                "delta_w_planned": delta_w,
                "delta_base_planned": delta_eth,
                "base_asset": base_asset,
                "side": action_side,
                "ratio": cur_ratio,
                "entry": entry,
                "bands": {
                    "neg_entry": -entry,
                    "neg_exit":  -exitb,
                    "pos_exit":   exitb,
                    "pos_entry":  entry,
                },
                "quote_usd": q_usd,
                "base_usd": b_usd,
                "funding_rate": funding_rate,
                "gate": ("OPEN" if gate_ok else "CLOSED"),
                "gate_reason": gate_reason, 
                "dist_to_buy_bps": round(dist_to_buy_bps, 1),
                "dist_to_sell_bps": round(dist_to_sell_bps, 1),
                "zone": zone,
                "ascii_meter": meter,
                "risk": {
                    "mode": getattr(cfg.risk, "risk_mode", "fixed_basis"),
                    "max_daily_loss_btc": float(getattr(cfg.risk, "max_daily_loss_btc", 0.0) or 0.0),
                    "max_dd_btc": float(getattr(cfg.risk, "max_dd_btc", 0.0) or 0.0),
                    "max_daily_loss_frac": float(getattr(cfg.risk, "max_daily_loss_frac", 0.0) or 0.0),
                    "max_dd_frac": float(getattr(cfg.risk, "max_dd_frac", 0.0) or 0.0),
                    "maxdd_hit": bool(state.get("risk_maxdd_hit", False)),
                    "daily_limit_hit": bool(state.get("risk_daily_limit_hit", False)),
                },
            })

            log.info(
                "[STATUS] %s | px=%.8f W=%.6f | zone=%s gate=%s | w: cur=%.4f tgt=%.4f step=%.2f | "
                "plan: Δw=%+.4f (Δ%s=%+.6f) → %s | dist: BUY=%0.1fbps SELL=%0.1fbps",
                args.mode, price, W,
                zone, ("OPEN" if gate_ok else "CLOSED"),
                cur_w, target_w, step,
                delta_w, base_asset, delta_eth,
                action_side,
                dist_to_buy_bps, dist_to_sell_bps
            )

            zone_ok = (zone == "buy_band") or (zone == "sell_band")
            gate_open_ok = bool(gate_ok)
            abs_delta_for_ready = abs(delta_w)
            delta_ok = abs_delta_for_ready >= float(cfg.strategy.rebalance_threshold_w)
            risk_ok = not (daily_limit_hit or maxdd_hit)

            mark_trade_readiness(
                zone_ok=zone_ok,
                gate_ok=gate_open_ok,
                delta_ok=delta_ok,
                risk_ok=risk_ok,
                balance_ok=True,
                size_ok=True,
            )

            tol = 1e-12
            abs_delta = abs(delta_w)

            if abs_delta < tol:
                SKIPS.labels("delta_zero").inc()
                mark_decision("skip_delta_zero")
                log.info("Skip: cur_w==target_w==%.4f (no change).", cur_w)
                state["last_target_w"] = target_w
                state["last_bar_close"] = bar_ts
                save_state(args.state, state)
                EXPOSURE_W.labels("target").set(target_w)
                EXPOSURE_W.labels("current").set(cur_w)
                PNL_QUOTE.set(W - float(state.get("session_start_W", W)))
                last_seen_bar = bar_ts
                if args.once:
                    log.info("Run-once complete (no-op).")
                    break
                time.sleep(cfg.execution.poll_sec)
                continue

            if abs_delta < cfg.strategy.rebalance_threshold_w:
                SKIPS.labels('threshold').inc()
                mark_decision("skip_threshold")   
                log.info(
                    "Skip: |Δw|=%.4f < threshold=%.4f (need ≥ %.4f).",
                    abs_delta, cfg.strategy.rebalance_threshold_w, cfg.strategy.rebalance_threshold_w
                )
                state["last_target_w"] = target_w
                state["last_bar_close"] = bar_ts
                save_state(args.state, state)
                EXPOSURE_W.labels("target").set(target_w)
                EXPOSURE_W.labels("current").set(cur_w)
                PNL_QUOTE.set(W - float(state.get("session_start_W", W)))
                last_seen_bar = bar_ts
                if args.once:
                    log.info("Run-once complete (skip threshold).")
                    break
                time.sleep(cfg.execution.poll_sec)
                continue

            side = "BUY" if delta_eth > 0 else "SELL"

            if side == "SELL" and base_bal <= 1e-12:
                SKIPS.labels("balance").inc()
                mark_decision("skip_balance")
                log.info("Skip: SELL requested but %s balance is 0 (cur_w=%.4f, target_w=%.4f).", base_asset, cur_w, target_w)
                state["last_target_w"] = target_w
                state["last_bar_close"] = bar_ts
                save_state(args.state, state)
                EXPOSURE_W.labels("target").set(target_w)
                EXPOSURE_W.labels("current").set(cur_w)
                PNL_QUOTE.set(W - float(state.get("session_start_W", W)))
                last_seen_bar = bar_ts
                if args.once:
                    log.info("Run-once complete (skip balance).")
                    break
                time.sleep(cfg.execution.poll_sec)
                continue

            qty_abs = abs(delta_eth)
            qty_rounded = adapter.round_qty(qty_abs, f.step_size)

            min_trade_quote = max(cfg.execution.min_trade_floor_btc,
                                cfg.execution.min_trade_btc or (cfg.execution.min_trade_frac * cfg.risk.basis_btc))
            if cfg.execution.min_trade_cap_btc > 0:
                min_trade_quote = min(min_trade_quote, cfg.execution.min_trade_cap_btc)

            notional_quote = qty_rounded * price
            need = max(min_trade_quote, f.min_notional)
            
            if notional_quote < need:
                SKIPS.labels("min_notional").inc()
                mark_decision("skip_min_notional")
                log.info("Skip: notional below minimum (%.8f %s < min %.8f)", notional_quote, quote_asset, need)
                state["last_target_w"] = target_w
                state["last_bar_close"] = bar_ts
                save_state(args.state, state)
                last_seen_bar = bar_ts
                if args.once:
                    log.info("Run-once complete (skip min_notional).")
                    break
                time.sleep(cfg.execution.poll_sec)
                continue

            # ---- Balance-aware clamp (Generic) ----------------------------
            if side == "BUY":
                max_qty_by_balance = adapter.round_qty(max((quote_bal * 0.999) / max(price, 1e-12), 0.0), f.step_size)
            else:  
                max_qty_by_balance = adapter.round_qty(max(base_bal, 0.0), f.step_size)

            qty_exec = min(qty_rounded, max_qty_by_balance)

            if qty_exec <= 0:
                SKIPS.labels("balance").inc()
                mark_decision("skip_balance")
                log.info(
                    "Skip: insufficient free balance for %s (qty_rounded=%.8f, max_by_balance=%.8f, %s=%.8f, %s=%.8f)",
                    side, qty_rounded, max_qty_by_balance, quote_asset, quote_bal, base_asset, base_bal
                )
                inc_rejection("insufficient_balance")
                state["last_target_w"] = target_w
                state["last_bar_close"] = bar_ts
                save_state(args.state, state)
                last_seen_bar = bar_ts
                if args.once:
                    log.info("Run-once complete (skip balance clamped).")
                    break
                time.sleep(cfg.execution.poll_sec)            
                continue

            notional_quote = qty_exec * price
            
            if notional_quote < need:
                SKIPS.labels("min_notional").inc()
                mark_decision('skip_min_notional')
                log.info(
                    "Skip: notional below minimum after clamp (%.8f %s < min %.8f) [side=%s qty=%.8f]",
                    notional_quote, quote_asset, need, side, qty_exec
                )
                state["last_target_w"] = target_w
                state["last_bar_close"] = bar_ts
                save_state(args.state, state)
                last_seen_bar = bar_ts
                if args.once:
                    log.info("Run-once complete (skip min_notional clamped).")
                    break
                time.sleep(cfg.execution.poll_sec)            
                continue

            # ---- EXECUTION: MAKER vs TAKER -------------------------------
            use_maker = False
            if cfg.execution.taker_fallback and maker_chase:
                use_maker = True

            executed_qty = 0.0
            
            if args.mode == "dry":
                ORDERS_SUBMITTED.labels("MARKET", side).inc()
                mark_decision("exec_buy" if side == "BUY" else "exec_sell")
                log.info(
                    "[DRY] Would EXEC %s %0.8f @ ~%0.8f (Δw=%+.4f, Δ%s=%+.6f)",
                    side, qty_exec, price, delta_w, base_asset, delta_eth
                )
                executed_qty = qty_exec # Assume fill
            else:
                # --- MAKER EXECUTION ---
                if use_maker:
                    log.info("Attempting MAKER execution for %s %s...", side, qty_exec)
                    try:
                        filled_maker = maker_chase(
                            adapter, args.symbol, side, qty_exec, f.tick_size,
                            max_reprices=3, step_sec=8
                        )
                        executed_qty += filled_maker
                        if filled_maker > 0:
                            FILLS.inc(filled_maker)
                            log.info("MAKER Filled: %.8f / %.8f", filled_maker, qty_exec)
                    except Exception as e:
                        log.error("Maker chase error: %s", e)
                
                # --- TAKER FALLBACK ---
                remaining = qty_exec - executed_qty
                rem_notional = remaining * price
                
                if remaining > 0 and rem_notional >= need:
                    if use_maker:
                        log.info("Falling back to TAKER for remaining %.8f...", remaining)
                    
                    try:
                        oid = adapter.market_order(args.symbol, side, remaining)
                        ORDERS_SUBMITTED.labels("MARKET", side).inc()
                        mark_decision("exec_buy" if side == "BUY" else "exec_sell")
                        log.info("EXEC TAKER %s %0.8f %s @ ~%0.8f (oid=%s)", side, remaining, args.symbol, price, oid)
                        
                        if side == "BUY":
                            TRADE_DECISION.labels("exec_buy").set(1)
                        else:
                            TRADE_DECISION.labels("exec_sell").set(1)
                        executed_qty += remaining
                    except Exception as e:
                        msg = str(e)
                        reason = "insufficient_balance" if "-2010" in msg else "order_error"
                        inc_rejection(reason)
                        log.exception("Taker order rejected: %s", e)
                        # If we filled nothing, we effectively skipped/failed
                        if executed_qty == 0:
                            state["last_target_w"] = target_w
                            state["last_bar_close"] = bar_ts
                            save_state(args.state, state)
                            last_seen_bar = bar_ts
                            mark_decision("skip_order_error")
                            if args.once: break
                            time.sleep(cfg.execution.poll_sec)
                            continue

                elif remaining > 0:
                    log.info("Remainder %.8f too small for Taker (Notional %.8f < %.8f). Stopping.", remaining, rem_notional, need)

            # Update state/metrics after trade logic
            state["last_target_w"] = target_w
            state["last_bar_close"] = bar_ts
            save_state(args.state, state)

            EXPOSURE_W.labels("target").set(target_w)
            EXPOSURE_W.labels("current").set(new_w)
            PNL_QUOTE.set(W - float(state.get("session_start_W", W)))
            last_seen_bar = bar_ts
            
            if args.once:
                log.info("Run-once complete (executed).")
                break

        # pacing
        dt = time.time() - t0
        sleep_left = max(0.0, cfg.execution.poll_sec - dt)
        time.sleep(sleep_left)

if __name__ == "__main__":
    main()