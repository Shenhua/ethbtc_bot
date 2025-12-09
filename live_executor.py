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
from binance.um_futures import UMFutures
from core.futures_adapter import BinanceFuturesAdapter
from core.config_schema import load_config
from core.binance_adapter import BinanceSpotAdapter
from pathlib import Path
from core.alert_manager import AlertManager
# --- METRICS ---
from core.metrics import (
    ORDERS_SUBMITTED, FILLS, REJECTIONS, PNL_QUOTE, EXPOSURE_W, SPREAD_BPS, BAR_LATENCY, 
    GATE_STATE, SIGNAL_ZONE, TRADE_DECISION, DELTA_W, DELTA_BASE, WEALTH_TOTAL, WEALTH_USD, 
    PRICE_MID, BAL_FREE, SKIPS, PRICE_ASSET_USD, SIGNAL_RATIO, SIGNAL_BAND,
    DIST_TO_BUY_BPS, DIST_TO_SELL_BPS, FUNDING_RATE,REGIME_SCORE,REGIME_THRESHOLD,STRATEGY_MODE, 
    REGIME_STATE, PHOENIX_ACTIVE, POSITION_STEP, REALIZED_VOL,
    LEVERAGE, EXPOSURE_SIGNAL_WEIGHT, EXPOSURE_NOTIONAL, CONFIG_LONG_ONLY,
    FEES_PAID, LIQ_DIST, SLIPPAGE, LAST_TRADE_TS,
    start_metrics_server, mark_gate, mark_zone, mark_decision, mark_signal_metrics, 
    snapshot_wealth_balances, set_delta_metrics, mark_risk_mode, mark_risk_flags, 
    mark_trade_readiness, mark_funding_rate, mark_asset_price_usd,
    mark_futures_risk, mark_execution_stats, mark_bot_up,
)
from core.ascii_levelbar import dist_to_buy_sell_bps, ascii_level_bar
from core.story_writer import StoryWriter

# --- STRATEGIES (Safe Import) ---
from core.ethbtc_accum_bot import EthBtcStrategy, StratParams
from core.trend_strategy import TrendStrategy, TrendParams
from core.meta_strategy import MetaStrategy
from core.regime import get_regime_score

# --- SHARED STRATEGY FACTORY (Parity with Backtest) ---
from core.strategy_factory import (
    merge_strategy_params, build_strategy, 
    build_mr_params, build_tr_params, get_active_params
)

log = logging.getLogger("live_executor")


# --- CONFIG PARITY HELPER ---
# This function matches the merge logic in ethbtc_accum_bot.py:build_strategy_from_config()
# to ensure live execution uses the same params as backtest
def merge_strategy_params(cfg) -> dict:
    """
    Merge base strategy config with overrides for meta strategy.
    Returns a dict with 'mr_params' and 'tr_params' dicts containing all merged values.
    For non-meta strategies, returns the base strategy values.
    """
    base = cfg.strategy.model_dump() if hasattr(cfg.strategy, 'model_dump') else dict(cfg.strategy)
    strategy_type = base.get("strategy_type", "mean_reversion")
    
    if strategy_type == "meta":
        mr_opts = base.get("mean_reversion_overrides", {})
        tr_opts = base.get("trend_overrides", {})
        
        # Merge: base + overrides (overrides win)
        mr_merged = {**base, **mr_opts}
        tr_merged = {**base, **tr_opts}
    else:
        # Non-meta: both use base params
        mr_merged = base.copy()
        tr_merged = base.copy()
    
    return {
        "strategy_type": strategy_type,
        "mr_params": mr_merged,
        "tr_params": tr_merged,
        "base_params": base,
    }


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

def reset_trade_decision(instance_name: str):
    """Reset all trade decision metrics to 0 for a specific instance."""
    for k in DECISION_KEYS:
        try:
            TRADE_DECISION.labels(instance=instance_name, trade_decision=k).set(0)
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

def inc_rejection(instance: str, reason: str = "error") -> None:
    try:
        REJECTIONS.labels(instance=instance, reason=reason).inc()
    except Exception:
        REJECTIONS.labels(instance=instance, reason="error").inc()

logging.basicConfig(level=os.getenv("LOGLEVEL","INFO"))
log = logging.getLogger("live_enhanced")

# Global Stop Event for graceful shutdown of maker threads
STOP_EVENT = threading.Event()

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
    """
    Updates the risk state (High Water Mark, Daily Loss, Max Drawdown).
    Detects crashes but DOES NOT handle resets (Phoenix Protocol).
    Resets are handled in the main execution loop.
    """
    risk = cfg.risk
    risk_mode = getattr(risk, "risk_mode", "fixed_basis")
    max_dd_frac = float(getattr(risk, "max_dd_frac", 0.0) or 0.0)
    max_daily_loss_frac = float(getattr(risk, "max_daily_loss_frac", 0.0) or 0.0)

    # 1. Load Current State
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

    # 2. Update High Water Mark (Only if we aren't already crashed)
    # If we are in MaxDD state, we do NOT update HWM (we are in the penalty box)
    if not maxdd_hit:
        if wealth > equity_high:
            equity_high = wealth
    
    dd_now = equity_high - wealth

    # 3. Check for Max Drawdown Violation
    if not maxdd_hit:
        if risk_mode == "dynamic":
            if max_dd_frac > 0.0:
                threshold_dd = equity_high * max_dd_frac
            else:
                threshold_dd = float(risk.max_dd_btc)
        else:
            threshold_dd = float(risk.max_dd_btc)

        if threshold_dd > 0.0 and dd_now >= threshold_dd:
            maxdd_hit = True
            # Record Time of Death for Phoenix Protocol
            state["risk_maxdd_hit_ts"] = ts.isoformat() 

    # 4. Check for Daily Loss Violation
    cur_date = ts.normalize().date()
    if cur_date != current_date:
        # New Day: Reset Daily Counters
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

    # 5. Save State
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

    # Determine instance name for Prometheus metrics labels
    instance_name = os.getenv("INSTANCE_NAME") or f"{args.mode}_{args.symbol.lower()}"
    log.info("Instance name for metrics: %s", instance_name)

    cfg = load_config(args.params)
    
    # === PARITY FIX: Merge strategy params to match backtest behavior ===
    merged_cfg = merge_strategy_params(cfg)
    mr_params = merged_cfg["mr_params"]
    tr_params = merged_cfg["tr_params"]

    state = load_state(args.state)
    if "session_start_W" not in state:
        state["session_start_W"] = 0.0

    env_base = (os.getenv("BINANCE_BASE_URL") or "").strip()
    if env_base:
        base_url = env_base
    elif args.mode == "testnet":
        base_url = "https://testnet.binance.vision"
    else:
        base_url = "https://api.binance.com"


    log.info("🎯 Beginning main loop (interval=%ds, mode=%s)...", cfg.execution.poll_sec, args.mode)
    state["loop_started_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    save_state(args.state, state) # save_state expects path, then state

    log.info("Using Binance base_url=%s (mode=%s)", base_url, args.mode)

    # 1. Determine Mode
    is_futures = (getattr(cfg.execution, "exchange_type", "spot") == "futures")
    
    if is_futures:
        log.info("🚀 STARTING IN FUTURES MODE (USDS-M) 🚀")
        # --- CLIENT SETUP ---
        # Hybrid Setup Support: Use Futures-specific keys if available, else fallback
        # Priority: BINANCE_FUTURES_KEY (Docker mapped) -> FUTURES_TESTNET_KEY (Local .env) -> BINANCE_KEY (Spot/Default)
        f_key = os.getenv("BINANCE_FUTURES_KEY", os.getenv("FUTURES_TESTNET_KEY", os.getenv("BINANCE_KEY", "")))
        f_secret = os.getenv("BINANCE_FUTURES_SECRET", os.getenv("FUTURES_TESTNET_SECRET", os.getenv("BINANCE_SECRET", "")))
        
        client = UMFutures(
            key=f_key,
            secret=f_secret,
            base_url="https://testnet.binancefuture.com" if args.mode == "testnet" else "https://fapi.binance.com"
        )
        adapter = BinanceFuturesAdapter(client)
        
        # Set Leverage on startup
        lev = getattr(cfg.execution, "leverage", 1)
        adapter.set_leverage(args.symbol, lev)
        LEVERAGE.labels(instance=instance_name).set(lev)
        log.info("[FUTURES] Leverage set to %dx for %s", lev, args.symbol)
        
    else:
        # Initialize Spot Client (Existing Logic)
        client = Spot(
            api_key=os.getenv("BINANCE_KEY", ""),
            api_secret=os.getenv("BINANCE_SECRET", ""),
            base_url=base_url,
        )
        adapter = BinanceSpotAdapter(client)
        LEVERAGE.labels(instance=instance_name).set(1)

    # Set Long Only Metric - Use merged MR params (parity with backtest)
    is_long_only = bool(int(mr_params.get("long_only", 1)))
    CONFIG_LONG_ONLY.labels(instance=instance_name).set(1 if is_long_only else 0)

    try:
        # Futures and Spot have different exchange_info() signatures
        if is_futures:
            info = client.exchange_info()  # Futures: no symbol parameter
            s_info = next((s for s in info["symbols"] if s["symbol"] == args.symbol), None)
            if not s_info:
                raise ValueError(f"Symbol {args.symbol} not found in futures exchange info")
        else:
            info = client.exchange_info(symbol=args.symbol)  # Spot: with symbol parameter
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

    # --- STORY WRITER INITIALIZATION ---
    story_file = os.path.join(os.path.dirname(args.state), f"story_{args.symbol.lower()}.txt")
    alerter = AlertManager(prefix=args.symbol)
    story = StoryWriter(story_file, symbol=args.symbol, alerter=alerter)    
    # --- INITIALIZATION ---
        
    # Bug Fix #2: Removed duplicate adapter assignment (adapter already set above)
    metrics_port = int(os.getenv("METRICS_PORT", "9109"))
    status_port  = int(os.getenv("STATUS_PORT", "9110"))
    
    start_metrics_server(metrics_port, story_file=story_file)  # Pass story file for /story endpoint
    update_status = start_status_server(status_port)

    if args.mode in ["live", "testnet"]:
        log.info("Checking for open orders...")
        cancelled = adapter.cancel_open_orders(args.symbol)
        if cancelled:
            log.info(f"Cleaned up {len(cancelled)} zombie orders.")
    try:
        global_filters = adapter.get_filters(args.symbol)
        log.info(f"Loaded Filters: step={global_filters.step_size} tick={global_filters.tick_size} min={global_filters.min_notional}")
    except Exception as e:
        log.error(f"CRITICAL: Could not load filters. {e}")
        # In dry run we might survive, but in live this is fatal.
        if args.mode != "dry": raise e
        # Fallback for dry run
        from core.exchange_adapter import Filters
        global_filters = Filters(0.0001, 0.01, 5.0)
    
    state = load_state(args.state)
    if "session_start_W" not in state:
        state["session_start_W"] = 0.0
    last_seen_bar = 0
    startup_logged_this_run = False

    while True:
        t0 = time.time()
        now_s = int(t0)
        bar_ts = last_closed_bar_ts(now_s, cfg.execution.interval)
        bar_dt = pd.to_datetime(bar_ts, unit="s", utc=True) # Define bar_dt early for story logging

        if last_seen_bar == bar_ts:
            if args.once:
                log.info("Run-once mode: Bar not closed yet, but logic complete. Exiting.")
                break
            time.sleep(cfg.execution.poll_sec)
            continue

        # Mark bot as up/healthy at the start of each bar processing
        mark_bot_up(instance_name, is_up=True)

        # Initial threshold from merged MR params (will be overridden if meta/trend mode)
        active_rebalance_threshold = float(mr_params.get("rebalance_threshold_w", 0.0))

        with BAR_LATENCY.labels(instance=instance_name).time():
            try:
                ks = adapter.get_klines(args.symbol, cfg.execution.interval, limit=600)
                df = pd.DataFrame(ks)
                
                # Ensure numeric types
                cols = ["open", "high", "low", "close", "volume"]
                df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

                if "close_time" in df.columns:
                    df.index = pd.to_datetime(df["close_time"], unit="ms", utc=True)
                
                # --- FIX START: Strict Deduplication ---
                df = df[~df.index.duplicated(keep='last')]
                df = df.sort_index()
                # --- FIX END ---

                # --- FIX: Repainting / Look-Ahead ---
                # Binance returns the currently OPEN candle as the last element.
                # We must ensure we only use CLOSED candles.
                # Check if the last candle's close time is in the future.
                if not df.empty:
                    last_close_time = df.index[-1] + pd.Timedelta(cfg.execution.interval)
                    # Actually, df.index is close_time (from line 389)? 
                    # Line 389: df.index = pd.to_datetime(df["close_time"], unit="ms", utc=True)
                    # So index IS the close time.
                    if df.index[-1] > pd.Timestamp.now(tz="UTC"):
                        log.debug("Dropping incomplete candle at %s (now=%s)", df.index[-1], pd.Timestamp.now(tz="UTC"))
                        df = df.iloc[:-1]
                
                if df.empty:
                    log.warning("No closed candles available after filtering.")
                    time.sleep(5)
                    continue
                # ------------------------------------

                price = float(df["close"].iloc[-1])
            except Exception as e:
                log.error("Failed to fetch klines: %s", e)
                time.sleep(5)
                continue

            # --- BALANCE FETCHING (Updated for Spot/Futures) ---
            # Initialize current_position for use in snap-to-zero logic later
            current_position = 0.0
            W = 0.0 # Initialize W to prevent UnboundLocalError if fetch fails
            cur_w = 0.0 # Initialize cur_w as well
            
            try:
                if is_futures:
                    # 1. FUTURES MODE
                    # In USDS-M Futures, our "Wallet" is the Margin Balance (USDT)
                    quote_bal = adapter.get_account_balance(quote_asset)
                    
                    # --- NEW: Risk Metrics (Margin & Liquidation) ---
                    try:
                        # 1. Margin Utilization
                        # adapter.client is UMFutures
                        acc_info = adapter.client.account()
                        total_maint = float(acc_info.get('totalMaintMargin', 0))
                        total_bal = float(acc_info.get('totalMarginBalance', 1e-9))
                        margin_util_pct = (total_maint / total_bal) * 100 if total_bal > 0 else 0.0

                        # 2. Liquidation Distance
                        pos_risks = adapter.client.get_position_risk(symbol=args.symbol)
                        # API returns list. Filter for current symbol.
                        risk_entry = next((r for r in pos_risks if r['symbol'] == args.symbol), None)
                        liq_dist_pct = 0.0
                        if risk_entry:
                             liq_px = float(risk_entry.get('liquidationPrice', 0))
                             mark_px = float(risk_entry.get('markPrice', 0))
                             if liq_px > 0 and mark_px > 0:
                                 liq_dist_pct = abs(mark_px - liq_px) / mark_px * 100
                        
                        mark_futures_risk(instance_name, margin_util_pct, liq_dist_pct, args.symbol)
                    except Exception as e:
                        # Log debug to avoid spamming main log
                        log.debug("Risk metric fetch failed: %s", e)
                    base_bal = 0.0 # We don't hold the base asset in futures, we hold contracts
                    
                    # Current Exposure comes from the open Position size
                    # FIX #5: Add fallback if position fetch fails
                    try:
                        current_position = adapter.get_position(args.symbol)
                        # Store successful fetch for future fallback
                        state["last_known_position"] = current_position
                    except Exception as e:
                        log.error("Position fetch failed, using last known: %s", e)
                        current_position = state.get("last_known_position", 0.0)
                    
                    # Value of position = Size * Price
                    # (current_position can be negative if Short)
                    position_val = current_position * price
                    
                    # Total Wealth = Margin Balance (includes unrealized PnL)
                    W = quote_bal 
                    
                    # Current Weight = Notional Exposure / Total Wealth
                    # If W is very small, guard against div/0
                    cur_w = position_val / W if W > 1e-6 else 0.0
                    
                    effective_base = current_position # Used for logging/logic later
                    
                    # Calculate signal weight (unleveraged) for comparison with backtest
                    signal_weight = cur_w / lev if lev > 0 else cur_w
                    
                    # Export exposure metrics
                    EXPOSURE_NOTIONAL.labels(instance=instance_name).set(cur_w * 100)  # Leveraged exposure as %
                    EXPOSURE_SIGNAL_WEIGHT.labels(instance=instance_name).set(signal_weight * 100)  # Signal weight as %
                    
                    # Log balance occasionally
                    if state.get("last_balance_log_ts", 0) < now_s - 300:
                        log.info("[FUTURES BALANCE] Margin=%.2f %s, Position=%.4f %s | Exposure: %.1f%% notional (signal_w=%.1f%% @ %dx lev)", 
                                 quote_bal, quote_asset, current_position, args.symbol, 
                                 cur_w*100, signal_weight*100, lev)
                        state["last_balance_log_ts"] = now_s
                                       
                else:
                    # 2. SPOT MODE (Legacy Logic)
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
                    
                    # Spot Calculation
                    min_base_val = global_filters.min_notional / max(price, 1e-12)        

                    effective_base = base_bal
                    if base_bal > 0 and base_bal < min_base_val:
                        effective_base = 0.0
                        # ... dust logging ...

                    W = quote_bal + base_bal * price
                    cur_w = 0.0 if W <= 0 else (effective_base * price) / W
                    log.debug(f"[SPOT] Wealth calculation: quote={quote_bal:.8f}, base={base_bal:.8f}, W={W:.8f}, cur_w={cur_w:.4f}")   

            except Exception as e:
                log.error("CRITICAL: Failed to fetch %s account balance. Reason: %s", args.mode.upper(), e)
                # Fallback logic...
                quote_bal, base_bal = cfg.risk.basis_btc, 0.0
                if "last_known_quote" in state:
                    quote_bal = state["last_known_quote"]
                    # ...
            state["last_known_quote"] = quote_bal
            state["last_known_base"]  = base_bal

            # Bug Fix #3: Removed duplicate balance calculation (already done above in lines 336-390)            

            # Log startup once per process execution (not just once per state file)
            if not startup_logged_this_run and W >= 0:
                story.log_startup(bar_dt, W, args.mode, quote_asset)
                startup_logged_this_run = True
            
            if W <= 0 and state.get("last_balance_log_ts", 0) < now_s - 300:
                log.warning(f"⚠️ Zero Balance detected ({W} {quote_asset}). Bot will not trade.")
                story.log_custom(bar_dt, "⚠️", "ZERO BALANCE", f"Wallet is empty: {W} {quote_asset}")
                alerter.send(f"⚠️ ZERO BALANCE [{args.symbol}]\nWallet is empty: {W} {quote_asset}. Bot will not trade.", level="WARNING")

            if state.get("session_start_W", 0.0) == 0.0 and W > 0:
                state["session_start_W"] = W

            _ensure_risk_state(state, W, bar_dt)
            _update_risk_state(state, W, bar_dt, cfg)

            risk_mode_str = getattr(cfg.risk, "risk_mode", "fixed_basis")
            mark_risk_mode(instance_name, risk_mode_str)

            daily_limit_hit = bool(state.get("risk_daily_limit_hit", False))
            maxdd_hit = bool(state.get("risk_maxdd_hit", False))
            mark_risk_flags(instance_name, daily_limit_hit=daily_limit_hit, maxdd_hit=maxdd_hit)
            PHOENIX_ACTIVE.labels(instance=instance_name).set(1.0 if maxdd_hit else 0.0)  # Update Phoenix status


            # --- ALERT: Risk Trigger (Place this here!) ---
            if maxdd_hit and not state.get("alert_sent_maxdd", False):
                # Calculate DD % manually since it's not a local variable
                eq_high = float(state.get("risk_equity_high", W))
                dd_pct = (eq_high - W) / eq_high if eq_high > 0 else 0.0
                log.warning("🚨 MAX DD HIT: %.2f%%. Halting all trading.", dd_pct * 100)
                # alerter.send(...) -> Moved to StoryWriter.log_safety_breaker
                story.log_safety_breaker(bar_dt, dd_pct)
                state["alert_sent_maxdd"] = True
            
            # Reset alert flag if we recover (optional but good practice)
            if not maxdd_hit:
                state["alert_sent_maxdd"] = False

            # NOTE: WEALTH_TOTAL, PRICE_MID, BAL_FREE are set via snapshot_wealth_balances() 
            # called later in the loop (line ~927) - no need to duplicate here

            q_usd, b_usd = 0.0, 0.0
            try:
                if quote_asset == "USDT":
                    mark_asset_price_usd(instance_name, "usdt", 1.0)
                    q_usd = 1.0
                else:
                    q_usd = adapter.get_usd_price(f"{quote_asset}USDT")
                    mark_asset_price_usd(instance_name, quote_asset, q_usd)

                b_usd = adapter.get_usd_price(f"{base_asset}USDT")
                mark_asset_price_usd(instance_name, base_asset, b_usd)
            except Exception:
                pass
            
            if q_usd > 0:
                WEALTH_USD.labels(instance=instance_name).set(W * q_usd)
            
            log.debug("Wallet: %s=%.8f, %s=%.8f, W=%.8f, cur_w=%.4f", 
                      quote_asset, quote_bal, base_asset, base_bal, W, cur_w)

            # --- INDICATORS (Common) ---
            # Fix: Use correct params for bands if Meta Strategy is active
            strat_type = getattr(cfg.strategy, "strategy_type", "mean_reversion")
            
            # Use merged params for signal display (parity with backtest)
            p_entry = float(mr_params.get("flip_band_entry", 0.025))
            p_exit = float(mr_params.get("flip_band_exit", 0.015))
            p_k = float(mr_params.get("vol_adapt_k", 0.0))
            L = int(mr_params.get("trend_lookback", 200))
            trend_kind = mr_params.get("trend_kind", "roc")
            vol_window = int(mr_params.get("vol_window", 60))

            ser_close = df["close"].astype(float)
            if trend_kind == "sma":
                sma = ser_close.rolling(L).mean()
                cur_ratio = float(ser_close.iloc[-1] / max(sma.iloc[-1], 1e-12) - 1.0)
            else:
                prev = ser_close.shift(L).iloc[-1]
                cur_ratio = float(ser_close.iloc[-1] / max(prev, 1e-12) - 1.0)
            rv = float(ser_close.pct_change().rolling(vol_window).std().iloc[-1])
            
            entry = p_entry + p_k * (rv if rv == rv else 0.0)
            exitb = p_exit + p_k * (rv if rv == rv else 0.0)

            # Export Dynamic Bands to Prometheus
            SIGNAL_BAND.labels(instance=instance_name, kind="upper_entry").set(entry)
            SIGNAL_BAND.labels(instance=instance_name, kind="lower_entry").set(-entry)
            SIGNAL_BAND.labels(instance=instance_name, kind="upper_exit").set(exitb)
            SIGNAL_BAND.labels(instance=instance_name, kind="lower_exit").set(-exitb)

            # --- GATE & FUNDING CHECKS ---
            gate_ok = True
            gate_reason = "open"
            funding_rate = 0.0
            
            # Extract gate and funding params from merged config (parity with backtest)
            gate_window_days = int(mr_params.get("gate_window_days", 0))
            gate_roc_threshold = float(mr_params.get("gate_roc_threshold", 0.0))
            funding_limit_long = float(mr_params.get("funding_limit_long", 0.05))
            funding_limit_short = float(mr_params.get("funding_limit_short", -0.05))
            
            # 1. Trend Gate (Legacy)
            if gate_window_days and gate_roc_threshold:
                day_close = ser_close.resample("1D").last().shift(1)
                if len(day_close) > gate_window_days:
                    droc = float(day_close.iloc[-1] / max(day_close.shift(gate_window_days).iloc[-1], 1e-12) - 1.0)
                    if abs(droc) < gate_roc_threshold:
                        gate_ok = False
                        gate_reason = "trend_weak"

            # 2. Funding Gate
            funding_ticker = f"{base_asset}USDT"
            try:
                funding_rate = adapter.get_funding_rate(funding_ticker)  
                mark_funding_rate(instance_name, funding_rate)
                
                if funding_rate > funding_limit_long:
                    # Don't block here, just mark the reason. Strategy logic will handle direction.
                    # But for safety, we flag it.
                    if cur_ratio <= -entry: # Only care if we might buy
                        gate_ok = False
                        gate_reason = f"funding_high ({funding_rate:.4f}%)"
                        log.warning("Gate CLOSE: Market Euphoria! Funding=%.4f%%", funding_rate)

                if funding_rate < funding_limit_short:
                    if cur_ratio >= entry: # Only care if we might sell
                        gate_ok = False
                        gate_reason = f"funding_low ({funding_rate:.4f}%)"
                        log.warning("Gate CLOSE: Market Panic! Funding=%.4f%%", funding_rate)
            except Exception as e:
                log.warning("Funding check warning: %s", e)


            # --- STRATEGY EXECUTION ---
            target_w = None
            strat_type = getattr(cfg.strategy, "strategy_type", "mean_reversion")
            
            # Bug Fix #5: Initialize these outside strategy blocks for Phoenix Protocol
            mr_merged = {}
            tr_merged = {}

            try:
                if strat_type == "trend":
                    # TREND Strategy - Use tr_params for parity with backtest
                    tp = TrendParams(
                        fast_period=int(tr_params.get("fast_period", 50)),
                        slow_period=int(tr_params.get("slow_period", 200)),
                        ma_type=tr_params.get("ma_type", "ema"),
                        cooldown_minutes=int(tr_params.get("cooldown_minutes", 180)),
                        step_allocation=float(tr_params.get("step_allocation", 1.0)),
                        max_position=float(tr_params.get("max_position", 1.0)),
                        long_only=bool(tr_params.get("long_only", True)),
                        funding_limit_long=float(tr_params.get("funding_limit_long", 0.05)),
                        funding_limit_short=float(tr_params.get("funding_limit_short", -0.05))
                    )
                    strat = TrendStrategy(tp)
                    plan = strat.generate_positions(df) # Requires OHLC
                    target_w = float(plan["target_w"].iloc[-1])




                    # --- Broadcast Meta-Strategy Brain ---
                    # --- ALERT: Regime Switch (Place this here!) ---
                    if "regime_score" in plan.columns:
                        current_score = float(plan["regime_score"].iloc[-1])
                        adx_thresh = float(mr_params.get("adx_threshold", 25.0))
                        
                        # Determine current regime
                        current_regime = "TREND" if current_score > adx_thresh else "CHOP"
                        
                        # Compare with memory
                        last_regime = state.get("last_regime", current_regime)
                        
                        if current_regime != last_regime:
                            # alerter.send(...) -> Moved to StoryWriter.check_regime_switch
                            state["last_regime"] = current_regime
                        
                        # Update Metrics
                        REGIME_SCORE.labels(instance=instance_name).set(current_score)
                        STRATEGY_MODE.labels(instance=instance_name).set(1.0 if current_regime == "TREND" else 0.0)
                        
                        # Log regime switch to story
                        story.check_regime_switch(bar_dt, current_score, adx_thresh, strat_type="meta")

                elif strat_type == "meta":
                    # META Strategy with Overrides
                    # 1. Base Global Params
                    base_params = cfg.strategy.model_dump()
                    
                    # 2. Extract Overrides
                    mr_opts = base_params.get("mean_reversion_overrides", {})
                    tr_opts = base_params.get("trend_overrides", {})

                    # 3. Construct Mean Reversion Params (Merge Base + Overrides)
                    mr_merged = {**base_params, **mr_opts}
                    mr_p = StratParams(
                        trend_kind=mr_merged.get("trend_kind", "roc"),
                        trend_lookback=int(mr_merged.get("trend_lookback", 200)),
                        flip_band_entry=float(mr_merged.get("flip_band_entry", 0.025)),
                        flip_band_exit=float(mr_merged.get("flip_band_exit", 0.015)),
                        vol_window=int(mr_merged.get("vol_window", 60)),
                        vol_adapt_k=float(mr_merged.get("vol_adapt_k", 0.0)),
                        cooldown_minutes=int(mr_merged.get("cooldown_minutes", 60)), # Individual Cooldown
                        step_allocation=float(mr_merged.get("step_allocation", 0.5)),
                        max_position=float(mr_merged.get("max_position", 1.0)),
                        long_only=bool(int(mr_merged.get("long_only", 1))),  # Explicitly convert: 0->False, 1->True
                        funding_limit_long=float(mr_merged.get("funding_limit_long", 0.05)),
                        funding_limit_short=float(mr_merged.get("funding_limit_short", -0.05))
                    )

                    # 4. Construct Trend Params (Merge Base + Overrides)
                    tr_merged = {**base_params, **tr_opts}
                    tr_p = TrendParams(
                        fast_period=int(tr_merged.get("fast_period", 50)),
                        slow_period=int(tr_merged.get("slow_period", 200)),
                        ma_type=tr_merged.get("ma_type", "ema"),
                        cooldown_minutes=int(tr_merged.get("cooldown_minutes", 180)), # Trend usually needs longer cooldown
                        step_allocation=float(tr_merged.get("step_allocation", 1.0)),
                        max_position=float(tr_merged.get("max_position", 1.0)),
                        long_only=bool(tr_merged.get("long_only", True)),
                        funding_limit_long=float(tr_merged.get("funding_limit_long", 0.05)),
                        funding_limit_short=float(tr_merged.get("funding_limit_short", -0.05))
                    )
                    
                    strat = MetaStrategy(mr_p, tr_p, adx_threshold=float(mr_params.get("adx_threshold", 25.0)))
                    # Pass the dataframe (requires OHLC) logic handled by generate_positions
                    plan = strat.generate_positions(df) 
                    target_w = float(plan["target_w"].iloc[-1])

                else:
                    # DEFAULT: Mean Reversion - Use mr_params for parity with backtest
                    sp = StratParams(
                        trend_kind=mr_params.get("trend_kind", "roc"),
                        trend_lookback=int(mr_params.get("trend_lookback", 200)),
                        flip_band_entry=float(mr_params.get("flip_band_entry", 0.025)),
                        flip_band_exit=float(mr_params.get("flip_band_exit", 0.015)),
                        vol_window=int(mr_params.get("vol_window", 60)),
                        vol_adapt_k=float(mr_params.get("vol_adapt_k", 0.0)),
                        target_vol=float(mr_params.get("target_vol", 0.5)),
                        cooldown_minutes=int(mr_params.get("cooldown_minutes", 60)),
                        step_allocation=float(mr_params.get("step_allocation", 0.33)),
                        max_position=float(mr_params.get("max_position", 1.0)),
                        rebalance_threshold_w=float(mr_params.get("rebalance_threshold_w", 0.0)),
                        min_trade_btc=float(mr_params.get("min_trade_btc", 0.0)),
                        gate_window_days=int(mr_params.get("gate_window_days", 0)),
                        gate_roc_threshold=float(mr_params.get("gate_roc_threshold", 0.0)),
                        long_only=bool(int(mr_params.get("long_only", 1))),
                        funding_limit_long=float(mr_params.get("funding_limit_long", 0.05)),
                        funding_limit_short=float(mr_params.get("funding_limit_short", -0.05))
                    )
                    strat = EthBtcStrategy(sp)
                    # Note: EthBtcStrategy usually takes 'close' series. 
                    # The new one we updated takes (close, funding).
                    # Live execution logic for funding is handled via gate_ok mostly, 
                    # but let's pass None to generate_positions and rely on safety override below.
                    plan = strat.generate_positions(ser_close) 
                    target_w = float(plan["target_w"].iloc[-1])

            except Exception as e:
                log.exception("Strategy calculation failed (%s): %s", strat_type, e)
                target_w = cur_w # Hold position on error

            # --- SMART PHOENIX PROTOCOL (Auto-Recovery) ---
            # Checks if we should wake up from a MaxDD Crash
            if maxdd_hit and 'plan' in locals():
                reset_days = float(getattr(cfg.risk, "drawdown_reset_days", 0.0))
                reset_score = float(getattr(cfg.risk, "drawdown_reset_score", 30.0))
                hit_ts_str = state.get("risk_maxdd_hit_ts")
                
                if reset_days > 0 and hit_ts_str:
                    try:
                        # 1. Check Time
                        crash_time = pd.to_datetime(hit_ts_str)
                        if crash_time.tzinfo is None and bar_dt.tzinfo is not None:
                            crash_time = crash_time.replace(tzinfo=bar_dt.tzinfo)
                        time_passed = bar_dt - crash_time
                        
                        # 2. Check Trend (The "Smart" part)
                        current_score = 0.0
                        if "regime_score" in plan.columns:
                            current_score = float(plan["regime_score"].iloc[-1])
                            adx_thresh = getattr(cfg.strategy, "adx_threshold", 25.0)
                            
                            current_regime = "TREND" if current_score > adx_thresh else "CHOP"
                            
                            # Pick the correct threshold based on regime (parity with backtest)
                            if current_regime == "TREND":
                                active_rebalance_threshold = float(tr_params.get("rebalance_threshold_w", 0.0))
                            else:
                                active_rebalance_threshold = float(mr_params.get("rebalance_threshold_w", 0.0))
                            
                        # 3. Decision
                        if (time_passed.total_seconds() >= (reset_days * 86400)) and (current_score >= reset_score):
                            log.warning(f"🐦 PHOENIX REBIRTH: Cooldown ({reset_days}d) passed AND Trend confirmed (Score {current_score:.1f} >= {reset_score}). Resuming!")
                            
                            # Reset Risk State
                            maxdd_hit = False
                            state["risk_maxdd_hit"] = False
                            state["risk_maxdd_hit_ts"] = None
                            state["risk_equity_high"] = W # Reset High Water Mark to NOW
                            state["alert_sent_maxdd"] = False
                            
                            # Notify User
                            # alerter.send(...) -> Moved to StoryWriter.log_phoenix_activation
                            PHOENIX_ACTIVE.labels(instance=instance_name).set(0.0)  # Phoenix reset complete
                            
                            # Log Phoenix activation to story
                            story.log_phoenix_activation(bar_dt, current_score, reset_days)
                    except Exception as e:
                        log.error(f"Phoenix logic failed: {e}")

            # Check and update ATH
            story.check_ath(bar_dt, W, quote_asset)
            
            # Check for period summaries (weekly/monthly/annual)
            story.check_and_log_weekly(bar_dt, W, quote_asset)
            story.check_and_log_monthly(bar_dt, W, quote_asset)
            story.check_and_log_annual(bar_dt, W, quote_asset)
            
            # --- METRICS UPDATE (General) ---
            # --- METRICS UPDATE (General) ---
            # Calculate Regime Score if missing (for observability)
            current_score = 0.0
            if 'plan' in locals() and "regime_score" in plan.columns:
                current_score = float(plan["regime_score"].iloc[-1])
            else:
                try:
                    # Calculate independently
                    rs_series = get_regime_score(df)
                    current_score = float(rs_series.iloc[-1])
                except Exception:
                    pass

            REGIME_SCORE.labels(instance=instance_name).set(current_score)

            # 2. Regime Threshold (FIX: Always publish config value)
            # Default to 25.0 if not set, so graph line is never 0
            adx_thresh = getattr(cfg.strategy, "adx_threshold", 25.0)
            REGIME_THRESHOLD.labels(instance=instance_name).set(adx_thresh)

            # 3. Strategy Mode
            if strat_type == "trend":
                STRATEGY_MODE.labels(instance=instance_name).set(1.0) # Always Trend
            elif strat_type == "meta":
                STRATEGY_MODE.labels(instance=instance_name).set(1.0 if current_score > adx_thresh else 0.0)
            else:
                STRATEGY_MODE.labels(instance=instance_name).set(0.0) # Always MR

            # 4. Regime State (FIX #7: Export actual state with hysteresis)
            if 'plan' in locals() and "regime_state" in plan.columns:
                regime_state_val = float(plan["regime_state"].iloc[-1])
                REGIME_STATE.labels(instance=instance_name).set(regime_state_val)

            # --- FIX #4: REGIME-AWARE THRESHOLD SELECTION ---
            # Meta Strategy should use different thresholds based on current regime
            # This was previously only updated during Phoenix recovery (line 760)
            if strat_type == "meta" and 'plan' in locals():
                if "regime_score" in plan.columns:
                    current_score = float(plan["regime_score"].iloc[-1])
                    adx_thresh = getattr(cfg.strategy, "adx_threshold", 25.0)
                    
                    if current_score > adx_thresh:
                        # TREND mode - use Trend threshold (parity with backtest)
                        active_rebalance_threshold = float(tr_params.get("rebalance_threshold_w", 0.0))
                        log.debug("[THRESHOLD] Using TREND threshold: %.4f", active_rebalance_threshold)
                    else:
                        # MR mode - use MR threshold (parity with backtest)
                        active_rebalance_threshold = float(mr_params.get("rebalance_threshold_w", 0.0))
                        log.debug("[THRESHOLD] Using MR threshold: %.4f", active_rebalance_threshold)
            # For non-Meta strategies, use config value (already set at line 375)
            # ------------------------------------------------

            # --- SAFETY OVERRIDE ---
            # If the "Global Gate" is closed (due to funding or trend check in main loop),
            # we must prevent BUYING.
            if not gate_ok:
                # If strategy wants to increase exposure, block it.
                if target_w > cur_w and target_w > 0:
                    log.warning("Safety Override: Gate CLOSED. Blocking Long Entry.")
                    target_w = cur_w
                # Prevent OPENING or ADDING to Shorts (if target < cur)
                # (This logic depends on your specific gate definition, but standard gates block Longs)
            # -----------------------

            # If Spot: Clamp [0, 1]
            # If Futures: Clamp [-1, 1] (or [-Leverage, +Leverage])
            if is_futures:
                # Allow Shorts!
                max_lev = float(getattr(cfg.execution, "leverage", 1.0))
                target_w = max(-max_lev, min(max_lev, target_w))
            else:
                # Spot is Long Only
                target_w = max(0.0, min(1.0, target_w))

            reset_trade_decision(instance_name)

            delta_eth = 0.0
            delta_w = 0.0
            side = "HOLD"

            meter = ascii_level_bar(cur_ratio, entry, exitb, width=64)
            dist_to_buy_bps, dist_to_sell_bps = dist_to_buy_sell_bps(cur_ratio, entry, exitb)
            mark_signal_metrics(instance_name, cur_ratio, dist_to_buy_bps, dist_to_sell_bps)
            snapshot_wealth_balances(instance_name, W, price, quote_bal, base_bal, quote_asset, base_asset)
            
            gate_display = "OPEN" if gate_ok else f"CLOSED ({gate_reason})"
            log.info("[SIG] ratio=%+0.4f  bands: -entry=%0.4f  -exit=%0.4f  +exit=%0.4f  +entry=%0.4f  gate=%s  %s",
                    cur_ratio, -entry, -exitb, exitb, entry, gate_display, meter)

            mark_gate(instance_name, gate_ok)

            if cur_ratio <= -entry:
                zone = "buy_band"
            elif cur_ratio >= entry:
                zone = "sell_band"
            else:
                zone = "neutral"
            mark_zone(instance_name, zone)

            action_side = ('BUY' if (target_w > cur_w) else ('SELL' if (target_w < cur_w) else 'HOLD'))
            
            # === DYNAMIC POSITION SIZING ===
            from core.position_sizer import PositionSizer, PositionSizerConfig
            
            # Use appropriate params based on current regime (parity with backtest)
            # Default to MR params; trend mode uses tr_params if in trend regime
            active_sizing_params = mr_params  # Default: Mean Reversion
            is_in_trend_mode = (merged_cfg["strategy_type"] == "meta" and current_score > adx_thresh)
            if is_in_trend_mode:
                active_sizing_params = tr_params
            
            # Create position sizer config using merged params
            sizer_config = PositionSizerConfig(
                mode=active_sizing_params.get("position_sizing_mode", "static"),
                base_step=float(active_sizing_params.get("step_allocation", 0.33)),
                target_vol=float(active_sizing_params.get("position_sizing_target_vol", 0.5)),
                min_step=float(active_sizing_params.get("position_sizing_min_step", 0.1)),
                max_step=float(active_sizing_params.get("position_sizing_max_step", 1.0)),
                kelly_win_rate=float(active_sizing_params.get("kelly_win_rate", 0.55)),
                kelly_avg_win=float(active_sizing_params.get("kelly_avg_win", 0.02)),
                kelly_avg_loss=float(active_sizing_params.get("kelly_avg_loss", 0.01)),
            )
            sizer = PositionSizer(sizer_config)
            
            # Calculate dynamic step based on realized volatility
            step = sizer.calculate_step(realized_vol=rv)
            
            # Export metrics for Grafana
            POSITION_STEP.labels(instance=instance_name).set(step)
            REALIZED_VOL.labels(instance=instance_name).set(rv)

            # --- SNAP-TO-ZERO (Generic) ---
            # Bug Fix #7: Use position size for futures, base_bal for spot
            if target_w == 0.0:
                check_val = current_position if is_futures else base_bal
                # CRITICAL FIX: Use abs() to handle both long AND short positions
                if abs(check_val) > 0:
                    total_val_quote = abs(check_val) * price  # Absolute value for shorts
                    min_trade_quote = max(cfg.execution.min_trade_floor_btc,
                                    cfg.execution.min_trade_btc or (cfg.execution.min_trade_frac * cfg.risk.basis_btc))
                    
                    if total_val_quote > min_trade_quote and total_val_quote < (3.0 * min_trade_quote):
                        step = 1.0  # Override dynamic step for snap-to-zero
                        position_type = "SHORT" if check_val < 0 else "LONG"
                        log.info("[EXEC] Snap-to-Zero: %s %s position small (%.6f %s). Forcing exit.", position_type, base_asset, total_val_quote, quote_asset)

            # Legacy vol_scaled_step (deprecated, use position_sizing_mode instead)
            if getattr(cfg.strategy, "vol_scaled_step", False):
                z = cur_ratio / max(rv, 1e-6)
                step = min(1.0, max(0.1, step * min(2.0, abs(z))))


            # Final target calculation with smoothing
            new_w_ideal = cur_w + step * (target_w - cur_w)
            
            # Bug Fix #8: Apply correct clamp based on mode BEFORE risk checks
            if is_futures:
                max_lev = float(getattr(cfg.execution, "leverage", 1.0))
                new_w = max(-max_lev, min(max_lev, new_w_ideal))
            else:
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
            
            # Delta Base = Weight Change * Total Wealth / Price
            delta_eth = delta_w * W / max(price, 1e-12)
            
            # Bug Fix #4: Removed duplicate side determination (will be set later at line 906)

            set_delta_metrics(instance_name, delta_w, delta_eth)
            side = "BUY" if delta_eth > 0 else "SELL"

            # Spread Probe
            sp_bps = 0.0
            try:
                book = adapter.get_book(args.symbol)
                sp_bps = 1e4 * (book.best_ask - book.best_bid) / max(price, 1e-12)
                SPREAD_BPS.labels(instance=instance_name).set(sp_bps)
            except Exception as e:
                log.warning("Spread probe failed. Reason: %s", e)

            # Check Balance / Margin
            # In Futures, we check "Available Margin" vs "Initial Margin Requirement"
            # For simplicity in this spot-based architecture, we rely on the clamp logic below.
            
            # Bug Fix #6: Guard spot-only balance check
            if not is_futures and side == "SELL" and base_bal <= 1e-12:
                # Spot-only check: Can't sell what you don't have
                SKIPS.labels(instance=instance_name, reason="balance").inc()
                # ... (rest of skip balance logic) ...
                continue
            
            # Bug Fix: BUY-side balance check (prevents "insufficient balance" from Binance)
            # For BUY, we need BTC (quote asset) to buy ETH (base asset)
            if not is_futures and side == "BUY":
                # How much ETH can we afford with our BTC?
                max_affordable_eth = q_free / max(price, 1e-12) * 0.995  # 0.5% buffer for fees/rounding
                if abs(delta_eth) > max_affordable_eth:
                    if max_affordable_eth < step_size:
                        log.warning(
                            "BUY skip: Insufficient BTC balance (%.8f %s) to buy any ETH (need ~%.8f %s for min order)",
                            q_free, quote_asset, step_size * price, quote_asset
                        )
                        SKIPS.labels(instance=instance_name, reason="balance").inc()
                        mark_decision(instance_name, "skip_balance")
                        state["last_target_w"] = target_w
                        state["last_bar_close"] = bar_ts
                        save_state(args.state, state)
                        last_seen_bar = bar_ts
                        if args.once: break
                        time.sleep(cfg.execution.poll_sec)
                        continue
                    else:
                        log.info(
                            "BUY capped: Insufficient BTC for full order. Requested=%.6f %s, Affordable=%.6f %s",
                            abs(delta_eth), base_asset, max_affordable_eth, base_asset
                        )
                        delta_eth = max_affordable_eth  # Cap to what we can afford

            qty_abs = abs(delta_eth)
            
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
            delta_ok = abs_delta_for_ready >= float(mr_params.get("rebalance_threshold_w", 0.0))
            risk_ok = not (daily_limit_hit or maxdd_hit)

            mark_trade_readiness(
                instance_name,
                zone_ok=zone_ok,
                gate_ok=gate_open_ok,
                delta_ok=delta_ok,
                risk_ok=risk_ok,
                balance_ok=True,
                size_ok=True
            )

            tol = 1e-12
            abs_delta = abs(delta_w)

            if abs_delta < tol:
                SKIPS.labels(instance=instance_name, reason="delta_zero").inc()
                mark_decision(instance_name, "skip_delta_zero")
                log.info("Skip: cur_w==target_w==%.4f (no change).", cur_w)
                state["last_target_w"] = target_w
                state["last_bar_close"] = bar_ts
                save_state(args.state, state)
                EXPOSURE_W.labels(instance=instance_name, kind="target").set(target_w)
                EXPOSURE_W.labels(instance=instance_name, kind="current").set(cur_w)
                PNL_QUOTE.labels(instance=instance_name).set(W - float(state.get("session_start_W", W)))
                last_seen_bar = bar_ts
                if args.once:
                    log.info("Run-once complete (no-op).")
                    break
                time.sleep(cfg.execution.poll_sec)
                continue

            if abs_delta < active_rebalance_threshold:
                SKIPS.labels(instance=instance_name, reason='threshold').inc()
                mark_decision(instance_name, "skip_threshold")   
                log.info(
                    "Skip: |Δw|=%.4f < threshold=%.4f (need ≥ %.4f).",
                    abs_delta, active_rebalance_threshold, active_rebalance_threshold
                )
                state["last_target_w"] = target_w
                state["last_bar_close"] = bar_ts
                save_state(args.state, state)
                EXPOSURE_W.labels(instance=instance_name, kind="target").set(target_w)
                EXPOSURE_W.labels(instance=instance_name, kind="current").set(cur_w)
                PNL_QUOTE.labels(instance=instance_name).set(W - float(state.get("session_start_W", W)))
                last_seen_bar = bar_ts
                if args.once:
                    log.info("Run-once complete (skip threshold).")
                    break
                time.sleep(cfg.execution.poll_sec)
                continue

            # Bug Fix #6: Guard spot-only balance check
            if not is_futures and side == "SELL" and base_bal <= 1e-12:
                SKIPS.labels(instance=instance_name, reason="balance").inc()
                mark_decision(instance_name, "skip_balance")
                log.info("Skip: SELL requested but %s balance is 0 (cur_w=%.4f, target_w=%.4f).", base_asset, cur_w, target_w)
                state["last_target_w"] = target_w
                state["last_bar_close"] = bar_ts
                save_state(args.state, state)
                EXPOSURE_W.labels(instance=instance_name, kind="target").set(target_w)
                EXPOSURE_W.labels(instance=instance_name, kind="current").set(cur_w)
                PNL_QUOTE.labels(instance=instance_name).set(W - float(state.get("session_start_W", W)))
                last_seen_bar = bar_ts
                if args.once:
                    log.info("Run-once complete (skip balance).")
                    break
                time.sleep(cfg.execution.poll_sec)
                continue

            qty_abs = abs(delta_eth)
            qty_rounded = adapter.round_qty(qty_abs, global_filters.step_size)

            # --- Anti-Zeno Deadlock (Force Exit) ---
            # If Target is 0 (Exit) AND we have a position, but the calculated 
            # partial step is too small to execute (0 or < min_notional),
            # we force the entire position to close.
            if target_w == 0.0 and is_futures and abs(current_position) > 0:
                # Calculate value of the partial trade
                partial_val = qty_rounded * price
                
                # If partial trade is invalid (0 or too small)
                if qty_rounded == 0.0 or partial_val < global_filters.min_notional:
                    log.info("Force Exit: Partial step too small (%.8f), closing FULL position (%.8f)", qty_rounded, abs(current_position))
                    qty_rounded = adapter.round_qty(abs(current_position), global_filters.step_size)


            min_trade_quote = max(cfg.execution.min_trade_floor_btc,
                                cfg.execution.min_trade_btc or (cfg.execution.min_trade_frac * cfg.risk.basis_btc))
            if cfg.execution.min_trade_cap_btc > 0:
                min_trade_quote = min(min_trade_quote, cfg.execution.min_trade_cap_btc)

            notional_quote = qty_rounded * price

            need = max(min_trade_quote, global_filters.min_notional)
            
            if notional_quote < need:
                SKIPS.labels(instance=instance_name, reason="min_notional").inc()
                mark_decision(instance_name, "skip_min_notional")
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

            # ---- Balance-aware clamp (Mode-specific) ----------------------------
            if is_futures:
                # FUTURES MODE: Margin-based trading
                # For both BUY and SELL, we can use margin up to our available balance
                # The exchange will handle position management
                max_qty_by_balance = adapter.round_qty(max((quote_bal * 0.999) / max(price, 1e-12), 0.0), global_filters.step_size)
                log.debug(f"[FUTURES] Max qty by balance: {max_qty_by_balance:.8f} (margin={quote_bal:.2f})")
            else:
                # SPOT MODE: Wallet-based trading
                if side == "BUY":
                    max_qty_by_balance = adapter.round_qty(max((quote_bal * 0.999) / max(price, 1e-12), 0.0), global_filters.step_size)
                else:  
                    max_qty_by_balance = adapter.round_qty(max(base_bal, 0.0), global_filters.step_size)
                log.debug(f"[SPOT] Max qty by balance: {max_qty_by_balance:.8f} (side={side}, quote={quote_bal:.8f}, base={base_bal:.8f})")

            qty_exec = min(qty_rounded, max_qty_by_balance)

            if qty_exec <= 0:
                SKIPS.labels(instance=instance_name, reason="balance").inc()
                mark_decision(instance_name, "skip_balance")
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
                SKIPS.labels(instance=instance_name, reason="min_notional").inc()
                mark_decision(instance_name, 'skip_min_notional')
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
                ORDERS_SUBMITTED.labels(instance=instance_name, kind="MARKET", side=side).inc()
                mark_decision(instance_name, "exec_buy" if side == "BUY" else "exec_sell")
                log.info(
                    "[DRY] Would EXEC %s %0.8f @ ~%0.8f (Δw=%+.4f, Δ%s=%+.6f)",
                    side, qty_exec, price, delta_w, base_asset, delta_eth
                )
                # Log trade to story (even in dry mode)
                story.log_trade(bar_dt, side, delta_eth, price, base_asset, quote_asset)
                executed_qty = qty_exec # Assume fill
            else:
                # --- MAKER EXECUTION ---
                if use_maker:
                    log.info("Attempting MAKER execution for %s %s...", side, qty_exec)
                    try:
                        filled_maker = maker_chase(
                            adapter, args.symbol, side, qty_exec, global_filters.tick_size,
                            max_reprices=3, step_sec=8, stop_event=STOP_EVENT
                        )
                        executed_qty += filled_maker
                        if filled_maker > 0:
                            FILLS.labels(instance=instance_name).inc()
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
                        ORDERS_SUBMITTED.labels(instance=instance_name, kind="MARKET", side=side).inc()
                        mark_decision(instance_name, "exec_buy" if side == "BUY" else "exec_sell")
                        log.info("EXEC TAKER %s %0.8f %s @ ~%0.8f (oid=%s)", side, remaining, args.symbol, price, oid)
                        
                        # Wait briefly and check if market order filled (especially important on testnet)
                        time.sleep(1.0)
                        is_filled, filled_qty = adapter.check_order(args.symbol, oid)
                        if is_filled:
                            log.info("TAKER order %s FILLED: %.8f", oid, filled_qty)
                            executed_qty += filled_qty
                            FILLS.labels(instance=instance_name).inc()
                        else:
                            log.warning("TAKER order %s NOT FILLED yet (status check after 1s). Filled: %.8f", oid, filled_qty)
                            executed_qty += filled_qty  # Add partial fills
                        
                        if side == "BUY":
                            TRADE_DECISION.labels(instance=instance_name, trade_decision="exec_buy").set(1)
                        else:
                            TRADE_DECISION.labels(instance=instance_name, trade_decision="exec_sell").set(1)
                        
                        # Log trade to story
                        story.log_trade(bar_dt, side, delta_eth, price, base_asset, quote_asset)

                        # --- NEW: Execution Stats (Slippage) ---
                        try:
                            # Fetch fill details for precise slippage
                            if is_futures: 
                                 ord_status = adapter.client.query_order(symbol=args.symbol, orderId=oid)
                            else:
                                 ord_status = adapter.client.get_order(symbol=args.symbol, orderId=oid)
                            
                            fill_px = float(ord_status.get('avgPrice', price))
                            if fill_px == 0: fill_px = price # Fallback
                            
                            slip_bps = (abs(fill_px - price) / price) * 10000
                            mark_execution_stats(instance_name, slip_bps, 0.0, "")
                        except Exception as e:
                            log.warning("ExStats failed: %s", e)
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
                            mark_decision(instance_name, "skip_order_error")
                            if args.once: break
                            time.sleep(cfg.execution.poll_sec)
                            continue

                elif remaining > 0:
                    log.info("Remainder %.8f too small for Taker (Notional %.8f < %.8f). Stopping.", remaining, rem_notional, need)

            # Update state/metrics after trade logic
            state["last_target_w"] = target_w
            state["last_bar_close"] = bar_ts
            save_state(args.state, state)

            # CRITICAL: Re-fetch actual position if we had unfilled orders
            # This ensures Grafana shows REAL Binance data, not bot's assumptions
            actual_current_w = new_w  # Default to calculated
            if is_futures and executed_qty < qty_exec:
                # Some or all of the order didn't fill - get truth from exchange
                try:
                    actual_position = adapter.get_position(args.symbol)
                    # FIX #5: Store successful position fetch
                    state["last_known_position"] = actual_position
                    state["last_known_cur_w"] = (actual_position * price) / max(W, 1e-12)
                    actual_current_w = state["last_known_cur_w"]
                    if abs(actual_current_w - new_w) > 0.01:
                        log.warning("Position mismatch! Bot calculated: %.4f, Actual: %.4f (unfilled orders)", new_w, actual_current_w)
                except Exception as e:
                    log.error("Failed to fetch actual position: %s", e)

            EXPOSURE_W.labels(instance=instance_name, kind="target").set(target_w)
            EXPOSURE_W.labels(instance=instance_name, kind="current").set(actual_current_w)  # Use ACTUAL position
            PNL_QUOTE.labels(instance=instance_name).set(W - float(state.get("session_start_W", W)))
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