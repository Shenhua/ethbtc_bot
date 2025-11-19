#live_executor.py
from __future__ import annotations
import os, json, time, logging, argparse, math, threading
from typing import Dict, Any
import pandas as pd
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Lock
from binance.spot import Spot
from core.config_schema import load_config
from core.binance_adapter import BinanceSpotAdapter

from core.metrics import (
    ORDERS_SUBMITTED, FILLS, REJECTIONS, PNL_BTC, EXPOSURE_W, SPREAD_BPS, BAR_LATENCY, GATE_STATE, SIGNAL_ZONE, TRADE_DECISION, DELTA_W, DELTA_ETH, WEALTH_BTC_TOTAL, PRICE_MID, BAL_FREE, SKIPS, PRICE_BTC_USD, PRICE_ETH_USD, SIGNAL_RATIO, DIST_TO_BUY_BPS, DIST_TO_SELL_BPS, start_metrics_server, mark_gate, mark_zone, mark_decision, mark_signal_metrics, snapshot_wealth_balances, set_delta_metrics,mark_risk_mode, mark_risk_flags
)
from ascii_levelbar import dist_to_buy_sell_bps, ascii_level_bar
# --- Simple JSON /status on :9110 ------------------------------------------------

DECISION_KEYS = (
    "exec_buy", "exec_sell", "skip_threshold", "skip_balance", "skip_min_notional", "skip_cooldown", "skip_gate_closed", "skip_delta_zero","skip_order_error"
)

def reset_trade_decision():
    # zero all decision gauges once per bar so panels don’t get stale
    for k in DECISION_KEYS:
        try:
            TRADE_DECISION.labels(k).set(0)
        except Exception:
            pass


# ---- Human-friendly JSON status server ---------------------------------------

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




# --------------------------------------------------------------------------------
# --- Strategy parity import (used INSIDE main only) ---------------------------
try:
    from ethbtc_accum_bot import EthBtcStrategy, StratParams  # source of truth
except Exception:
    EthBtcStrategy = None
    StratParams = None
def inc_rejection(reason: str = "error") -> None:
    try:
        # metrics with a label called 'reason'
        REJECTIONS.labels(reason=reason).inc()
    except Exception:
        try:
            # metrics with single positional label
            REJECTIONS.labels(reason).inc()
        except Exception:
            try:
                # metrics without labels
                REJECTIONS.inc()
            except Exception:
                pass


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
    from pathlib import Path
    import json, os, errno

    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(p) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(st, f, indent=2, sort_keys=True)
        os.replace(tmp, str(p))
    except OSError as e:
        # read-only FS (e.g., /data on macOS outside Docker)
        if e.errno in (errno.EROFS,):
            fallback = Path.cwd() / "run_state" / p.name
            fallback.parent.mkdir(parents=True, exist_ok=True)
            tmp = str(fallback) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(st, f, indent=2, sort_keys=True)
            os.replace(tmp, str(fallback))
        else:
            raise
def _ensure_risk_state(state: Dict[str, Any], wealth_btc: float, ts: pd.Timestamp) -> None:
    """
    Initialize risk-related fields in the persisted state dict, if missing.
    """
    if "risk_equity_high_btc" not in state:
        state["risk_equity_high_btc"] = wealth_btc
    if "risk_current_date" not in state:
        state["risk_current_date"] = ts.normalize().date().isoformat()
    if "risk_daily_start_wealth" not in state:
        state["risk_daily_start_wealth"] = wealth_btc
    if "risk_daily_limit_hit" not in state:
        state["risk_daily_limit_hit"] = False
    if "risk_maxdd_hit" not in state:
        state["risk_maxdd_hit"] = False


def _update_risk_state(state: Dict[str, Any], wealth_btc: float, ts: pd.Timestamp, cfg) -> None:
    """
    Update risk tracking (max DD & daily loss) in-place on the state dict.

    Behaviour:
      - In 'dynamic' mode, fractional caps (max_dd_frac, max_daily_loss_frac)
        are interpreted as fractions of peak / daily-start equity.
      - In 'fixed_basis' mode, we use the BTC caps directly.
    """
    risk = cfg.risk
    risk_mode = getattr(risk, "risk_mode", "fixed_basis")
    max_dd_frac = float(getattr(risk, "max_dd_frac", 0.0) or 0.0)
    max_daily_loss_frac = float(getattr(risk, "max_daily_loss_frac", 0.0) or 0.0)

    equity_high = float(state.get("risk_equity_high_btc", wealth_btc))
    current_date_str = state.get("risk_current_date")
    if current_date_str:
        try:
            current_date = pd.to_datetime(current_date_str).date()
        except Exception:
            current_date = ts.normalize().date()
    else:
        current_date = ts.normalize().date()

    daily_start = float(state.get("risk_daily_start_wealth", wealth_btc))
    daily_limit_hit = bool(state.get("risk_daily_limit_hit", False))
    maxdd_hit = bool(state.get("risk_maxdd_hit", False))

    # --- Max drawdown tracking ------------------------------------------------
    if wealth_btc > equity_high:
        equity_high = wealth_btc
    dd_now = equity_high - wealth_btc

    if risk_mode == "dynamic":
        if max_dd_frac > 0.0:
            threshold_dd = equity_high * max_dd_frac
        else:
            threshold_dd = float(risk.max_dd_btc)
    else:
        threshold_dd = float(risk.max_dd_btc)

    if threshold_dd > 0.0 and dd_now >= threshold_dd:
        maxdd_hit = True

    # --- Daily loss tracking --------------------------------------------------
    cur_date = ts.normalize().date()
    if cur_date != current_date:
        # New calendar day → reset daily counters
        current_date = cur_date
        daily_start = wealth_btc
        daily_limit_hit = False

    daily_pnl = wealth_btc - daily_start

    if risk_mode == "dynamic":
        if max_daily_loss_frac > 0.0:
            threshold_loss = daily_start * max_daily_loss_frac
        else:
            threshold_loss = float(risk.max_daily_loss_btc)
    else:
        threshold_loss = float(risk.max_daily_loss_btc)

    if threshold_loss > 0.0 and daily_pnl <= -threshold_loss:
        daily_limit_hit = True

    # Persist updated state
    state["risk_equity_high_btc"] = equity_high
    state["risk_current_date"] = current_date.isoformat()
    state["risk_daily_start_wealth"] = daily_start
    state["risk_daily_limit_hit"] = daily_limit_hit
    state["risk_maxdd_hit"] = maxdd_hit



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--mode", choices=["dry","testnet","live"], default="dry")
    ap.add_argument("--symbol", default="ETHBTC")
    from pathlib import Path
    # Choose default based on env or whether we're inside a container
    _default_state = os.getenv("STATE_FILE") or (
        "/data/state.json" if Path("/.dockerenv").exists()
        else str(Path.cwd() / "run_state" / "state.json")
    )
    ap.add_argument("--state", default=_default_state, help="state file path")
    args = ap.parse_args()

    cfg = load_config(args.params)

    # client
    #
    # If BINANCE_BASE_URL is not set:
    # - testnet mode → use testnet endpoint
    # - dry/live     → use mainnet public endpoint
    if args.mode == "testnet":
        default_base = "https://testnet.binance.vision"
    else:
        # dry + live both read public data from mainnet unless overridden
        default_base = "https://api.binance.com"

    base_url = os.getenv("BINANCE_BASE_URL", default_base)

    client = Spot(
        api_key=os.getenv("BINANCE_KEY", ""),
        api_secret=os.getenv("BINANCE_SECRET", ""),
        base_url=base_url,
    )

    # adapter & metrics
    adapter = BinanceSpotAdapter(client)
    metrics_port = int(os.getenv("METRICS_PORT", "9109"))
    status_port  = int(os.getenv("STATUS_PORT", "9110"))
    # Start Prometheus metrics HTTP server
    start_metrics_server(metrics_port)
    # Start human-readable /status server, and get its update_status function
    update_status = start_status_server(status_port)

    # state
    state = load_state(args.state)
    if "session_start_W" not in state:
        state["session_start_W"] = 0.0
    last_seen_bar = 0


    while True:
        t0 = time.time()
        now_s = int(t0)
        bar_ts = last_closed_bar_ts(now_s, cfg.execution.interval)
        if last_seen_bar == bar_ts:
            time.sleep(cfg.execution.poll_sec); continue

        # klines & price
        ks = adapter.get_klines(args.symbol, cfg.execution.interval, limit=600)
        df = pd.DataFrame(ks)
        if "close_time" in df.columns:
            df.index = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        price = float(df["close"].iloc[-1])

        # balances → weights
        try:
            acct = client.account()
            bal_map = {x["asset"]: float(x["free"]) + float(x["locked"]) for x in acct["balances"]}
            btc = bal_map.get("BTC", 0.0)
            eth = bal_map.get("ETH", 0.0)
        except Exception:
            btc, eth = cfg.risk.basis_btc, 0.0

        W = btc + eth * price
        cur_w = 0.0 if W <= 0 else (eth * price) / W

        if state.get("session_start_W", 0.0) == 0.0 and W > 0:
            state["session_start_W"] = W

        # --- Risk tracking (max DD / daily loss) ------------------------------
        bar_dt = pd.to_datetime(bar_ts, unit="s", utc=True)
        _ensure_risk_state(state, W, bar_dt)
        _update_risk_state(state, W, bar_dt, cfg)
        # --- Risk metrics → Prometheus ---------------------------------------
        risk_mode_str = getattr(cfg.risk, "risk_mode", "fixed_basis")
        daily_limit_hit = bool(state.get("risk_daily_limit_hit", False))
        maxdd_hit = bool(state.get("risk_maxdd_hit", False))

        # 1-of-N mode: exactly one of these will be 1.0
        mark_risk_mode(risk_mode_str)

        # Flags: 0/1 gauges
        mark_risk_flags(
            daily_limit_hit=daily_limit_hit,
            maxdd_hit=maxdd_hit,
        )
        # Snapshot → metrics
        WEALTH_BTC_TOTAL.set(W)
        PRICE_MID.set(price)
        BAL_FREE.labels('btc').set(btc)
        BAL_FREE.labels('eth').set(eth)

        # Snapshot → USD prices (BTC/ETH)
        try:
            btc_usd = adapter.get_usd_price("BTCUSDT")
            eth_usd = adapter.get_usd_price("ETHUSDT")
            PRICE_BTC_USD.set(btc_usd)
            PRICE_ETH_USD.set(eth_usd)
        except Exception as e:
            log.debug("USD price fetch failed: %s", e)
        log.debug("Wallet: BTC=%.8f, ETH=%.8f, W=%.8f BTC, cur_w=%.4f", btc, eth, W, cur_w)

        # indicators for diagnostics (not for decision if strategy is present)
        L = int(cfg.strategy.trend_lookback)
        ser_close = df["close"].astype(float)
        if cfg.strategy.trend_kind == "sma":
            sma = ser_close.rolling(L).mean()
            cur_ratio = float(ser_close.iloc[-1] / max(sma.iloc[-1], 1e-12) - 1.0)
        else:  # 'roc'
            prev = ser_close.shift(L).iloc[-1]
            cur_ratio = float(ser_close.iloc[-1] / max(prev, 1e-12) - 1.0)
        rv = float(ser_close.pct_change().rolling(cfg.strategy.vol_window).std().iloc[-1])
        entry = cfg.strategy.flip_band_entry + cfg.strategy.vol_adapt_k * (rv if rv == rv else 0.0)
        exitb = cfg.strategy.flip_band_exit + cfg.strategy.vol_adapt_k * (rv if rv == rv else 0.0)

        # gate diagnostic (calendar daily ROC)
        gate_ok = True
        if cfg.strategy.gate_window_days and cfg.strategy.gate_roc_threshold:
            day_close = ser_close.resample("1D").last()
            if len(day_close) > cfg.strategy.gate_window_days:
                droc = float(day_close.iloc[-1] / max(day_close.shift(cfg.strategy.gate_window_days).iloc[-1], 1e-12) - 1.0)
                gate_ok = abs(droc) >= cfg.strategy.gate_roc_threshold

        # --- STRATEGY PARITY: compute target_w from EthBtcStrategy if available ---
        target_w = None
        
        if EthBtcStrategy and StratParams:
            try:
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
                )
                strat = EthBtcStrategy(sp)
                plan = strat.generate_positions(ser_close)
                target_w = float(plan["target_w"].iloc[-1])
                # Spot-only: cannot short ETH. Clamp to [0, 1].
                raw_target_w = target_w
                target_w = max(0.0, min(1.0, target_w))
                if target_w != raw_target_w:
                    log.info("Clamp target_w from %.4f to %.4f for spot (no shorting).", raw_target_w, target_w)
            except Exception as e:
                log.warning("Parity strategy import failed (%s); using inline logic.", e)
                target_w = cur_w

        # Fallback inline logic
        if target_w is None:
            target_w = 0.0
            if gate_ok:
                if cur_ratio > entry:
                    target_w = min(cfg.strategy.max_position, 1.0)
                elif cur_ratio < -exitb:
                    target_w = 0.0
                else:
                    target_w = float(state.get("last_target_w", 0.0))


        # --- Per-bar initialization ------------------------------------------------
        reset_trade_decision()  # zero all decision gauges this bar

        # Snapshot → metrics (wealth, price, balances)
        try:
            WEALTH_BTC_TOTAL.set(W)
            PRICE_MID.set(price)
            BAL_FREE.labels("btc").set(float(btc))
            BAL_FREE.labels("eth").set(float(eth))
        except Exception:
            pass

        delta_eth = 0.0
        delta_w = 0.0
        side = "HOLD"

        # --- ASCII diagnostic & distances --------------------------------------------
        meter = ascii_level_bar(cur_ratio, entry, exitb, width=64)
        bar_reason = None

        # Distances in bps to next triggers
        dist_to_buy_bps, dist_to_sell_bps = dist_to_buy_sell_bps(cur_ratio, entry, exitb)
        mark_signal_metrics(cur_ratio, dist_to_buy_bps, dist_to_sell_bps)

        snapshot_wealth_balances(W, price, btc, eth)

        log.info("[SIG] ratio=%+0.4f  bands: -entry=%0.4f  -exit=%0.4f  +exit=%0.4f  +entry=%0.4f  gate=%s  %s",
                cur_ratio, -entry, -exitb, exitb, entry, "OPEN" if gate_ok else "CLOSED", meter)

        # ---- Metrics: gate / zone / signal / distances
        mark_gate(gate_ok)

        # Decide zone
        if cur_ratio <= -entry:
            zone = "buy_band"
        elif cur_ratio >= entry:
            zone = "sell_band"
        else:
            zone = "neutral"
        mark_zone(zone)

      
        # quick preview (same step, without vol-scaling nuance if you prefer)
        action_side = ('BUY' if (target_w > cur_w) else ('SELL' if (target_w < cur_w) else 'HOLD'))

        # step toward target
        step = cfg.strategy.step_allocation
        if getattr(cfg.strategy, "vol_scaled_step", False):
            z = cur_ratio / max(rv, 1e-6)
            step = min(1.0, max(0.1, step * min(2.0, abs(z))))

        # Spot-only: clamp the *target* as well (keeps metrics & state consistent)
        target_w = max(0.0, min(1.0, target_w))
        # target_w = 1.0  # TEMP uncomment : force full ETH allocation for test only

        new_w_ideal = cur_w + step * (target_w - cur_w)
        new_w = max(0.0, min(1.0, new_w_ideal))  # spot guard

        # --- Apply risk caps (max DD / daily loss) ----------------------------
        risk_mode = getattr(cfg.risk, "risk_mode", "fixed_basis")
        maxdd_hit = bool(state.get("risk_maxdd_hit", False))
        daily_limit_hit = bool(state.get("risk_daily_limit_hit", False))

        if maxdd_hit:
            # Hard stop: force BTC-only from here on
            if new_w > 0.0:
                log.warning("Risk: max drawdown hit → forcing BTC-only (new_w=0.0).")
            target_w = 0.0
            new_w = 0.0
        elif daily_limit_hit:
            # Freeze position for the rest of the day (no further rebalancing)
            if abs(new_w - cur_w) > 1e-12:
                log.warning("Risk: daily loss limit hit → freezing position at w=%.4f.", cur_w)
            new_w = cur_w

        # --- Actual deltas after risk caps -----------------------------------
        delta_w = new_w - cur_w
        delta_eth = delta_w * W / max(price, 1e-12)

        # --- Step-aware planned deltas → metrics (record even if we skip) -----
        set_delta_metrics(delta_w, delta_eth)

        WEALTH_BTC_TOTAL.set(W)
        PRICE_MID.set(price)
        BAL_FREE.labels("btc").set(btc)
        BAL_FREE.labels("eth").set(eth)
        # ---- Snapshot → metrics (wealth, mid price, balances)
        snapshot_wealth_balances(W, price, btc, eth)

        # ---- Spread probe (optional but used by Grafana panels)
        try:
            book = adapter.get_book(args.symbol)  # must return {"best_bid","best_ask"}
            sp_bps = 1e4 * (book["best_ask"] - book["best_bid"]) / max(price, 1e-12)
            SPREAD_BPS.set(sp_bps)
        except Exception:
            pass
        action_side = ('BUY' if (target_w > cur_w) else ('SELL' if (target_w < cur_w) else 'HOLD'))
        zone = ('BUY' if dist_to_buy_bps == 0 else 'SELL' if dist_to_sell_bps == 0 else 'NEUTRAL')
        

        update_status({
            "mode": args.mode,
            "symbol": args.symbol,
            "price": price,
            "wealth_btc": W,
            "cur_w": cur_w,
            "target_w": target_w,
            "step": step,
            "delta_w_planned": delta_w,
            "delta_eth_planned": delta_eth,
            "side": action_side,
            "ratio": cur_ratio,
            "entry": entry,
            "bands": {
                "neg_entry": -entry,
                "neg_exit":  -exitb,
                "pos_exit":   exitb,
                "pos_entry":  entry,
            },
            "btc_usd": float(btc_usd) if 'btc_usd' in locals() else None,
            "eth_usd": float(eth_usd) if 'eth_usd' in locals() else None,
            "dist_to_buy_bps": round(dist_to_buy_bps, 1),
            "dist_to_sell_bps": round(dist_to_sell_bps, 1),
            "zone": ('BUY' if dist_to_buy_bps == 0 else 'SELL' if dist_to_sell_bps == 0 else 'NEUTRAL'),
            "gate": ("OPEN" if gate_ok else "CLOSED"),
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
            "plan: Δw=%+.4f (ΔETH=%+.6f) → %s | dist: BUY=%0.1fbps SELL=%0.1fbps",
            args.mode, price, W,
            zone, ("OPEN" if gate_ok else "CLOSED"),
            cur_w, target_w, step,
            delta_w, delta_eth,
            action_side,
            dist_to_buy_bps, dist_to_sell_bps
        )


        # --- Skip tiny rebalances / no-op --------------------------------------------
        tol = 1e-12
        abs_delta = abs(delta_w)

        if abs_delta < tol:
            # No-op: target equals current (within numerical tolerance)
            SKIPS.labels("delta_zero").inc()
            bar_reason = "skip_delta_zero"
            mark_decision("skip_delta_zero")
            update_status({
                "mode": args.mode,
                "symbol": args.symbol,
                "price": price,
                "wealth_btc": W,
                "cur_w": cur_w,
                "target_w": target_w,
                "step": step,
                # for a true no-op, treat actual trade delta as zero
                "delta_w_planned": 0.0,
                "delta_eth_planned": 0.0,
                "side": "HOLD",
                "ratio": cur_ratio,
                "entry": entry,
                "bands": {
                    "neg_entry": -entry,
                    "neg_exit":  -exitb,
                    "pos_exit":   exitb,
                    "pos_entry":  entry,
                },
                "btc_usd": float(btc_usd) if 'btc_usd' in locals() else None,
                "eth_usd": float(eth_usd) if 'eth_usd' in locals() else None,
                "dist_to_buy_bps": round(dist_to_buy_bps, 1),
                "dist_to_sell_bps": round(dist_to_sell_bps, 1),
                "zone": ('BUY' if dist_to_buy_bps == 0 else
                         'SELL' if dist_to_sell_bps == 0 else
                         'NEUTRAL'),
                "gate": ("OPEN" if gate_ok else "CLOSED"),
                "ascii_meter": meter,
                "reason": "skip_delta_zero",
            })
            log.info("Skip: cur_w==target_w==%.4f (no change).", cur_w)
            state["last_target_w"] = target_w
            state["last_bar_close"] = bar_ts
            save_state(args.state, state)
            EXPOSURE_W.labels("target").set(target_w)
            EXPOSURE_W.labels("current").set(cur_w)
            PNL_BTC.set(W - float(state.get("session_start_W", W)))
            last_seen_bar = bar_ts
            time.sleep(cfg.execution.poll_sec)
            continue

        if abs_delta < cfg.strategy.rebalance_threshold_w:
            SKIPS.labels('threshold').inc()
            bar_reason = "skip_threshold"
            mark_decision("skip_threshold")   
            try:
                update_status({
                    "mode":   args.mode,
                    "symbol": args.symbol,
                    "price":  price,
                    "wealth_btc":      W,
                    "cur_w":  cur_w,
                    "target_w": target_w,
                    "step":   step,
                    "delta_w": 0.0,
                    "delta_eth": 0.0,
                    "zone":  "BUY" if cur_ratio <= -entry else (
                            "SELL" if cur_ratio >=  entry else "NEUTRAL"),
                    "gate":  "OPEN" if gate_ok else "CLOSED",
                    "reason": "below threshold",
                })
            except Exception:
                pass

            log.info(
                "Skip: |Δw|=%.4f < threshold=%.4f (need ≥ %.4f).",
                abs_delta, cfg.strategy.rebalance_threshold_w, cfg.strategy.rebalance_threshold_w
            )
            state["last_target_w"] = target_w
            state["last_bar_close"] = bar_ts
            save_state(args.state, state)
            with BAR_LATENCY.time():
                pass
            EXPOSURE_W.labels("target").set(target_w)
            EXPOSURE_W.labels("current").set(cur_w)
            PNL_BTC.set(W - float(state.get("session_start_W", W)))
            last_seen_bar = bar_ts
            time.sleep(cfg.execution.poll_sec)
            
            continue

        # Order planning
        try:
            DELTA_ETH.set(delta_eth)
        except Exception:
            pass

        side = "BUY" if delta_eth > 0 else "SELL"

        if side == "SELL" and eth <= 1e-12:
            SKIPS.labels("balance").inc()
            bar_reason = "skip_balance"
            mark_decision("skip_balance")
            log.info("Skip: SELL requested but ETH balance is 0 on spot (cur_w=%.4f, target_w=%.4f).", cur_w, target_w)
            try:
                update_status({
                    "mode":   args.mode,
                    "symbol": args.symbol,
                    "price":  price,
                    "wealth_btc":      W,
                    "cur_w":  cur_w,
                    "target_w": target_w,
                    "step":   step,
                    "delta_w": float(new_w - cur_w),
                    "delta_eth": float(delta_eth),
                    "zone":  "BUY" if cur_ratio <= -entry else (
                            "SELL" if cur_ratio >=  entry else "NEUTRAL"),
                    "gate":  "OPEN" if gate_ok else "CLOSED",
                    "reason": "no ETH to SELL",
                })
            except Exception:
                pass
            
            state["last_target_w"] = target_w
            state["last_bar_close"] = bar_ts
            save_state(args.state, state)
            # keep metrics in sync
            EXPOSURE_W.labels("target").set(target_w)
            EXPOSURE_W.labels("current").set(cur_w)
            PNL_BTC.set(W - float(state.get("session_start_W", W)))
            last_seen_bar = bar_ts
            time.sleep(cfg.execution.poll_sec)
            try:
                update_status(
                    evt="bar", symbol=args.symbol, bar_close=pd.Timestamp.utcfromtimestamp(bar_ts).isoformat()+"Z",
                    wealth_btc=W, target_w=target_w, new_w=new_w, cur_w=cur_w
                )
            except Exception:
                pass
            continue


        qty_abs = abs(delta_eth)

        f = adapter.get_filters(args.symbol)
        qty_rounded = adapter.round_qty(qty_abs, f.step_size)

        min_trade_btc = max(cfg.execution.min_trade_floor_btc,
                            cfg.execution.min_trade_btc or (cfg.execution.min_trade_frac * cfg.risk.basis_btc))
        if cfg.execution.min_trade_cap_btc > 0:
            min_trade_btc = min(min_trade_btc, cfg.execution.min_trade_cap_btc)

        notional_btc = qty_rounded * price
        need = max(min_trade_btc, f.min_notional)
        if notional_btc < need:
            SKIPS.labels("min_notional").inc()
            bar_reason = "skip_min_notional"
            mark_decision("skip_min_notional")
            log.info("Skip: notional below minimum (%.8f BTC < min %.8f)", notional_btc, need)
            state["last_target_w"] = target_w
            state["last_bar_close"] = bar_ts
            save_state(args.state, state)
            last_seen_bar = bar_ts            

            try:
                update_status(
                    evt="bar", symbol=args.symbol, bar_close=pd.Timestamp.utcfromtimestamp(bar_ts).isoformat()+"Z",
                    wealth_btc=W, target_w=target_w, new_w=new_w, cur_w=cur_w
                )
            except Exception:
                pass
            time.sleep(cfg.execution.poll_sec); continue

       
        # ---- Balance-aware clamp before placing the order ----------------------------
        if side == "BUY":
            max_qty_by_balance = adapter.round_qty(max(btc / max(price, 1e-12), 0.0), f.step_size)
        else:  
            max_qty_by_balance = adapter.round_qty(max(eth, 0.0), f.step_size)

        qty_exec = min(qty_rounded, max_qty_by_balance)

            # After successful submit/fill
        try:
            if side == "BUY":
                TRADE_DECISION.labels("exec_buy").set(1)
            else:
                TRADE_DECISION.labels("exec_sell").set(1)
        except Exception:
            pass

        if qty_exec <= 0:
            SKIPS.labels("balance").inc()
            bar_reason = "skip_balance"
            mark_decision("skip_balance")
            log.info(
                "Skip: insufficient free balance for %s (qty_rounded=%.8f, max_by_balance=%.8f, btc=%.8f, eth=%.8f)",
                side, qty_rounded, max_qty_by_balance, btc, eth
            )
            inc_rejection("insufficient_balance")
            state["last_target_w"] = target_w
            state["last_bar_close"] = bar_ts
            save_state(args.state, state)
            last_seen_bar = bar_ts
            time.sleep(cfg.execution.poll_sec)            
            continue

        # Recompute notional and min checks with the clamped size
        notional_btc = qty_exec * price
        need = max(cfg.execution.min_trade_btc or (cfg.execution.min_trade_frac * cfg.risk.basis_btc),
                f.min_notional, cfg.execution.min_trade_floor_btc)
        if cfg.execution.min_trade_cap_btc > 0:
            need = min(need, cfg.execution.min_trade_cap_btc)

        if notional_btc < need:
            SKIPS.labels("min_notional").inc()
            bar_reason = "skip_min_notional"
            mark_decision('skip_min_notional')

            log.info(
                "Skip: notional below minimum after clamp (%.8f BTC < min %.8f) [side=%s qty=%.8f]",
                notional_btc, need, side, qty_exec
            )
            state["last_target_w"] = target_w
            state["last_bar_close"] = bar_ts
            save_state(args.state, state)
            last_seen_bar = bar_ts
            time.sleep(cfg.execution.poll_sec)            
            continue

        # ---- Place the order (or simulate in dry mode) -------------------------------
        if args.mode == "dry":
            # Simulation: do not send any real order, just log + metrics
            ORDERS_SUBMITTED.labels("MARKET", side).inc()
            mark_decision("exec_buy" if side == "BUY" else "exec_sell")
            log.info(
                "[DRY] Would EXEC %s %0.8f @ ~%0.8f (Δw=%+.4f, ΔETH=%+.6f)",
                side, qty_exec, price, delta_w, delta_eth
            )
            try:
                update_status(
                    evt="bar", symbol=args.symbol,
                    bar_close=pd.Timestamp.utcfromtimestamp(bar_ts).isoformat()+"Z",
                    wealth_btc=W, target_w=target_w, new_w=new_w, cur_w=cur_w,
                    sizing={'delta_eth': delta_eth, 'side': side},
                )
            except Exception:
                pass
        else:
                    # ---- Place the order (or simulate it in dry mode) ----------------------------
            if args.mode == "dry":
                # Simulation only: log a fake trade, don't hit the Binance order API
                ORDERS_SUBMITTED.labels("MARKET", side).inc()
                mark_decision("exec_buy" if side == "BUY" else "exec_sell")

                log.info(
                    "[DRY TRADE] %s %.8f %s @ ~%.8f (Δw=%+.4f, ΔETH=%+.6f, W=%.6f)",
                    side,
                    qty_exec,
                    args.symbol,
                    price,
                    delta_w,
                    delta_eth,
                    W,
                )

                try:
                    update_status(
                        evt="bar",
                        symbol=args.symbol,
                        bar_close=pd.Timestamp.utcfromtimestamp(bar_ts).isoformat() + "Z",
                        wealth_btc=W,
                        target_w=target_w,
                        new_w=new_w,
                        cur_w=cur_w,
                        sizing={"delta_eth": float(delta_eth), "side": side},
                        dry_run=True,
                    )
                except Exception:
                    pass

            else:
                # Live / testnet: actually send the order
                try:
                    oid = adapter.market_order(args.symbol, side, qty_exec)
                    ORDERS_SUBMITTED.labels("MARKET", side).inc()
                    mark_decision("exec_buy" if side == "BUY" else "exec_sell")
                    log.info("EXEC %s %0.8f %s @ ~%0.8f (oid=%s)", side, qty_exec, args.symbol, price, oid)

                    try:
                        update_status(
                            evt="bar",
                            symbol=args.symbol,
                            bar_close=pd.Timestamp.utcfromtimestamp(bar_ts).isoformat() + "Z",
                            wealth_btc=W,
                            target_w=target_w,
                            new_w=new_w,
                            cur_w=cur_w,
                            sizing={"delta_eth": float(delta_eth), "side": side},
                            dry_run=False,
                        )
                    except Exception:
                        pass

                except Exception as e:
                    # Label the rejection reason when possible
                    msg = str(e)
                    reason = "insufficient_balance" if "-2010" in msg else "order_error"
                    inc_rejection(reason)
                    log.exception("Order rejected: %s", e)
                    # Continue loop without crashing
                    state["last_target_w"] = target_w
                    state["last_bar_close"] = bar_ts
                    save_state(args.state, state)
                    last_seen_bar = bar_ts
                    time.sleep(cfg.execution.poll_sec)
                    mark_decision(
                        "skip_balance" if reason == "insufficient_balance" else "skip_order_error"
                    )
                    continue

        # Update state/metrics
        state["last_target_w"] = target_w
        state["last_bar_close"] = bar_ts
        save_state(args.state, state)

        EXPOSURE_W.labels("target").set(target_w)
        EXPOSURE_W.labels("current").set(new_w)
        PNL_BTC.set(W - float(state.get("session_start_W", W)))
        last_seen_bar = bar_ts

        # pacing
        dt = time.time() - t0
        sleep_left = max(0.0, cfg.execution.poll_sec - dt)
        time.sleep(sleep_left)

if __name__ == "__main__":
    main()
