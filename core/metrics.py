# core/metrics.py
from prometheus_client import Counter, Gauge, Summary, start_http_server, REGISTRY
import threading
from http.server import HTTPServer
import os

# --- Reload-safe registry setup for tests ------------------------------------
try:
    REGISTRY._names_to_collectors.clear()
    REGISTRY._collector_to_names.clear()
except Exception:
    pass

# ---- Core Metrics (with instance labels) -----------------------------------

# Bot Health & Status
UP_METRIC = Gauge("up", "Bot instance health status", ["instance"])

# Trading Activity
ORDERS_SUBMITTED = Counter("orders_submitted_total", "Orders submitted", ["instance", "kind", "side"])
FILLS = Counter("fills_total", "Executed fills", ["instance"])
REJECTIONS = Counter("rejections_total", "Order rejections", ["instance", "reason"])

# Performance
PNL_QUOTE = Gauge("pnl_quote", "PnL in Quote Asset since session start", ["instance"])
EXPOSURE_W = Gauge("exposure_base_weight", "Base Asset target/curr weights", ["instance", "kind"])
SPREAD_BPS = Gauge("spread_bps", "Current spread in basis points", ["instance"])
BAR_LATENCY = Summary("bar_latency_seconds", "Latency of processing a closed bar", ["instance"])

# ---- State/Decision Metrics ------------------------------------------------

GATE_STATE = Gauge("gate_state", "Trading gate state", ["instance", "gate_state"])
SIGNAL_ZONE = Gauge("signal_zone", "Signal zone", ["instance", "signal_zone"])
TRADE_DECISION = Gauge("trade_decision", "Decision this bar", ["instance", "trade_decision"])

DELTA_W = Gauge("delta_w", "Planned Δweight", ["instance"])
DELTA_BASE = Gauge("delta_base_asset", "Planned ΔBase units", ["instance"])

WEALTH_TOTAL = Gauge("wealth_total", "Total wealth in Quote Asset units", ["instance"])
WEALTH_USD = Gauge("wealth_usd", "Total wealth in USD", ["instance"])
PRICE_MID = Gauge("price_mid", "Mid price of the traded pair", ["instance"])
BAL_FREE = Gauge("balance_free", "Free balance by asset", ["instance", "asset"])

SKIPS = Counter("skips_total", "Skips by reason", ["instance", "reason"])
PRICE_ASSET_USD = Gauge("price_asset_usd", "Approx USD price of asset", ["instance", "asset"])

# ---- Signal & Risk --------------------------------------------------------

SIGNAL_RATIO = Gauge("signal_ratio", "Current ratio signal", ["instance"])
DIST_TO_BUY_BPS = Gauge("dist_to_buy_bps", "Distance to BUY entry in bps", ["instance"])
DIST_TO_SELL_BPS = Gauge("dist_to_sell_bps", "Distance to SELL entry in bps", ["instance"])
NEXT_ACTION_DIST_BPS = Gauge("next_action_dist_bps", "Distance to nearest band entry", ["instance"])
SIGNAL_BAND = Gauge("signal_band", "Dynamic Signal Thresholds", ["instance", "kind"])

RISK_MODE = Gauge("risk_mode", "Risk mode active flag", ["instance", "mode"])
RISK_FLAGS = Gauge("risk_flags", "Risk state flags", ["instance", "kind"])
FUNDING_RATE = Gauge("funding_rate_pct", "Current funding rate percentage", ["instance"])

# ---- Trade Readiness Aggregate --------------------------------------------
TRADE_READY = Gauge("trade_ready", "Overall trade readiness (1=OK, 0=Blocked)", ["instance"])
TRADE_READY_COND = Gauge("trade_ready_condition", "Sub-condition readiness", ["instance", "cond"])

# --- Strategy & Regime Metrics ---
REGIME_SCORE = Gauge("regime_score", "Current Trend Consensus Score (0-100)", ["instance"])
REGIME_THRESHOLD = Gauge("regime_threshold", "ADX Threshold for Regime Switch", ["instance"])
STRATEGY_MODE = Gauge("strategy_mode", "Active Strategy: 0=MeanRev, 1=Trend", ["instance"])
REGIME_STATE = Gauge("regime_state", "Regime State: -1=MR, 1=Trend (with hysteresis)", ["instance"])
PHOENIX_ACTIVE = Gauge("phoenix_active", "Phoenix Protocol Status: 1=Waiting for Reset, 0=Normal Trading", ["instance"])
POSITION_STEP = Gauge("position_step_size", "Dynamic position step size (0.0-1.0)", ["instance"])
REALIZED_VOL = Gauge("realized_volatility", "Current realized volatility (annualized)", ["instance"])

# --- LEVERAGE & EXPOSURE METRICS ---
LEVERAGE = Gauge("leverage", "Current leverage multiplier", ["instance"])
EXPOSURE_SIGNAL_WEIGHT = Gauge("exposure_signal_weight", "Signal weight (unleveraged target)", ["instance"])
EXPOSURE_NOTIONAL = Gauge("exposure_notional", "Actual leveraged notional exposure as % of margin", ["instance"])
CONFIG_LONG_ONLY = Gauge("config_long_only", "1 if Long Only, 0 if Shorts Allowed", ["instance"])
POSITION_STALE = Gauge("position_stale", "1 if position data is stale, 0 otherwise", ["instance"])

# --- RISK & EXECUTION METRICS ---
MARGIN_UTIL = Gauge("margin_utilization_pct", "Used Margin percentage (Futures Only)", ["instance"])
LIQ_DIST = Gauge("liquidation_distance_pct", "Distance to Liquidation Price %", ["instance", "symbol"])
SLIPPAGE = Summary("execution_slippage_bps", "Trade slippage in BPS (Fill vs Decision Price)", ["instance"])
FEES_PAID = Counter("fees_paid_total", "Cumulative Fees Paid", ["instance", "asset"])
LAST_TRADE_TS = Gauge("last_trade_timestamp_seconds", "Timestamp of last trade execution", ["instance"])

# ---- Helper functions -----------------------------------------------------

def start_metrics_server(port: int, story_file: str = None) -> None:
    """
    Start Prometheus metrics HTTP server with optional /story endpoint.
    """
    if story_file:
        # Use custom handler that serves both /metrics and /story
        import core.story_server as ss
        ss.StoryMetricsHandler.story_file_path = story_file
        
        # FIX: Manually start HTTPServer to support custom handler
        httpd = HTTPServer(("", port), ss.StoryMetricsHandler)
        t = threading.Thread(target=httpd.serve_forever)
        t.daemon = True
        t.start()
    else:
        # Standard Prometheus server (metrics only)
        start_http_server(port)

def mark_bot_up(instance: str, is_up: bool = True) -> None:
    """Mark bot instance as up (1) or down (0)."""
    UP_METRIC.labels(instance=instance).set(1.0 if is_up else 0.0)

def mark_signal_metrics(instance: str, ratio: float, dist_buy: float, dist_sell: float) -> None:
    SIGNAL_RATIO.labels(instance=instance).set(ratio)
    DIST_TO_BUY_BPS.labels(instance=instance).set(dist_buy)
    DIST_TO_SELL_BPS.labels(instance=instance).set(dist_sell)
    try:
        NEXT_ACTION_DIST_BPS.labels(instance=instance).set(min(dist_buy, dist_sell))
    except Exception:
        pass

def mark_gate(instance: str, open_: bool) -> None:
    GATE_STATE.labels(instance=instance, gate_state="open").set(1.0 if open_ else 0.0)
    GATE_STATE.labels(instance=instance, gate_state="closed").set(0.0 if open_ else 1.0)

def mark_zone(instance: str, zone: str) -> None:
    for z in ("buy_band", "sell_band", "neutral"):
        SIGNAL_ZONE.labels(instance=instance, signal_zone=z).set(1.0 if z == zone else 0.0)

def mark_decision(instance: str, kind: str) -> None:
    all_kinds = [
        "exec_buy", "exec_sell", "skip_threshold", "skip_balance", 
        "skip_min_notional", "skip_cooldown", "skip_gate_closed", 
        "skip_delta_zero", "skip_order_error"
    ]
    for k in all_kinds:
        TRADE_DECISION.labels(instance=instance, trade_decision=k).set(1.0 if k == kind else 0.0)

def mark_risk_mode(instance: str, active_mode: str) -> None:
    for mode in ("fixed_basis", "dynamic"):
        RISK_MODE.labels(instance=instance, mode=mode).set(1.0 if mode == active_mode else 0.0)

def mark_risk_flags(instance: str, *, daily_limit_hit: bool, maxdd_hit: bool) -> None:
    RISK_FLAGS.labels(instance=instance, kind="daily_limit_hit").set(1.0 if daily_limit_hit else 0.0)
    RISK_FLAGS.labels(instance=instance, kind="maxdd_hit").set(1.0 if maxdd_hit else 0.0)

def set_delta_metrics(instance: str, delta_w: float, delta_base: float) -> None:
    DELTA_W.labels(instance=instance).set(delta_w)
    DELTA_BASE.labels(instance=instance).set(delta_base)

def snapshot_wealth_balances(
    instance: str,
    wealth_total: float,
    price_mid: float,
    quote_val: float,
    base_val: float,
    quote_asset: str,
    base_asset: str
) -> None:
    WEALTH_TOTAL.labels(instance=instance).set(wealth_total)
    PRICE_MID.labels(instance=instance).set(price_mid)
    BAL_FREE.labels(instance=instance, asset=quote_asset.lower()).set(quote_val)
    BAL_FREE.labels(instance=instance, asset=base_asset.lower()).set(base_val)

def mark_trade_readiness(instance: str, *, zone_ok: bool, gate_ok: bool, delta_ok: bool, risk_ok: bool, balance_ok: bool, size_ok: bool) -> None:
    ready = all([zone_ok, gate_ok, delta_ok, risk_ok, balance_ok, size_ok])
    TRADE_READY.labels(instance=instance).set(1.0 if ready else 0.0)
    TRADE_READY_COND.labels(instance=instance, cond="zone_ok").set(1.0 if zone_ok else 0.0)
    TRADE_READY_COND.labels(instance=instance, cond="gate_open").set(1.0 if gate_ok else 0.0)
    TRADE_READY_COND.labels(instance=instance, cond="delta_ok").set(1.0 if delta_ok else 0.0)
    TRADE_READY_COND.labels(instance=instance, cond="risk_ok").set(1.0 if risk_ok else 0.0)
    TRADE_READY_COND.labels(instance=instance, cond="balance_ok").set(1.0 if balance_ok else 0.0)
    TRADE_READY_COND.labels(instance=instance, cond="size_ok").set(1.0 if size_ok else 0.0)

def mark_funding_rate(instance: str, rate: float) -> None:
    FUNDING_RATE.labels(instance=instance).set(rate)

def mark_asset_price_usd(instance: str, asset: str, price: float) -> None:
    PRICE_ASSET_USD.labels(instance=instance, asset=asset.lower()).set(price)

def mark_execution_stats(instance: str, slippage_bps: float, fee_amt: float, fee_asset: str) -> None:
    SLIPPAGE.labels(instance=instance).observe(slippage_bps)
    if fee_asset:
        FEES_PAID.labels(instance=instance, asset=fee_asset.lower()).inc(fee_amt)
    LAST_TRADE_TS.labels(instance=instance).set_to_current_time()

def mark_futures_risk(instance: str, margin_util: float, liq_dist_pct: float, symbol: str) -> None:
    MARGIN_UTIL.labels(instance=instance).set(margin_util)
    LIQ_DIST.labels(instance=instance, symbol=symbol).set(liq_dist_pct)
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
