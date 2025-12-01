# core/metrics.py
from prometheus_client import Counter, Gauge, Summary, start_http_server, REGISTRY

# --- Reload-safe registry setup for tests ------------------------------------
try:
    REGISTRY._names_to_collectors.clear()
    REGISTRY._collector_to_names.clear()
except Exception:
    pass

# ---- Core Metrics ----------------------------------------------------------

ORDERS_SUBMITTED = Counter("orders_submitted_total", "Orders submitted", ["kind", "side"])
FILLS = Counter("fills_total", "Executed fills")
REJECTIONS = Counter("rejections_total", "Order rejections", ["reason"])

# Renamed for clarity (was pnl_btc) - now generic PnL in Quote units
PNL_QUOTE = Gauge("pnl_quote", "PnL in Quote Asset since session start")

EXPOSURE_W = Gauge("exposure_base_weight", "Base Asset target/curr weights", ["kind"])
SPREAD_BPS = Gauge("spread_bps", "Current spread in basis points")
BAR_LATENCY = Summary("bar_latency_seconds", "Latency of processing a closed bar")

# ---- State/Decision Metrics ------------------------------------------------

GATE_STATE = Gauge("gate_state", "Trading gate state", ["gate_state"])
SIGNAL_ZONE = Gauge("signal_zone", "Signal zone", ["signal_zone"])
TRADE_DECISION = Gauge("trade_decision", "Decision this bar", ["trade_decision"])

DELTA_W = Gauge("delta_w", "Planned Δweight")
DELTA_BASE = Gauge("delta_base_asset", "Planned ΔBase units")

# GENERIC WEALTH (Was WEALTH_BTC_TOTAL)
WEALTH_TOTAL = Gauge("wealth_total", "Total wealth in Quote Asset units")
PRICE_MID = Gauge("price_mid", "Mid price of the traded pair")

# BALANCES (Asset label will be dynamic)
BAL_FREE = Gauge("balance_free", "Free balance by asset", ["asset"])

SKIPS = Counter("skips_total", "Skips by reason", ["reason"])

# ---- GENERIC USD PRICES (New) ---------------------------------------------
# Replaces PRICE_BTC_USD / PRICE_ETH_USD
PRICE_ASSET_USD = Gauge("price_asset_usd", "Approx USD price of asset", ["asset"])

# ---- Signal & Risk --------------------------------------------------------

SIGNAL_RATIO = Gauge("signal_ratio", "Current ratio signal")
DIST_TO_BUY_BPS = Gauge("dist_to_buy_bps", "Distance to BUY entry in bps")
DIST_TO_SELL_BPS = Gauge("dist_to_sell_bps", "Distance to SELL entry in bps")
NEXT_ACTION_DIST_BPS = Gauge("next_action_dist_bps", "Distance to nearest band entry")

RISK_MODE = Gauge("risk_mode", "Risk mode active flag", ["mode"])
RISK_FLAGS = Gauge("risk_flags", "Risk state flags", ["kind"])
FUNDING_RATE = Gauge("funding_rate_pct", "Current funding rate percentage")

# ---- Trade Readiness Aggregate --------------------------------------------
TRADE_READY = Gauge("trade_ready", "Overall trade readiness (1=OK, 0=Blocked)")
TRADE_READY_COND = Gauge("trade_ready_condition", "Sub-condition readiness", ["cond"])

REGIME_SCORE = Gauge("regime_score", "Current Trend Consensus Score (0-100)")
# ---- Helper functions -----------------------------------------------------

def start_metrics_server(port: int) -> None:
    start_http_server(port)

def mark_signal_metrics(ratio: float, dist_buy: float, dist_sell: float) -> None:
    SIGNAL_RATIO.set(ratio)
    DIST_TO_BUY_BPS.set(dist_buy)
    DIST_TO_SELL_BPS.set(dist_sell)
    try:
        NEXT_ACTION_DIST_BPS.set(min(dist_buy, dist_sell))
    except Exception:
        pass

def mark_gate(open_: bool) -> None:
    GATE_STATE.labels("open").set(1.0 if open_ else 0.0)
    GATE_STATE.labels("closed").set(0.0 if open_ else 1.0)

def mark_zone(zone: str) -> None:
    for z in ("buy_band", "sell_band", "neutral"):
        SIGNAL_ZONE.labels(z).set(1.0 if z == zone else 0.0)

def mark_decision(kind: str) -> None:
    all_kinds = [
        "exec_buy", "exec_sell", "skip_threshold", "skip_balance", 
        "skip_min_notional", "skip_cooldown", "skip_gate_closed", 
        "skip_delta_zero", "skip_order_error"
    ]
    for k in all_kinds:
        TRADE_DECISION.labels(k).set(1.0 if k == kind else 0.0)

def mark_risk_mode(active_mode: str) -> None:
    for mode in ("fixed_basis", "dynamic"):
        RISK_MODE.labels(mode=mode).set(1.0 if mode == active_mode else 0.0)

def mark_risk_flags(*, daily_limit_hit: bool, maxdd_hit: bool) -> None:
    RISK_FLAGS.labels(kind="daily_limit_hit").set(1.0 if daily_limit_hit else 0.0)
    RISK_FLAGS.labels(kind="maxdd_hit").set(1.0 if maxdd_hit else 0.0)

def set_delta_metrics(delta_w: float, delta_base: float) -> None:
    DELTA_W.set(delta_w)
    DELTA_BASE.set(delta_base)

def snapshot_wealth_balances(
    wealth_total: float,
    price_mid: float,
    quote_val: float,
    base_val: float,
    quote_asset: str,
    base_asset: str
) -> None:
    """
    Generic snapshot of wealth + balances + mid price.
    """
    WEALTH_TOTAL.set(wealth_total)
    PRICE_MID.set(price_mid)
    BAL_FREE.labels(quote_asset.lower()).set(quote_val)
    BAL_FREE.labels(base_asset.lower()).set(base_val)

def mark_trade_readiness(*, zone_ok: bool, gate_ok: bool, delta_ok: bool, risk_ok: bool, balance_ok: bool, size_ok: bool) -> None:
    ready = all([zone_ok, gate_ok, delta_ok, risk_ok, balance_ok, size_ok])
    TRADE_READY.set(1.0 if ready else 0.0)
    TRADE_READY_COND.labels(cond="zone_ok").set(1.0 if zone_ok else 0.0)
    TRADE_READY_COND.labels(cond="gate_open").set(1.0 if gate_ok else 0.0)
    TRADE_READY_COND.labels(cond="delta_ok").set(1.0 if delta_ok else 0.0)
    TRADE_READY_COND.labels(cond="risk_ok").set(1.0 if risk_ok else 0.0)
    TRADE_READY_COND.labels(cond="balance_ok").set(1.0 if balance_ok else 0.0)
    TRADE_READY_COND.labels(cond="size_ok").set(1.0 if size_ok else 0.0)

def mark_funding_rate(rate: float) -> None:
    FUNDING_RATE.set(rate)

def mark_asset_price_usd(asset: str, price: float) -> None:
    """Record USD price for a specific asset label"""
    PRICE_ASSET_USD.labels(asset=asset.lower()).set(price)