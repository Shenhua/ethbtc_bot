# core/metrics.py
from prometheus_client import Counter, Gauge, Summary, start_http_server

# ---- Core existing metrics (from original bot) -----------------------------

ORDERS_SUBMITTED = Counter(
    "orders_submitted_total", "Orders submitted", ["kind", "side"]
)
FILLS = Counter(
    "fills_total", "Executed fills"
)
REJECTIONS = Counter(
    "rejections_total", "Order rejections", ["reason"]
)

PNL_BTC = Gauge(
    "pnl_btc", "PnL in BTC since session start"
)
EXPOSURE_W = Gauge(
    "exposure_eth_weight", "ETH target/curr weights", ["kind"]
)
SPREAD_BPS = Gauge(
    "spread_bps", "Current spread in basis points"
)
BAR_LATENCY = Summary(
    "bar_latency_seconds", "Latency of processing a closed bar"
)

# ---- New state/decision metrics for the dashboard -------------------------

# Gate: OPEN / CLOSED
GATE_STATE = Gauge(
    "gate_state", "Trading gate state", ["gate_state"]
)

# Zone: buy_band / neutral / sell_band
SIGNAL_ZONE = Gauge(
    "signal_zone", "Signal zone", ["signal_zone"]
)

# Per-bar decision:
#  - exec_buy / exec_sell
#  - skip_threshold / skip_balance / skip_min_notional / skip_cooldown / skip_gate_closed / skip_delta_zero
TRADE_DECISION = Gauge(
    "trade_decision", "Decision this bar", ["trade_decision"]
)

# Planned changes this bar
DELTA_W = Gauge(
    "delta_w", "Planned Δweight for this bar"
)
DELTA_ETH = Gauge(
    "delta_eth", "Planned ΔETH units for this bar"
)

# Wealth & mid price & balances
WEALTH_BTC_TOTAL = Gauge(
    "wealth_btc_total", "Total wealth in BTC"
)
PRICE_MID = Gauge(
    "price_mid", "Mid price used for sizing"
)
BAL_FREE = Gauge(
    "balance_free", "Free balance by asset", ["asset"]
)

# ---- Skips -----------------------------------------------------------------

# Counter of skips by reason
SKIPS = Counter(
    "skips_total", "Skips by reason", ["reason"]
)

# ---- Optional USD price metrics (for USD panels in Grafana) ---------------

PRICE_BTC_USD = Gauge(
    "price_btc_usd", "BTC price in USD"
)
PRICE_ETH_USD = Gauge(
    "price_eth_usd", "ETH price in USD"
)

# ---- Signal & band distance metrics ---------------------------------------

SIGNAL_RATIO = Gauge(
    "signal_ratio", "Current ratio signal"
)

DIST_TO_BUY_BPS = Gauge(
    "dist_to_buy_bps",
    "Distance to BUY entry in bps (0 if inside BUY zone)",
)

DIST_TO_SELL_BPS = Gauge(
    "dist_to_sell_bps",
    "Distance to SELL entry in bps (0 if inside SELL zone)",
)

# risk mode & flags -------------------------------------------------------
RISK_MODE = Gauge(
    "risk_mode",
    "Risk mode active flag (1 for active mode, 0 for inactive)",
    ["mode"],  # "dynamic" or "fixed_basis"
)

RISK_FLAGS = Gauge(
    "risk_flags",
    "Risk state flags (0/1) for risk-related events",
    ["kind"],  # "daily_limit_hit" or "maxdd_hit"
)

# ---- Helper functions used from live_executor.py ---------------------------


def start_metrics_server(port: int) -> None:
    """
    Start the Prometheus metrics HTTP server on the given port.
    Called once at startup from live_executor.py.
    """
    start_http_server(port)


def mark_signal_metrics(ratio: float, dist_buy_bps: float, dist_sell_bps: float) -> None:
    """
    Update signal-related metrics in one call.

    ratio          → SIGNAL_RATIO (can be +/-)
    dist_buy_bps   → distance to BUY band in bps (clamped at >= 0)
    dist_sell_bps  → distance to SELL band in bps (clamped at >= 0)
    """
    SIGNAL_RATIO.set(float(ratio))
    DIST_TO_BUY_BPS.set(max(0.0, float(dist_buy_bps)))
    DIST_TO_SELL_BPS.set(max(0.0, float(dist_sell_bps)))


def mark_gate(open_: bool) -> None:
    """
    Update GATE_STATE{gate_state=...} so that exactly one of 'open'/'closed'
    is 1 and the other is 0.
    """
    GATE_STATE.labels("open").set(1.0 if open_ else 0.0)
    GATE_STATE.labels("closed").set(0.0 if open_ else 1.0)


def mark_zone(zone: str) -> None:
    """
    Mark the current signal zone:
      'buy_band', 'sell_band', or 'neutral'.
    """
    for z in ("buy_band", "sell_band", "neutral"):
        SIGNAL_ZONE.labels(z).set(1.0 if z == zone else 0.0)


def mark_decision(kind: str) -> None:
    """
    Mark the decision for this bar.

    Expected values:
      'exec_buy', 'exec_sell',
      'skip_threshold', 'skip_balance', 'skip_min_notional',
      'skip_cooldown', 'skip_gate_closed', 'skip_delta_zero'
    """
    all_kinds = [
        "exec_buy",
        "exec_sell",
        "skip_threshold",
        "skip_balance",
        "skip_min_notional",
        "skip_cooldown",
        "skip_gate_closed",
        "skip_delta_zero",
    ]
    for k in all_kinds:
        TRADE_DECISION.labels(k).set(1.0 if k == kind else 0.0)


def inc_skip(reason: str) -> None:
    """
    Increment skips_total{reason=...}.
    `reason` in {"threshold", "balance", "min_notional", "cooldown", "gate_closed", "delta_zero"}.
    """
    SKIPS.labels(reason=reason).inc()

def mark_risk_mode(active_mode: str) -> None:
    """
    Set risk_mode{mode="..."} as 1 for the active mode, 0 for others.
    Expected modes: "fixed_basis", "dynamic".
    """
    try:
        for m in ("fixed_basis", "dynamic"):
            RISK_MODE.labels(mode=m).set(1.0 if m == active_mode else 0.0)
    except Exception:
        # Keep metrics failures from breaking the bot
        pass


def mark_risk_flags(*, daily_limit_hit: bool, maxdd_hit: bool) -> None:
    """
    Expose risk flags as 0/1 via risk_flags{kind="daily_limit_hit|maxdd_hit"}.
    """
    try:
        RISK_FLAGS.labels(kind="daily_limit_hit").set(1.0 if daily_limit_hit else 0.0)
        RISK_FLAGS.labels(kind="maxdd_hit").set(1.0 if maxdd_hit else 0.0)
    except Exception:
        pass
    
# --- Snapshot helpers -------------------------------------------------------

def set_delta_metrics(delta_w: float, delta_eth: float) -> None:
    """
    Record planned Δweight and ΔETH for this bar.
    """
    DELTA_W.set(delta_w)
    DELTA_ETH.set(delta_eth)

def snapshot_wealth_balances(
    wealth_btc: float,
    price_mid: float,
    btc_free: float,
    eth_free: float,
) -> None:
    """
    One-shot snapshot of wealth + balances + mid price.
    """
    WEALTH_BTC_TOTAL.set(wealth_btc)
    PRICE_MID.set(price_mid)
    BAL_FREE.labels("btc").set(btc_free)
    BAL_FREE.labels("eth").set(eth_free)


def snapshot_wealth(wealth_btc: float) -> None:
    """
    Backwards-compatible: snapshot only wealth in BTC.
    """
    WEALTH_BTC_TOTAL.set(wealth_btc)


def snapshot_balances(btc_free: float, eth_free: float, price_mid: float) -> None:
    """
    Backwards-compatible: snapshot balances + mid price.
    """
    BAL_FREE.labels("btc").set(btc_free)
    BAL_FREE.labels("eth").set(eth_free)
    PRICE_MID.set(price_mid)