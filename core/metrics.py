# core/metrics.py
from prometheus_client import Counter, Gauge, Summary, start_http_server, REGISTRY

# --- Reload-safe registry setup for tests ------------------------------------
# Pytest's test_metrics_risk.py uses importlib.reload(core.metrics) between tests.
# Prometheus normally disallows re-registering the same metric names in the same
# CollectorRegistry, so on reload we get "Duplicated timeseries" errors.
#
# In this project, core.metrics is the only place we define metrics on the
# default REGISTRY, so it's safe to clear it on module import to make reloads
# idempotent.
try:
    REGISTRY._names_to_collectors.clear()
    REGISTRY._collector_to_names.clear()
except Exception:
    # If internal structure changes in future prometheus_client versions,
    # just fail silently instead of blowing up here.
    pass

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

NEXT_ACTION_DIST_BPS = Gauge(
    "next_action_dist_bps",
    "Distance to nearest band entry in bps (0 if inside a band)",
)

# Risk mode: 1 for active mode, 0 for inactive
RISK_MODE = Gauge(
    "risk_mode",
    "Risk mode active flag (1 for active mode, 0 for inactive)",
    ["mode"],
)

# Risk flags: 0/1 for risk events (daily loss / max drawdown)
RISK_FLAGS = Gauge(
    "risk_flags",
    "Risk state flags (0/1) for risk-related events",
    ["kind"],
)

# Overall trade readiness: 1 = all high-level conditions OK, 0 = something blocks
TRADE_READY = Gauge(
    "trade_ready",
    "Overall trade readiness (1 = strategy & risk say 'OK to trade', 0 = blocked).",
)

# Per-condition readiness: each label is a step in the decision pipeline.
# cond in {zone_ok, gate_open, delta_ok, risk_ok, balance_ok, size_ok}
TRADE_READY_COND = Gauge(
    "trade_ready_condition",
    "Per-condition readiness flags (0/1) for trade decisions.",
    ["cond"],
)

FUNDING_RATE = Gauge("funding_rate_pct", "Current funding rate percentage")

# ---- Helper functions used from live_executor.py ---------------------------

def mark_trade_readiness(
    zone_ok: bool,
    gate_ok: bool,
    delta_ok: bool,
    risk_ok: bool,
    balance_ok: bool,
    size_ok: bool,
) -> None:
    """
    Update per-condition 0/1 flags and overall trade_ready.

    We keep this intentionally *high-level*:
      - zone_ok: we are in BUY or SELL band (not neutral)
      - gate_ok: gate is OPEN
      - delta_ok: |Δw| >= rebalance threshold
      - risk_ok: daily/maxDD risk not hit
      - balance_ok: (optionally) sufficient balance to do the intended side
      - size_ok: (optionally) notional >= minimum size

    For now we will pass balance_ok=True, size_ok=True from live_executor
    so this is purely “strategy + risk”; you can refine later if you want.
    """
    try:
        TRADE_READY_COND.labels("zone_ok").set(1 if zone_ok else 0)
        TRADE_READY_COND.labels("gate_open").set(1 if gate_ok else 0)
        TRADE_READY_COND.labels("delta_ok").set(1 if delta_ok else 0)
        TRADE_READY_COND.labels("risk_ok").set(1 if risk_ok else 0)
        TRADE_READY_COND.labels("balance_ok").set(1 if balance_ok else 0)
        TRADE_READY_COND.labels("size_ok").set(1 if size_ok else 0)

        ready = (
            zone_ok
            and gate_ok
            and delta_ok
            and risk_ok
            and balance_ok
            and size_ok
        )
        TRADE_READY.set(1 if ready else 0)
    except Exception:
        # never let metrics crash the bot
        pass

def start_metrics_server(port: int) -> None:
    """
    Start the Prometheus metrics HTTP server on the given port.
    Called once at startup from live_executor.py.
    """
    start_http_server(port)


def mark_signal_metrics(ratio: float, dist_buy_bps: float, dist_sell_bps: float) -> None:
    SIGNAL_RATIO.set(ratio)
    DIST_TO_BUY_BPS.set(dist_buy_bps)
    DIST_TO_SELL_BPS.set(dist_sell_bps)

    # New: “distance to next action” = min(BUY, SELL)
    try:
        NEXT_ACTION_DIST_BPS.set(min(dist_buy_bps, dist_sell_bps))
    except Exception:
        # Defensive: don’t break if gauge missing in some env
        pass


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
    Mark the currently active risk mode.

    Parameters
    ----------
    active_mode : str
        Either "dynamic" or "fixed_basis" (or any future mode).
    """
    # We explicitly keep the two canonical modes in sync:
    for mode in ("fixed_basis", "dynamic"):
        value = 1.0 if mode == active_mode else 0.0
        RISK_MODE.labels(mode=mode).set(value)


def mark_risk_flags(*, daily_limit_hit: bool, maxdd_hit: bool) -> None:
    """
    Update risk flags gauges based on current risk state.

    Parameters
    ----------
    daily_limit_hit : bool
        True if daily loss limit is currently tripped.
    maxdd_hit : bool
        True if max drawdown limit is currently tripped.
    """
    RISK_FLAGS.labels(kind="daily_limit_hit").set(1.0 if daily_limit_hit else 0.0)
    RISK_FLAGS.labels(kind="maxdd_hit").set(1.0 if maxdd_hit else 0.0)

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

def mark_funding_rate(rate: float) -> None:
    """
    Record the current funding rate (in %) to Prometheus.
    """
    FUNDING_RATE.set(rate)