# Grafana Dashboard Implementation Gap Analysis

## Overview
This document compares the requirements of the new "Hybrid Spot & Futures – Multi-Bot Grafana Dashboard" specification against the current codebase instrumentation (`core/metrics.py`).

## 1. ✅ KPIs Already Available (Ready to Wire)
These metrics exist and can be mapped directly to the new dashboard panels.

| Panel / Feature | Existing Metric(s) | Notes |
| :--- | :--- | :--- |
| **Total AUM** | `wealth_usd` | Sum across instances. |
| **Active Bots** | `up` | Standard Prometheus metric. |
| **PnL (24h/Total)** | `wealth_total`, `pnl_quote` | Use `delta(wealth_total[24h])` in Grafana. |
| **Asset Balances** | `balance_free` | Label `asset` allows breakdown. |
| **Exposure (Target)** | `exposure_signal_weight` | Unleveraged target. |
| **Exposure (Actual/Lev)** | `exposure_notional`, `leverage` | Realized exposure. |
| **Funding Rate** | `funding_rate_pct` | Snapshot current rate. |
| **Latency** | `bar_latency_seconds` | System health. |
| **Regime/State** | `regime_score`, `regime_state` | Strategy internals. |
| **Mode** | `config_long_only`, `strategy_mode` | Configuration state. |
| **Trade Counts** | `fills_total`, `skips_total` | Execution activity. |

## 2. ⚠️ Missing Instrumentation (Development Required)
These requirements cannot be met with current metrics and require code changes in `live_executor.py` or `core/`.

### A. Financial & Risk
*   **[CRITICAL] Margin Utilization**: No metric for used margin vs total margin.
    *   *Need*: `margin_utilization_pct{venue="..."}`.
*   **[CRITICAL] Liquidation Distance**: Critical for Futures monitoring.
    *   *Need*: `liquidation_distance_pct{symbol="..."}`.
*   **PnL Drift (Recon)**: Difference between internal and external equity.
    *   *Need*: `pnl_drift_pct` (requires `reconcile_pnl.py` integration).

### B. Execution Quality
*   **Slippage**: Distribution of execution slippage.
    *   *Need*: Summary/Histogram `execution_slippage_bps` (Recorded at fill time).
*   **Fees & Funding Paid**: Cumulative costs.
    *   *Need*: Counter `fees_paid_total`, `funding_paid_total` (by asset/symbol).
*   **Win Rate / R:R**: Rolling performance stats.
    *   *Need*: Gauges `win_rate_rolling`, `risk_reward_rolling` (or compute in backend).

### C. Operational Health
*   **Error Counts**: Log-based error tracking.
    *   *Need*: Counter `errors_total{type="..."}` or Promtail integration.
*   **Last Trade Time**: Timestamp of last execution.
    *   *Need*: Gauge `last_trade_timestamp_seconds`.

## 3. Needs Clarification / Details
*   **"Gross vs Net Exposure"**: We have `exposure_notional` (Net?). Do we track individual leg sizes for Gross calculation if using Hedged mode?
    *   *Assumption*: `exposure_notional` is Net. Gross requires separate Long/Short exposure metrics if in Hedged mode.
*   **"Bot Heartbeats"**: Is `up` metric sufficient (scraper health)? Or do we need application-level `heartbeat_timestamp` to detect zombie processes that scrape but don't trade?

## 4. Next Steps
1.  **Phase 1 (Immediate)**: Build dashboard using "Ready" metrics (Wealth, PnL, Exposure, Signal).
2.  **Phase 2 (Code)**: Instrument `margin_utilization`, `liquidation_distance`, and `slippage`.
3.  **Phase 3 (Enrichment)**: Add specialized execution stats (Win Rate, Fees).
