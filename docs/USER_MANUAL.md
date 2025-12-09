# ETH/BTC Accumulation Bot — Complete User Manual

> **Version**: 5.4 | **Last Updated**: December 2024

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Quick Start Guide](#2-quick-start-guide)
3. [Installation & Setup](#3-installation--setup)
4. [Architecture Overview](#4-architecture-overview)
5. [Configuration Reference](#5-configuration-reference)
6. [Strategies In-Depth](#6-strategies-in-depth)
7. [Backtesting Guide](#7-backtesting-guide)
8. [Optimization Workflow](#8-optimization-workflow)
9. [Live Trading](#9-live-trading)
10. [ML Regime Detection](#10-ml-regime-detection)
11. [Monitoring & Observability](#11-monitoring--observability)
12. [Docker Deployment](#12-docker-deployment)
13. [CLI Command Reference](#13-cli-command-reference)
14. [Configuration Examples](#14-configuration-examples)
15. [Troubleshooting](#15-troubleshooting)
16. [FAQ](#16-faq)
17. [Appendix](#17-appendix)

---

# 1. Introduction

## 1.1 What is this Bot?

This is a **professional-grade cryptocurrency trading bot** designed to accumulate a quote asset (e.g., BTC when trading ETHBTC, or USDT when trading BTCUSDT) through algorithmic trading on Binance.

### Core Philosophy

| Principle | Description |
|-----------|-------------|
| **Accumulation Focus** | Goals is to grow quote asset, not speculate on price |
| **Regime-Aware** | Automatically switches strategies based on market conditions |
| **Safety-First** | Multiple circuit breakers prevent catastrophic losses |
| **Parity Guaranteed** | Backtest and live execution use identical code paths |

### Supported Trading Modes

| Mode | Exchange | Leverage | Shorting |
|------|----------|----------|----------|
| **Spot** | Binance Spot | 1x | No |
| **Futures** | Binance USDS-M | 1-10x | Yes (configurable) |

## 1.2 Key Features

- ✅ Three trading strategies (Mean Reversion, Trend Following, Meta)
- ✅ Optuna-based hyperparameter optimization
- ✅ Walk-Forward Optimization (WFO) support
- ✅ ML-based regime detection (optional)
- ✅ Real-time Prometheus metrics + Grafana dashboards
- ✅ Human-readable "Story" logging with Discord/Telegram alerts
- ✅ Phoenix Protocol for automatic recovery from drawdowns
- ✅ Docker-based multi-bot deployment

---

# 2. Quick Start Guide

## 2.1 Minimum Requirements

- Python 3.11+ (3.12 recommended)
- 4GB RAM minimum
- Binance API keys with trading permissions
- Historical data (downloadable via included tools)

## 2.2 5-Minute Quick Start

```bash
# 1. Clone and setup
cd /path/to/ethbtc_bot_3
pip install -r requirements.txt

# 2. Download historical data
python tools/download_vision.py --symbol ETHBTC --start 2021-01-01 --interval 15m

# 3. Run a backtest with default config
python core/ethbtc_accum_bot.py backtest \
  --data data/raw/ETHBTC_15m_2021-2025_vision.csv \
  --config configs/prod_meta_live.json

# 4. (Optional) Start testnet trading
export BINANCE_KEY=your_testnet_key
export BINANCE_SECRET=your_testnet_secret
python live_executor.py --params configs/prod_meta_live.json --mode testnet
```

---

# 3. Installation & Setup

## 3.1 Python Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies

| Package | Purpose |
|---------|---------|
| `binance-connector` | Spot API client |
| `binance-futures-connector` | Futures API client |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `pydantic` | Config validation |
| `optuna` | Hyperparameter optimization |
| `scikit-learn` | ML regime detection |
| `prometheus-client` | Metrics export |

## 3.2 API Key Setup

Create a `.env` file in the project root:

```env
# Production Keys (Mainnet)
BINANCE_KEY=your_api_key_here
BINANCE_SECRET=your_api_secret_here

# Testnet Keys (Recommended for testing)
BINANCE_TESTNET_KEY=your_testnet_key
BINANCE_TESTNET_SECRET=your_testnet_secret

# Optional: Separate Futures keys
BINANCE_FUTURES_KEY=your_futures_key
BINANCE_FUTURES_SECRET=your_futures_secret

# Discord Alerts (Optional)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

### API Key Permissions Required

| Permission | Spot | Futures |
|------------|------|---------|
| Read | ✅ Required | ✅ Required |
| Spot Trading | ✅ Required | - |
| Futures Trading | - | ✅ Required |
| Withdraw | ❌ Never enable | ❌ Never enable |

## 3.3 Download Historical Data

### OHLCV Price Data

```bash
# ETHBTC (15-minute bars, 2021-2025)
python tools/download_vision.py \
  --symbol ETHBTC \
  --start 2021-01-01 \
  --end 2025-01-01 \
  --interval 15m \
  --output data/raw/ETHBTC_15m_2021-2025_vision.csv
```

### Funding Rate Data (for Futures)

```bash
python tools/download_funding.py \
  --symbol ETHUSDT \
  --start 2021-01-01 \
  --output data/raw/ETHUSDT_funding_2021-2025.csv
```

---

# 4. Architecture Overview

## 4.1 System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        LIVE EXECUTOR                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐    │
│  │ Config       │──│ Strategy     │──│ Exchange Adapter      │    │
│  │ Schema       │  │ Factory      │  │ (Spot / Futures)      │    │
│  └──────────────┘  └──────────────┘  └──────────────────────┘    │
│         │                 │                     │                 │
│         ▼                 ▼                     ▼                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐    │
│  │ Risk Manager │  │ Meta Strategy│  │ Order Execution       │    │
│  │ (DD, Phoenix)│  │ (MR + Trend) │  │ (TWAP / Maker)        │    │
│  └──────────────┘  └──────────────┘  └──────────────────────┘    │
│         │                 │                     │                 │
│         ▼                 ▼                     ▼                 │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              OBSERVABILITY LAYER                          │    │
│  │  Prometheus Metrics │ Story Writer │ Discord Alerts       │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

## 4.2 Directory Structure

```
ethbtc_bot_3/
├── configs/                   # Strategy configurations (JSON)
├── core/                      # Core strategy & execution modules
│   ├── ethbtc_accum_bot.py        # Mean Reversion + Backtester
│   ├── trend_strategy.py          # Trend Following
│   ├── meta_strategy.py           # Meta (Ensemble) Strategy
│   ├── regime.py                  # ADX regime detection
│   ├── position_sizer.py          # Dynamic sizing
│   ├── config_schema.py           # Pydantic validation
│   ├── binance_adapter.py         # Spot adapter
│   ├── futures_adapter.py         # Futures adapter
│   ├── story_writer.py            # Narrative logging
│   ├── alert_manager.py           # Discord/Telegram
│   └── metrics.py                 # Prometheus metrics
├── tools/                     # CLI utilities
│   ├── optimizer_cli.py           # MR optimizer
│   ├── optimize_trend.py          # Trend optimizer
│   ├── optimize_meta.py           # Meta threshold optimizer
│   ├── optimize_ml_regime.py      # ML regime optimizer
│   ├── download_vision.py         # Historical data
│   ├── download_funding.py        # Funding rates
│   └── wf_pick.py                 # Walk-forward selection
├── data/raw/                  # Historical CSV files
├── models/                    # Trained ML models
├── grafana/                   # Dashboard JSON exports
├── run_state/                 # Persistent bot state
├── live_executor.py           # Main entry point
├── run_complete_optimization.sh  # Full optimization workflow
└── docker-compose.yml         # Multi-bot deployment
```

---

# 5. Configuration Reference

## 5.1 Configuration File Structure

All configurations are JSON files validated by Pydantic. The complete schema:

```json
{
  "fees": {
    "maker_fee": 0.0002,
    "taker_fee": 0.0004,
    "slippage_bps": 1.0,
    "bnb_discount": 0.25,
    "pay_fees_in_bnb": true
  },
  "strategy": {
    "strategy_type": "meta",
    "bar_interval_minutes": 15,
    "trend_kind": "sma",
    "trend_lookback": 200,
    "flip_band_entry": 0.025,
    "flip_band_exit": 0.015,
    "vol_window": 45,
    "vol_adapt_k": 0.0075,
    "cooldown_minutes": 60,
    "rebalance_threshold_w": 0.03,
    "step_allocation": 0.33,
    "max_position": 1.0,
    "long_only": true,
    "position_sizing_mode": "volatility",
    "target_vol": 0.5,
    "adx_threshold": 25.0,
    "use_ml_regime": false,
    "ml_model_path": "models/regime_classifier_v1.pkl",
    "ml_threshold": 50.0,
    "mean_reversion_overrides": {},
    "trend_overrides": {}
  },
  "execution": {
    "exchange_type": "futures",
    "leverage": 1,
    "interval": "15m",
    "poll_sec": 5,
    "ttl_sec": 30,
    "taker_fallback": false
  },
  "risk": {
    "basis_btc": 1.0,
    "risk_mode": "dynamic",
    "max_dd_frac": 0.15,
    "max_daily_loss_frac": 0.05,
    "drawdown_reset_days": 7.0,
    "drawdown_reset_score": 30.0
  }
}
```

## 5.2 Fees Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `maker_fee` | float | 0.0002 | Maker fee (0.02%) |
| `taker_fee` | float | 0.0004 | Taker fee (0.04%) |
| `slippage_bps` | float | 1.0 | Expected slippage in basis points |
| `bnb_discount` | float | 0.25 | BNB fee discount (25%) |
| `pay_fees_in_bnb` | bool | true | Use BNB for fees |

## 5.3 Strategy Section

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy_type` | string | "meta" | "mean_reversion", "trend", or "meta" |
| `bar_interval_minutes` | int | 15 | Bar interval |
| `trend_kind` | string | "sma" | "sma" or "roc" for trend calculation |
| `trend_lookback` | int | 200 | Bars for SMA/ROC calculation |
| `flip_band_entry` | float | 0.025 | Entry threshold (e.g., 2.5% deviation) |
| `flip_band_exit` | float | 0.015 | Exit threshold (1.5%) |
| `cooldown_minutes` | int | 180 | Minimum time between state flips (3 hours) |
| `step_allocation` | float | 0.5 | Position step size (50%) |
| `max_position` | float | 1.0 | Maximum position (100%) |
| `long_only` | bool | true | Disable shorting |

### Volatility Adaptation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vol_window` | int | 45 | Volatility calculation window |
| `vol_adapt_k` | float | 0.0075 | Band scaling factor |
| `target_vol` | float | 0.5 | Target volatility (50% annualized) |

### Position Sizing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `position_sizing_mode` | string | "volatility" | "static", "volatility", or "kelly" |

### Regime Detection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adx_threshold` | float | 25.0 | Threshold for ADX-based regime switch |
| `use_ml_regime` | bool | false | Enable ML regime detection |
| `ml_model_path` | string | "models/..." | Path to trained ML model |
| `ml_threshold` | float | 50.0 | ML probability threshold (0-100) |

### Strategy Overrides (Meta Strategy)

| Parameter | Type | Description |
|-----------|------|-------------|
| `mean_reversion_overrides` | object | Override MR-specific params |
| `trend_overrides` | object | Override Trend-specific params |

## 5.4 Execution Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `exchange_type` | string | "futures" | "spot" or "futures" |
| `leverage` | int | 1 | Leverage multiplier (futures only) |
| `interval` | string | "15m" | Trading interval |
| `poll_sec` | int | 5 | Poll frequency for order fills |
| `ttl_sec` | int | 30 | Order TTL before cancel |
| `taker_fallback` | bool | false | Fall back to taker if maker unfilled |

## 5.5 Risk Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `basis_btc` | float | 1.0 | Base position sizing (in quote) |
| `risk_mode` | string | "dynamic" | "fixed_basis" or "dynamic" |
| `max_dd_frac` | float | 0.15 | Max drawdown (15%) before halt |
| `max_daily_loss_frac` | float | 0.05 | Max daily loss (5%) |
| `drawdown_reset_days` | float | 7.0 | Days to wait for Phoenix |
| `drawdown_reset_score` | float | 30.0 | ADX threshold for Phoenix |

---

# 6. Strategies In-Depth

## 6.1 Mean Reversion Strategy

**File**: `core/ethbtc_accum_bot.py`

### Concept

Exploits the tendency of prices to revert to a mean after extreme moves. Works best in ranging/choppy markets (low ADX).

### Signal Generation

```python
# 1. Calculate ratio
ratio = (price / SMA(price, lookback)) - 1  # or ROC

# 2. Dynamic bands
entry_band = flip_band_entry + vol_adapt_k * volatility
exit_band = flip_band_exit + vol_adapt_k * volatility

# 3. State machine
if ratio < -entry_band:
    signal = "BUY" (oversold)
elif ratio > +entry_band:
    signal = "SELL" (overbought)
elif abs(ratio) < exit_band:
    signal = "NEUTRAL"
```

### Key Parameters

| Parameter | Meaning | Typical Range |
|-----------|---------|---------------|
| `trend_lookback` | SMA period | 100-300 bars |
| `flip_band_entry` | Deviation to enter | 0.02-0.05 (2-5%) |
| `flip_band_exit` | Deviation to exit | 0.01-0.03 (1-3%) |
| `vol_adapt_k` | Adjust bands with volatility | 0-0.01 |

## 6.2 Trend Following Strategy

**File**: `core/trend_strategy.py`

### Concept

Follows price momentum using moving average crossovers. Works best in trending markets (high ADX).

### Signal Generation

```python
# 1. Calculate MAs
fast_ma = EMA(price, fast_period)  # e.g., 50 bars
slow_ma = EMA(price, slow_period)  # e.g., 200 bars

# 2. Signal
if fast_ma > slow_ma:
    signal = "LONG"   # Golden Cross
else:
    signal = "SHORT"  # Death Cross (or NEUTRAL if long_only)
```

### Key Parameters

| Parameter | Meaning | Typical Range |
|-----------|---------|---------------|
| `fast_period` | Fast MA period | 20-100 bars |
| `slow_period` | Slow MA period | 100-500 bars |
| `ma_type` | Moving average type | "ema" or "sma" |

## 6.3 Meta Strategy (Ensemble)

**File**: `core/meta_strategy.py`

### Concept

Dynamically switches between Mean Reversion and Trend Following based on market regime (ADX score).

### Regime Detection

```python
# Multi-timeframe ADX
regime_score = 0.2 * ADX(15m) + 0.3 * ADX(30m) + 0.5 * ADX(1h)

# Hysteresis (prevents churn)
buffer = 2.0
if regime_score > (threshold + buffer):
    use_trend()
elif regime_score < (threshold - buffer):
    use_mean_reversion()
else:
    keep_previous_regime()
```

### When to Use Each Strategy

| Condition | ADX Score | Active Strategy |
|-----------|-----------|-----------------|
| Ranging/Consolidating | < 23 | Mean Reversion |
| Weak Trend | 23-27 | Keep previous |
| Strong Trend | > 27 | Trend Following |

---

# 7. Backtesting Guide

## 7.1 Running a Backtest

### Basic Command

```bash
python core/ethbtc_accum_bot.py backtest \
  --data data/raw/ETHBTC_15m_2021-2025_vision.csv \
  --config configs/prod_meta_live.json
```

### Full Options

```bash
python core/ethbtc_accum_bot.py backtest \
  --data data/raw/ETHBTC_15m_2021-2025_vision.csv \
  --config configs/prod_meta_live.json \
  --funding data/raw/ETHUSDT_funding_2021-2025.csv \
  --start 2023-01-01 \
  --end 2024-01-01 \
  --report \
  --output results/backtest_2023.xlsx
```

### CLI Arguments

| Argument | Description |
|----------|-------------|
| `--data` | Path to OHLCV CSV |
| `--config` | Path to config JSON |
| `--funding` | Optional funding rate CSV |
| `--start` | Start date (YYYY-MM-DD) |
| `--end` | End date (YYYY-MM-DD) |
| `--report` | Generate detailed report |
| `--output` | Output file path |

## 7.2 Interpreting Results

### Key Metrics

| Metric | Good Value | Description |
|--------|------------|-------------|
| **Sharpe Ratio** | > 1.0 | Risk-adjusted return |
| **Total Return** | > Buy & Hold | Actually beat the market |
| **Max Drawdown** | < 20% | Worst peak-to-trough loss |
| **Win Rate** | > 50% | Percentage of winning trades |
| **Profit Factor** | > 1.5 | Gross profit / Gross loss |
| **# Trades** | 100-2000 | Too few = not statistically valid |

### Report Sections

1. **Summary Statistics** - Overall performance
2. **Equity Curve** - Wealth over time
3. **Monthly Returns** - Breakdown by month
4. **Trade Log** - Individual trades
5. **Regime Analysis** - Time spent in each mode

---

# 8. Optimization Workflow

## 8.1 Complete Optimization Pipeline

The bot provides a comprehensive 6-step optimization workflow:

```bash
./run_complete_optimization.sh [OPTIONS]
```

### Options

| Flag | Description |
|------|-------------|
| `--wfo` | Walk-Forward Optimization mode |
| `--exhaustive` | Test all 8 combinations |
| `--train-start DATE` | Training start date |
| `--train-end DATE` | Training end date |
| `--test-start DATE` | Test start date |
| `--test-end DATE` | Test end date |

### Workflow Diagram

```
Step 1: Optimize Mean Reversion
    ├── Tests: trend_kind (sma, roc)
    ├── Tests: sizing_mode (static, volatility)
    └── Tests: long_only (true, false)
    → Output: results/opt_mr_*.csv

Step 2: Select Best MR Config
    └── Uses: wf_pick.py (family clustering)
    → Output: configs/best_mr_params_*.json

Step 3: Optimize Trend Following
    ├── Static mode: Single train/test split
    └── WFO mode: Rolling windows
    → Output: results/opt_trend_*.csv

Step 4: Select Best Trend Config
    ├── Static: wf_pick.py
    └── WFO: wfo_select_best.py
    → Output: configs/best_trend_params_*.json

Step 5: Optimize Meta Threshold
    └── Find optimal ADX switch point
    → Output: results/opt_meta_*.csv

Step 6: Assemble Final Config
    └── Merge MR + Trend + Meta
    → Output: configs/meta_optimized_v2_*.json
```

## 8.2 Individual Optimization Scripts

### Mean Reversion Optimizer

```bash
python tools/optimizer_cli.py \
  --data data/raw/ETHBTC_15m_2021-2025_vision.csv \
  --funding-data data/raw/ETHUSDT_funding_2021-2025.csv \
  --train-start 2021-01-01 \
  --train-end 2024-06-30 \
  --test-start 2024-07-01 \
  --test-end 2025-01-01 \
  --n-trials 50 \
  --jobs 4 \
  --out results/optimization_mr.csv
```

### Trend Optimizer

```bash
python tools/optimize_trend.py \
  --data data/raw/ETHBTC_15m_2021-2025_vision.csv \
  --train-start 2021-01-01 \
  --train-end 2024-06-30 \
  --test-start 2024-07-01 \
  --test-end 2025-01-01 \
  --n-trials 30 \
  --out results/optimization_trend.csv
```

### Meta Threshold Optimizer

```bash
python tools/optimize_meta.py \
  --data data/raw/ETHBTC_15m_2021-2025_vision.csv \
  --mr-config configs/best_mr.json \
  --trend-config configs/best_trend.json \
  --out results/optimization_meta.csv
```

## 8.3 Walk-Forward Optimization (WFO)

WFO tests parameter robustness across multiple time periods:

```bash
python tools/optimize_trend.py \
  --data data/raw/ETHBTC_15m_2021-2025_vision.csv \
  --wfo \
  --window-days 180 \
  --step-days 30 \
  --n-trials 30 \
  --out results/wfo_trend.csv
```

### WFO Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `--window-days` | Training window size | 180 days |
| `--step-days` | Step between windows | 30 days |
| `--n-trials` | Optuna trials per window | 30 |

---

# 9. Live Trading

## 9.1 Starting the Bot

### Testnet Mode (Recommended for First-Time)

```bash
python live_executor.py \
  --params configs/prod_meta_live.json \
  --mode testnet \
  --symbol ETHBTC
```

### Live Mode (Real Money)

```bash
python live_executor.py \
  --params configs/prod_meta_live.json \
  --mode live \
  --symbol ETHBTC
```

### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--params` | Config file path | Required |
| `--mode` | dry, testnet, live | Required |
| `--symbol` | Trading pair | ETHBTC |
| `--metrics-port` | Prometheus port | 9109 |
| `--status-port` | Status JSON port | 9110 |
| `--state-file` | Persistent state | run_state/symbol/state.json |

## 9.2 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `BINANCE_KEY` | API Key | Yes |
| `BINANCE_SECRET` | API Secret | Yes |
| `BINANCE_TESTNET_KEY` | Testnet key | For testnet |
| `BINANCE_TESTNET_SECRET` | Testnet secret | For testnet |
| `DISCORD_WEBHOOK_URL` | Discord alerts | No |
| `LOGLEVEL` | Log verbosity | No (default: INFO) |

## 9.3 State Persistence

The bot saves its state to disk after each trade decision:

```json
// run_state/eth/state_testnet.json
{
  "hwm_btc": 1.0,
  "last_known_signal": 0.33,
  "dd_halt_ts_utc": null,
  "daily_loss_halt_until": null,
  "last_regime_state": "TREND",
  "last_story_update": "2024-12-09T10:00:00Z"
}
```

## 9.4 Safety Gates

The bot implements multiple safety checks:

| Gate | Description | Action |
|------|-------------|--------|
| **Funding Rate** | Blocks entries when funding is extreme | Skip trade |
| **Trend Gate** | Blocks trades in flat markets | Skip trade |
| **Max Drawdown** | Halts trading after X% loss | Full halt |
| **Daily Loss** | Halts after X% daily loss | Halt until UTC midnight |

---

# 10. ML Regime Detection

## 10.1 Overview

The bot can optionally use Machine Learning instead of ADX for regime detection.

### ML vs ADX Comparison

| Method | Pros | Cons |
|--------|------|------|
| **ADX** | Simple, robust, no training needed | Reactive, not predictive |
| **ML** | Can predict regime changes early | Requires optimization, may overfit |

## 10.2 ML Optimization

```bash
python tools/optimize_ml_regime.py \
  --data data/raw/ETHBTC_15m_2021-2025_vision.csv \
  --funding data/raw/ETHUSDT_funding_2021-2025.csv \
  --fear-greed data/raw/fear_greed_index_2021-2025.csv \
  --config configs/prod_meta_live.json \
  --n-trials 50 \
  --output models/regime_classifier_optimized.pkl
```

### What Gets Optimized

| Parameter | Search Space | Description |
|-----------|--------------|-------------|
| `lookahead_bars` | 8-48 | How far ahead to predict |
| `adx_threshold` | 15-35 | Label generation threshold |
| `n_estimators` | 50-200 | Random Forest trees |
| `max_depth` | 5-20 | Tree depth |
| `ml_threshold` | 35-70 | Decision threshold |

## 10.3 Enabling ML Regime

After optimization, update your config:

```json
{
  "strategy": {
    "use_ml_regime": true,
    "ml_model_path": "models/regime_classifier_optimized.pkl",
    "ml_threshold": 60.0,
    "adx_threshold": 25.0  // Keep for fallback
  }
}
```

## 10.4 Features Used

The ML model uses these features:

| Feature | Description |
|---------|-------------|
| `adx_15m` | 15-minute ADX |
| `rsi_14` | 14-period RSI |
| `volume_ratio` | Volume vs 20-bar SMA |
| `bb_width` | Bollinger Band width |
| `price_roc` | 4-hour price change |
| `returns_vol` | Rolling volatility |
| `funding_rate` | Current funding |
| `funding_zscore` | Funding z-score |
| `fear_greed` | Fear & Greed Index |

---

# 11. Monitoring & Observability

## 11.1 Prometheus Metrics

The bot exposes 50+ metrics on port 9109:

### Key Metrics

| Metric | Description |
|--------|-------------|
| `wealth_total` | Portfolio value in quote |
| `wealth_usd` | Portfolio value in USD |
| `exposure_base_weight` | Current position (0-1) |
| `regime_score` | ADX trend score (0-100) |
| `strategy_mode` | 0=MR, 1=Trend |
| `phoenix_active` | 1=Halted |
| `fear_greed_index` | Market sentiment (0-100) |
| `ml_regime_active` | 1=ML mode |
| `trade_decision` | Last decision |

### Scrape Config (prometheus.yml)

```yaml
scrape_configs:
  - job_name: 'ethbtc_bots'
    static_configs:
      - targets: 
        - 'localhost:9109'  # ETH bot
        - 'localhost:9110'  # BTC bot
    scrape_interval: 15s
```

## 11.2 Grafana Dashboard

Import the dashboard from:
```
grafana/ethbtc_bot_grafana_live.json
```

### Dashboard Sections

1. **Command Center** - Total AUM, 24h PnL, Risk Status
2. **Fleet Status** - All bots at a glance
3. **Deep Dive** - Per-bot detailed charts
4. **Regime Analysis** - ADX, Fear/Greed, ML status

## 11.3 Story Writer

Human-readable logs in `results/story_*.txt`:

```
══════════════════════════════════════════════════════════════════
🚀 Bot Started | testnet | ETHBTC | 2024-12-09 10:00:00 UTC
══════════════════════════════════════════════════════════════════

10:15:00 | 🟢 BUY Signal | Price: 0.0502 | Weight: 0% → 33%
         | Regime: MR (ADX=18.5) | Ratio: -3.2% (Entry at -2.5%)
         | Filled: 0.65 ETH @ 0.05020 | Fee: 0.00002 BTC

10:30:00 | 🔄 Regime Switch | MR → TREND (ADX=28.5 > 27.0)

10:45:00 | 📈 New ATH | Wealth: 1.025 BTC (+2.5%)
```

## 11.4 Discord Alerts

Enable alerts by setting `DISCORD_WEBHOOK_URL`:

| Alert Type | Trigger |
|------------|---------|
| 🟢 Trade Executed | Any buy/sell |
| 🔄 Regime Switch | MR ↔ Trend |
| 📈 New ATH | All-time high wealth |
| 🚨 Risk Alert | Max DD or daily loss hit |
| 🔥 Phoenix Active | Recovery mode entered |

---

# 12. Docker Deployment

## 12.1 Single Bot

```bash
docker build -t ethbtc-bot .
docker run -d \
  --name ethbtc-bot \
  -e BINANCE_KEY=your_key \
  -e BINANCE_SECRET=your_secret \
  -v $(pwd)/run_state:/data \
  -p 9109:9109 \
  ethbtc-bot \
  python /app/live_executor.py --params configs/prod_meta_live.json --mode live
```

## 12.2 Multi-Bot with Docker Compose

```yaml
# docker-compose.yml
services:
  bot_eth:
    build: .
    environment:
      - BINANCE_KEY=${BINANCE_KEY}
      - BINANCE_SECRET=${BINANCE_SECRET}
    volumes:
      - ./run_state/eth:/data
      - ./configs:/app/configs:ro
    command: >
      python /app/live_executor.py 
        --params configs/prod_meta_live.json 
        --mode live 
        --symbol ETHBTC
    ports:
      - "9109:9109"
      - "9110:9110"

  bot_btc:
    build: .
    environment:
      - BINANCE_KEY=${BINANCE_KEY}
      - BINANCE_SECRET=${BINANCE_SECRET}
    volumes:
      - ./run_state/btc:/data
      - ./configs:/app/configs:ro
    command: >
      python /app/live_executor.py 
        --params configs/prod_btc_meta_live.json 
        --mode live 
        --symbol BTCUSDT
        --metrics-port 9111
    ports:
      - "9111:9111"
      - "9112:9110"

  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
```

### Commands

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f bot_eth

# Stop all
docker-compose down

# Rebuild after code changes
docker-compose build && docker-compose up -d
```

---

# 13. CLI Command Reference

## 13.1 Main Entry Points

### Backtesting

```bash
python core/ethbtc_accum_bot.py backtest --data FILE --config FILE [OPTIONS]
```

### Live Trading

```bash
python live_executor.py --params FILE --mode MODE [OPTIONS]
```

## 13.2 Data Tools

| Command | Description |
|---------|-------------|
| `python tools/download_vision.py` | Download OHLCV data |
| `python tools/download_funding.py` | Download funding rates |
| `python scripts/fetch_fear_greed.py` | Download Fear & Greed Index |

## 13.3 Optimization Tools

| Command | Description |
|---------|-------------|
| `python tools/optimizer_cli.py` | Mean Reversion optimizer |
| `python tools/optimize_trend.py` | Trend optimizer |
| `python tools/optimize_meta.py` | Meta threshold optimizer |
| `python tools/optimize_ml_regime.py` | ML regime optimizer |
| `python tools/wf_pick.py` | Parameter family selector |
| `python tools/wfo_select_best.py` | WFO aggregator |
| `python tools/assemble_v2_config.py` | Config assembler |

## 13.4 Utility Tools

| Command | Description |
|---------|-------------|
| `python tools/sanity_check_config.py` | Validate config |
| `python tools/analyze_meta.py` | Analyze meta strategy results |
| `python tools/reconcile_pnl.py` | Compare backtest vs live PnL |
| `python tools/dust_sweeper.py` | Convert dust to BNB |

## 13.5 Complete Workflows

```bash
# Full optimization (all steps)
./run_complete_optimization.sh

# With Walk-Forward Optimization
./run_complete_optimization.sh --wfo

# With custom date range
./run_complete_optimization.sh \
  --train-start 2022-01-01 \
  --train-end 2024-06-30 \
  --test-start 2024-07-01 \
  --test-end 2025-01-01

# Exhaustive mode (all 8 combinations)
./run_complete_optimization.sh --exhaustive
```

---

# 14. Configuration Examples

## 14.1 Conservative Spot Trading (ETHBTC)

```json
{
  "fees": {
    "maker_fee": 0.0002,
    "taker_fee": 0.0004,
    "slippage_bps": 1.0,
    "bnb_discount": 0.25,
    "pay_fees_in_bnb": true
  },
  "strategy": {
    "strategy_type": "meta",
    "trend_kind": "sma",
    "trend_lookback": 200,
    "flip_band_entry": 0.03,
    "flip_band_exit": 0.015,
    "step_allocation": 0.25,
    "max_position": 0.75,
    "long_only": true,
    "adx_threshold": 25.0
  },
  "execution": {
    "exchange_type": "spot",
    "interval": "15m",
    "poll_sec": 5
  },
  "risk": {
    "basis_btc": 0.5,
    "risk_mode": "fixed_basis",
    "max_dd_frac": 0.10,
    "max_daily_loss_frac": 0.03
  }
}
```

## 14.2 Aggressive Futures Trading (BTCUSDT)

```json
{
  "fees": {
    "maker_fee": 0.0001,
    "taker_fee": 0.0003,
    "slippage_bps": 0.5
  },
  "strategy": {
    "strategy_type": "meta",
    "trend_kind": "roc",
    "trend_lookback": 150,
    "flip_band_entry": 0.02,
    "step_allocation": 0.5,
    "max_position": 1.0,
    "long_only": false,
    "position_sizing_mode": "volatility",
    "target_vol": 0.4,
    "adx_threshold": 22.0
  },
  "execution": {
    "exchange_type": "futures",
    "leverage": 3,
    "interval": "15m"
  },
  "risk": {
    "basis_btc": 1000,
    "risk_mode": "dynamic",
    "max_dd_frac": 0.20,
    "max_daily_loss_frac": 0.05,
    "drawdown_reset_days": 7.0
  }
}
```

## 14.3 ML-Enhanced Meta Strategy

```json
{
  "strategy": {
    "strategy_type": "meta",
    "adx_threshold": 25.0,
    "use_ml_regime": true,
    "ml_model_path": "models/regime_classifier_optimized.pkl",
    "ml_threshold": 60.0
  }
}
```

---

# 15. Troubleshooting

## 15.1 Common Issues

### Bot Not Trading

| Symptom | Cause | Solution |
|---------|-------|----------|
| All decisions are `skip_threshold` | Price within bands | Normal - wait for entry signal |
| All decisions are `skip_balance` | Insufficient funds | Check balance, adjust basis_btc |
| `phoenix_active = 1` | Max DD hit | Wait for Phoenix reset or manual override |
| `gate_ok = 0` | Funding or trend gate blocked | Normal safety feature |

### Connection Errors

| Error | Solution |
|-------|----------|
| `APIError(code=-1021)` | Timestamp sync issue - check system time |
| `APIError(code=-2015)` | Invalid API key - check credentials |
| Connection timeout | Check network, Binance status |

### Strategy Errors

| Error | Solution |
|-------|----------|
| `index must be monotonic` | Data timestamp issues - check data quality |
| `ValueError: Need OHLC` | Passing Series instead of DataFrame |
| `KeyError: 'close'` | Missing columns in data |

## 15.2 Log Levels

```bash
# Verbose debugging
LOGLEVEL=DEBUG python live_executor.py ...

# Production (default)
LOGLEVEL=INFO python live_executor.py ...

# Minimal
LOGLEVEL=WARNING python live_executor.py ...
```

## 15.3 Data Quality Checks

```bash
# Check OHLCV data
python -c "
import pandas as pd
df = pd.read_csv('data/raw/ETHBTC_15m_2021-2025_vision.csv', index_col=0, parse_dates=True)
print(f'Rows: {len(df)}')
print(f'Date range: {df.index.min()} to {df.index.max()}')
print(f'Missing values: {df.isna().sum().sum()}')
print(f'Columns: {list(df.columns)}')
"
```

---

# 16. FAQ

## General

**Q: What's the minimum capital to start?**
A: Depends on the trading pair's minimum order size. For ETHBTC spot, ~0.01 BTC. For futures, varies by leverage.

**Q: Can I run multiple bots on the same account?**
A: Yes, but use different symbols or separate sub-accounts to avoid position conflicts.

**Q: Does the bot work 24/7?**
A: Yes, cryptocurrency markets never close. Use Docker for reliability.

## Strategy

**Q: Why does the bot sometimes not trade for hours?**
A: The bot only trades when signals hit the entry bands. Sideways markets may not trigger trades.

**Q: What's the difference between `static` and `volatility` sizing?**
A: Static uses fixed position size. Volatility reduces size in volatile markets to maintain consistent risk.

**Q: Should I use `long_only: true`?**
A: For spot trading, yes (can't short). For futures, depends on your risk tolerance.

## Technical

**Q: Why is my backtest different from live results?**
A: Common causes:
1. Slippage not accurately modeled
2. Funding rate data missing
3. Different data source (Binance Vision vs API)

**Q: How do I reset the Phoenix halt?**
A: Delete or edit `run_state/SYMBOL/state.json` and remove `dd_halt_ts_utc`.

---

# 17. Appendix

## A. Glossary

| Term | Definition |
|------|------------|
| **ADX** | Average Directional Index - measures trend strength |
| **HWM** | High Water Mark - highest portfolio value |
| **Phoenix** | Auto-recovery protocol after max drawdown |
| **WFO** | Walk-Forward Optimization |
| **Meta Strategy** | Ensemble of MR + Trend strategies |

## B. File Locations

| Purpose | Path |
|---------|------|
| Configs | `configs/*.json` |
| Historical Data | `data/raw/*.csv` |
| ML Models | `models/*.pkl` |
| Bot State | `run_state/SYMBOL/state.json` |
| Logs | `logs/*.log` |
| Story Logs | `results/story_*.txt` |
| Optimization Results | `results/opt_*.csv` |

## C. Version History

| Version | Date | Changes |
|---------|------|---------|
| 5.4 | Dec 2024 | ML regime detection, Fear/Greed integration |
| 5.3 | Nov 2024 | Futures support, Phoenix Protocol |
| 5.2 | Oct 2024 | Meta Strategy, WFO |
| 5.1 | Sep 2024 | Prometheus metrics |
| 5.0 | Aug 2024 | Initial release |

---

*This manual covers ETH/BTC Accumulation Bot v5.4. For bug reports or feature requests, contact the maintainer.*
