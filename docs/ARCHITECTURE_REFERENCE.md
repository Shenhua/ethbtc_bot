# 📚 ETH/BTC Bot v5 — Architecture Reference

> **Purpose**: Comprehensive technical reference for understanding the bot's architecture, key concepts, and component interactions. Use this as a quick reference when modifying or debugging the system.

---

## 1. Project Overview

### What is this bot?

This is a **cryptocurrency trading bot** that operates on Binance (Spot or USDS-M Futures) with three key strategies:

| Strategy | Description | When Used |
|----------|-------------|-----------|
| **Mean Reversion (MR)** | Buys when price is oversold, sells when overbought relative to a moving trend | Low ADX (choppy/ranging market) |
| **Trend Following** | Follows EMA/SMA crossover signals | High ADX (trending market) |
| **Meta Strategy** | Ensemble that dynamically switches between MR and Trend based on ADX regime score | Default production mode |

### Core Philosophy

- **Accumulate the Quote Asset** (e.g., BTC if trading ETHBTC, USDT if trading BTCUSDT)
- **Safety First**: Multiple circuit breakers (funding rate gates, max drawdown, daily loss limits)
- **Phoenix Protocol**: Auto-recovery after max drawdown hits, based on time + regime conditions

---

## 2. Directory Structure

```
ethbtc_bot_3/
├── configs/                   # JSON strategy configurations
│   ├── prod_btc_meta_live.json     # Production Meta Strategy config
│   ├── best_mr_params_*.json       # Optimized MR parameters
│   └── best_trend_params_*.json    # Optimized Trend parameters
│
├── core/                      # Core strategy & execution modules
│   ├── ethbtc_accum_bot.py        # Mean Reversion strategy + Backtester
│   ├── trend_strategy.py          # Trend Following strategy
│   ├── meta_strategy.py           # Ensemble Meta Strategy
│   ├── regime.py                  # ADX calculation & regime detection
│   ├── position_sizer.py          # Dynamic position sizing
│   ├── config_schema.py           # Pydantic config validation
│   ├── binance_adapter.py         # Spot exchange adapter
│   ├── futures_adapter.py         # USDS-M Futures adapter
│   ├── story_writer.py            # Real-time narrative logging
│   ├── alert_manager.py           # Discord/Telegram alerts
│   ├── metrics.py                 # Prometheus metrics
│   └── twap_maker.py              # TWAP/Maker-only execution
│
├── tools/                     # CLI utilities & optimization
│   ├── optimizer_cli.py           # Mean Reversion optimizer (Optuna)
│   ├── optimize_trend.py          # Trend strategy optimizer
│   ├── optimize_meta.py           # Meta threshold optimizer
│   ├── wf_pick.py                 # Walk-Forward family selection
│   ├── wfo_select_best.py         # WFO slice aggregation
│   ├── assemble_v2_config.py      # Final config assembly
│   ├── download_vision.py         # Historical data downloader
│   ├── download_funding.py        # Funding rate downloader
│   └── sanity_check_config.py     # Config validator
│
├── data/                      # Historical data CSVs
│   └── raw/                       # Binance Vision CSVs
│
├── run_state/                 # Persistent bot state (per-symbol)
│   ├── btc/state_testnet.json
│   ├── eth/state_testnet.json
│   └── bnb/state_testnet.json
│
├── results/                   # Optimization outputs
├── tests/                     # Unit tests
├── live_executor.py           # Main entry point (1378 lines!)
├── run_complete_optimization.sh  # Full optimization workflow
├── docker-compose.yml         # Multi-bot deployment
└── requirements.txt
```

---

## 3. Core Strategies

### 3.1 Mean Reversion (`EthBtcStrategy`)

**File**: `core/ethbtc_accum_bot.py`

**Logic**:
1. Calculate ratio: `price / SMA(price, lookback) - 1` (or ROC)
2. Dynamic bands: `entry = flip_band_entry + vol_adapt_k * volatility`
3. State machine:
   - If `ratio < -entry`: **BUY** (oversold)
   - If `ratio > +entry`: **SELL/SHORT** (overbought)
   - If crossed back to exit band: **NEUTRAL**
4. Cooldown prevents whipsaw trades

**Key Parameters**:
```python
@dataclass
class StratParams:
    trend_kind: str = "sma"        # "sma" or "roc"
    trend_lookback: int = 200      # Bars for trend calculation
    flip_band_entry: float = 0.025 # Entry threshold (2.5%)
    flip_band_exit: float = 0.015  # Exit threshold (1.5%)
    vol_window: int = 60           # Volatility calculation window
    vol_adapt_k: float = 0.0       # Volatility scaling factor
    cooldown_minutes: int = 60     # Min time between state flips
    step_allocation: float = 0.33  # Position step size
    max_position: float = 1.0      # Max exposure (100%)
    long_only: bool = True         # Disable shorting
```

---

### 3.2 Trend Following (`TrendStrategy`)

**File**: `core/trend_strategy.py`

**Logic**:
1. Calculate fast/slow EMAs (or SMAs)
2. Signal:
   - Fast > Slow: **LONG** (Golden Cross)
   - Fast < Slow: **SHORT** (Death Cross)
3. Cooldown + Funding filter for entries

**Key Parameters**:
```python
@dataclass
class TrendParams:
    fast_period: int = 50          # Fast EMA (e.g., 50 bars)
    slow_period: int = 200         # Slow EMA (e.g., 200 bars)
    ma_type: str = "ema"           # "ema" or "sma"
    cooldown_minutes: int = 60     # Flip cooldown
    step_allocation: float = 1.0   # Trend goes "all in"
    long_only: bool = True
    position_sizing_mode: str = "volatility"
```

---

### 3.3 Meta Strategy (`MetaStrategy`)

**File**: `core/meta_strategy.py`

**Logic**:
1. Generate signals from BOTH MR and Trend strategies
2. Calculate **Regime Score** (Multi-Timeframe ADX)
3. **Hysteresis Logic** (prevents churn):
   - Switch to TREND if `score > threshold + buffer`
   - Switch to MR if `score < threshold - buffer`
4. Output = MR signal OR Trend signal based on regime

**Regime Detection** (`core/regime.py`):
```python
# Weighted ADX from multiple timeframes
trend_score = 0.2 * ADX(15m) + 0.3 * ADX(30m) + 0.5 * ADX(1h)
```

---

## 4. Configuration Schema

**File**: `core/config_schema.py`

All configs are validated via Pydantic. The structure is:

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
    "strategy_type": "meta",     // "mean_reversion" | "trend" | "meta"
    "adx_threshold": 25.0,       // Regime switch level
    "mean_reversion_overrides": {...},
    "trend_overrides": {...}
  },
  "execution": {
    "exchange_type": "futures",  // "spot" | "futures"
    "leverage": 1,
    "interval": "15m",
    "poll_sec": 5
  },
  "risk": {
    "basis_btc": 1.0,
    "risk_mode": "dynamic",      // "fixed_basis" | "dynamic"
    "max_dd_frac": 0.2,          // 20% max drawdown
    "drawdown_reset_days": 7.0,  // Phoenix wait time
    "drawdown_reset_score": 30.0 // Phoenix ADX threshold
  }
}
```

---

## 5. Live Execution Flow

**File**: `live_executor.py` (~1378 lines)

### Main Loop:

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Wait for bar close (15m interval by default)                 │
│ 2. Fetch klines from Binance (600 bars)                         │
│ 3. Fetch balance (Spot: assets, Futures: margin + position)     │
│ 4. Calculate current weight (cur_w)                             │
│ 5. Update risk state (HWM, drawdown, daily loss)               │
│ 6. Check safety gates (funding rate, trend gate)               │
│ 7. Generate strategy signal (target_w)                          │
│ 8. Phoenix Protocol check (auto-recovery from crash)           │
│ 9. Calculate delta (target_w - cur_w)                           │
│ 10. Execute trade if delta > threshold                          │
│ 11. Log to StoryWriter, update Prometheus metrics              │
│ 12. Save state to disk                                          │
└─────────────────────────────────────────────────────────────────┘
```

### Exchange Adapters:

| Mode | Adapter | Client | Key Differences |
|------|---------|--------|-----------------|
| Spot | `BinanceSpotAdapter` | `binance.spot.Spot` | Hold actual assets, LIMIT_MAKER orders |
| Futures | `BinanceFuturesAdapter` | `binance.um_futures.UMFutures` | Margin-based, GTX (post-only), `get_position()` |

---

## 6. Position Sizing

**File**: `core/position_sizer.py`

Three modes:

| Mode | Formula | Use Case |
|------|---------|----------|
| `static` | `step = base_step` | Fixed allocation |
| `volatility` | `step = base * (target_vol / realized_vol)` | Reduce size in volatile markets |
| `kelly` | Half-Kelly with vol adjustment | Optimal growth (risky) |

---

## 7. Risk Management

### Safety Breakers:

| Breaker | Trigger | Effect |
|---------|---------|--------|
| **Max Drawdown** | `wealth < HWM * (1 - max_dd_frac)` | Halt all trading |
| **Daily Loss Limit** | `daily_pnl < -threshold` | Halt until next day (UTC) |
| **Funding Gate** | `funding_rate > limit` | Block new entries in that direction |
| **Trend Gate** | `abs(60d_roc) < threshold` | Block trades in weak trend |

### Phoenix Protocol (Auto-Recovery):

Conditions to resume after Max DD:
1. Wait `drawdown_reset_days` (e.g., 7 days)
2. Regime score must indicate favorable conditions

---

## 8. Optimization Workflow

### Full Pipeline (run_complete_optimization.sh):

```
┌────────────────────────────────────────────────────────────┐
│ Step 1: Optimize Mean Reversion (optimizer_cli.py)         │
│         → Tests combinations of trend_kind, sizing_mode    │
│         → Output: results/opt_mr_*.csv                     │
├────────────────────────────────────────────────────────────┤
│ Step 2: Select Best MR (wf_pick.py)                        │
│         → Cluster by "family" key, rank by robustness      │
│         → Output: configs/best_mr_params_*.json            │
├────────────────────────────────────────────────────────────┤
│ Step 3: Optimize Trend (optimize_trend.py)                 │
│         → Optional: Walk-Forward mode (--wfo)              │
│         → Output: results/opt_trend_*.csv                  │
├────────────────────────────────────────────────────────────┤
│ Step 4: Select Best Trend                                  │
│         → WFO: wfo_select_best.py                          │
│         → Static: wf_pick.py                               │
├────────────────────────────────────────────────────────────┤
│ Step 5: Optimize Meta Threshold (optimize_meta.py)         │
│         → Find best adx_threshold for regime switching     │
├────────────────────────────────────────────────────────────┤
│ Step 6: Assemble Final Config (assemble_v2_config.py)      │
│         → Merge MR + Trend + Meta into single config       │
│         → Output: configs/meta_optimized_v2_*.json         │
└────────────────────────────────────────────────────────────┘
```

### Key CLI Flags:

```bash
# Exhaustive mode (all 8 combinations)
./run_complete_optimization.sh --exhaustive

# Walk-Forward Optimization
./run_complete_optimization.sh --wfo

# Custom date range
./run_complete_optimization.sh --train-start 2022-01-01 --test-end 2024-12-01
```

---

## 9. Observability Stack

### Prometheus Metrics (`core/metrics.py`):

| Metric | Description |
|--------|-------------|
| `wealth_total` | Total portfolio value in quote asset |
| `exposure_base_weight` | Current position weight (0-1) |
| `regime_score` | ADX-based trend score (0-100) |
| `strategy_mode` | 0=MR, 1=Trend |
| `phoenix_active` | 1=Halted, 0=Trading |
| `funding_rate_pct` | Current funding rate |
| `trade_decision` | Last decision (exec_buy, skip_*, etc.) |

### Story Writer (`core/story_writer.py`):

Human-readable log of key events:
- 🚀 Startup
- 🟢 BUY / 🔴 SELL executions  
- 🔄 Regime switches
- 📈 New ATH
- 🚨 Safety breaker trips
- 🔥 Phoenix activations
- 📊 Daily/Weekly/Monthly summaries

### Grafana Dashboard:

- **Row 1 (HUD)**: Wealth, Session PnL, Risk Status, Gate, Exposure
- **Row 2**: Blocker Timeline (decision history)
- **Row 3**: Signal vs Bands, Proximity to Trade
- **Row 4**: Weight Tracking, Skip Reasons, Latency

---

## 10. Deployment

### Docker Compose:

```yaml
services:
  bot_btc:
    build: .
    environment:
      - SYMBOL=BTCUSDT
      - MODE=live
    volumes:
      - ./run_state/btc:/data
    command: python /app/live_executor.py --params configs/prod_btc_meta_live.json --mode live
    ports:
      - "9100:9109"  # Metrics
      - "9110:9110"  # Status JSON
```

### Environment Variables:

| Variable | Description |
|----------|-------------|
| `BINANCE_KEY` | API Key |
| `BINANCE_SECRET` | API Secret |
| `BINANCE_FUTURES_KEY` | Futures-specific key (optional) |
| `MODE` | `dry` / `testnet` / `live` |
| `LOGLEVEL` | `DEBUG` / `INFO` / `WARNING` |

---

## 11. Key Files Quick Reference

| Need to... | Look at... |
|------------|------------|
| Understand signal generation | `core/ethbtc_accum_bot.py:EthBtcStrategy.generate_positions` |
| Debug live trading | `live_executor.py:main()` (~line 230+) |
| Add new config parameter | `core/config_schema.py` |
| Modify regime detection | `core/regime.py:get_regime_score` |
| Add new Prometheus metric | `core/metrics.py` |
| Change optimization scoring | `tools/optimizer_cli.py:Objective.__call__` |
| Modify position sizing | `core/position_sizer.py` |
| Add new alerts | `core/story_writer.py` + `core/alert_manager.py` |

---

## 12. Common Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Bot not trading | Check `trade_decision` metric - likely `skip_threshold` | Normal if bands not hit |
| Weight lines diverging | `skip_balance` or `skip_min_notional` | Check balance, min_trade_btc |
| Strategy crashes | `index must be monotonic` error | Check `core/regime.py` resampling |
| Solver returns unknown | Constraint infeasibility | Check optimization constraints |
| Futures position wrong | `get_position()` fails | Check API keys, testnet vs live |

---

## 13. Version History & Known Issues

- **v5.3**: Current production version with Meta Strategy, Futures support
- **Known**: Optimizer can be slow in exhaustive mode (8 parallel jobs)
- **Known**: Phoenix Protocol requires `drawdown_reset_days > 0` to activate

---

*Last Updated: December 2024*
