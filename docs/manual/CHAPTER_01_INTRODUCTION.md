# Chapter 1: Introduction & System Overview

> **Purpose:** This chapter provides a complete understanding of what the ETH/BTC Algorithmic Trading Bot is, why it exists, the guiding principles behind its design, and the realistic performance targets it aims to achieve.

---

## 1.1 Project Philosophy

### 1. Concept & "The Why"

* **What it is:** The ETH/BTC Algorithmic Trading Bot is a production-grade, event-driven trading system designed for Binance Spot and Futures markets. It implements three core strategies—Mean Reversion, Trend Following, and Meta (ensemble)—with comprehensive risk management, automated execution, and real-time observability.

* **Purpose:** This system exists to systematically trade cryptocurrency markets with a philosophy that prioritizes **not losing money** over maximizing gains. Most retail trading bots fail due to:
  1. Over-optimization (curve fitting to historical data)
  2. Lack of risk controls (catastrophic losses during drawdowns)
  3. Complexity creep (too many features, too many failure modes)
  
  This bot addresses each by enforcing capital preservation above all else, using Walk-Forward Optimization to prevent overfitting, and maintaining a deliberately simple architecture.

* **Location:** The core philosophy is documented in [`docs/Theory.md`](../Theory.md), with implementation spread across the `core/` module.

### 2. The Four Pillars

The entire system is built on four non-negotiable principles:

| Pillar | Description | Implementation |
|--------|-------------|----------------|
| **Capital Preservation** | Surviving drawdowns is more important than maximizing returns | Phoenix Protocol, Max Drawdown Limits, Daily Loss Limits |
| **Simplicity** | Fewer moving parts means fewer failure modes | Single-threaded architecture, rule-based signals, JSON configs |
| **Adaptability** | Markets change; the system must adapt without manual intervention | ADX-based regime detection, Dynamic Kelly, Walk-Forward Optimization |
| **Operational Reliability** | Uptime and monitoring matter more than microsecond latency | Docker deployment, Prometheus metrics, Circuit Breaker pattern |

### 3. Configuration & Parameters

The project philosophy manifests in these key configuration choices:

| Parameter | Location | Value | Rationale |
|-----------|----------|-------|-----------|
| `risk.max_dd_frac` | `configs/*.json` | `0.15` – `0.20` | Hard stop to prevent catastrophic loss |
| `risk.drawdown_reset_days` | `configs/*.json` | `7.0` | Cooling-off period before resuming after halt |
| `strategy.long_only` | `configs/*.json` | `true` (default) | Simplifies risk profile for most users |
| `execution.interval` | `configs/*.json` | `15m` | Sweet spot between noise and opportunity |

**Hidden Logic:**
- The system is **hardcoded** to use a single-threaded execution model (`live_executor.py` line 127: `STOP_EVENT = threading.Event()`). Multi-threading is only used for the HTTP status server and maker order threads.
- Target latency is intentionally relaxed to **< 5 seconds** from signal to order, because alpha comes from signal quality, not execution speed.

### 4. Real-World Use Case (The "Cookbook")

**Scenario:** New user wants to understand if this bot fits their trading philosophy.

**Checklist:**
1. ✅ You want systematic, rules-based trading (not discretionary).
2. ✅ You can accept 20-50% annual returns and are not chasing 10x gains.
3. ✅ You prioritize not losing over winning big.
4. ✅ You have 3+ years of historical data for backtesting.
5. ✅ You can run Docker containers 24/7 (VPS, NAS, or home server).

**If all boxes are checked:** This bot is designed for you.

**If you want:**
- Sub-second execution: ❌ Look at specialized HFT infrastructure.
- ML/AI-driven predictions: ❌ This bot uses interpretable, rule-based signals.
- Multi-exchange arbitrage: ❌ Out of scope.

### 5. Troubleshooting & Edge Cases

* **What can go wrong:** Misalignment between user expectations and system design.
* **Error Messages:** None—this is a philosophy section, not executable code.
* **Edge Case:** Users trying to maximize returns by setting `max_dd_frac: 0.50` (50% drawdown tolerance) are violating the core philosophy. The system will allow this, but it defeats the purpose of capital preservation.

---

## 1.2 Target Performance Metrics

### 1. Concept & "The Why"

* **What it is:** A set of realistic, measurable performance targets derived from quantitative finance literature and practical trading experience. These are **not** marketing promises—they are calibrated expectations for what a well-run systematic strategy can achieve.

* **Purpose:** To set realistic expectations and provide benchmarks for evaluating whether the system is performing as designed. A common retail trader mistake is expecting 100%+ annual returns, which leads to over-leveraging and ruin.

* **Location:** Defined in [`docs/Theory.md`](../Theory.md), Part 1, Section II. Calculated in backtest reports via [`core/backtest_report.py`](../core/backtest_report.py).

### 2. Configuration & Parameters

| Metric | Target Range | Calculation Location | Notes |
|--------|--------------|---------------------|-------|
| **Sharpe Ratio** | 1.0 – 2.0 | `BacktestReport._calculate_sharpe()` | Risk-adjusted return. > 1.5 is excellent. |
| **Annual Return** | 20% – 50% | `BacktestReport.from_backtest_result()` | Regime-dependent. Bull markets = higher. |
| **Maximum Drawdown** | < 15% (hard stop at 20%) | `BacktestReport._calculate_max_drawdown()` | Enforced by Phoenix Protocol. |
| **Win Rate** | > 45% | `BacktestReport._calculate_trade_stats()` | Must have positive expectancy. |
| **Recovery Factor** | > 2.0 | `total_return / max_drawdown` | Measures ability to recover from losses. |
| **Maker Fill Rate** | > 80% | Prometheus metric `trade_decision` | Limit orders preserve alpha. |

**Defaults:** These targets are not configurable—they are design goals. The configuration parameters in `risk` section (e.g., `max_dd_frac: 0.15`) exist to enforce these targets.

**Hidden Logic:**
- Sharpe Ratio assumes risk-free rate of **0%** (standard for crypto).
- Annualization uses **365 trading days** (crypto markets are 24/7/365).
- Win rate alone is meaningless—the system also tracks `profit_factor` (sum of wins / sum of losses).

### 3. Step-by-Step Guide: Measuring Your Performance

1. **Run a backtest:**
   ```bash
   python core/ethbtc_accum_bot.py backtest \
     --data data/raw/BTCUSDT_15m_2021-2025_vision.csv \
     --funding data/raw/BTCUSDT_funding_2021-2025.csv \
     --params configs/prod_btc_meta_live.json
   ```

2. **Review the printed report:**
   ```
   ═══════════════════════════════════════════════════════
     ETHBTC Strategy Backtest Report
   ═══════════════════════════════════════════════════════
   
   📈 PERFORMANCE
   ─────────────────────────────────────────────────────
   Total Return:        +156.32%
   Annualized Return:   +28.4%
   Sharpe Ratio:        1.42
   
   📉 RISK
   ─────────────────────────────────────────────────────
   Max Drawdown:        -12.8%
   Recovery Factor:     2.31
   VaR (95%):           -2.1%
   CVaR (95%):          -3.4%
   ```

3. **Compare against targets:**
   - Sharpe 1.42 → ✅ Within 1.0–2.0 range
   - Max DD -12.8% → ✅ Below 15% target
   - Recovery Factor 2.31 → ✅ Above 2.0

### 4. Real-World Use Case (The "Cookbook")

**Scenario:** Trader wants to evaluate if the optimized BTC configuration meets targets.

**Configuration:**
```json
{
  "fees": { "maker_fee": 0.0002, "taker_fee": 0.0004 },
  "strategy": { "strategy_type": "meta", "adx_threshold": 10.0 },
  "execution": { "interval": "15m", "exchange_type": "futures" },
  "risk": { "max_dd_frac": 0.15, "basis_btc": 0.1 }
}
```

**Command:**
```bash
python core/ethbtc_accum_bot.py backtest \
  --data data/raw/BTCUSDT_15m_2021-2025_vision.csv \
  --params configs/prod_btc_meta_live.json \
  --report results/backtest_report.md
```

**Expected Outcome:**
- Markdown report saved to `results/backtest_report.md`
- Console output showing Sharpe, Max DD, Win Rate
- HODL comparison (strategy vs buy-and-hold)

### 5. Troubleshooting & Edge Cases

* **What can go wrong:**
  - **Sharpe < 1.0:** Strategy may be over-trading or parameters are poorly tuned. Run Walk-Forward Optimization.
  - **Max DD > 20%:** Risk parameters may be too aggressive. Reduce `step_allocation` or increase `max_dd_frac` threshold.
  - **Win Rate < 40%:** Normal for trend-following strategies (they win less often but win bigger). Check profit factor instead.

* **Error Messages:**
  ```
  ValueError: Cannot calculate Sharpe ratio with zero standard deviation
  ```
  **Cause:** The strategy had no trades or all trades had identical returns.
  **Fix:** Increase historical data range or adjust signal thresholds.

* **Edge Case:** During extreme sideways markets (2023 Q2), even well-tuned strategies may underperform. The Meta strategy mitigates this by switching between Mean Reversion and Trend based on ADX.

---

## 1.3 Architecture Overview

### 1. Concept & "The Why"

* **What it is:** A modular, event-driven Python application with four distinct layers: **Signal Generation → Position Sizing → Execution → Monitoring**. Each layer is independently testable and replaceable.

* **Purpose:** Separation of concerns enables:
  1. Independent testing of each component
  2. Swapping strategies without changing execution logic
  3. Adding new exchange adapters without modifying signal generation
  4. Debugging issues by isolating the problematic layer

* **Location:** 
  - Entry point: [`live_executor.py`](../live_executor.py) (1,373 lines)
  - Core modules: [`core/`](../core/) directory (26 Python files)
  - Configuration: [`core/config_schema.py`](../core/config_schema.py)

### 2. Configuration & Parameters

**System Architecture Diagram:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LIVE EXECUTOR                                   │
│                         (live_executor.py:main)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. DATA LAYER               │  2. SIGNAL LAYER           │  3. SIZING      │
├──────────────────────────────┼────────────────────────────┼─────────────────┤
│  • BinanceSpotAdapter        │  • EthBtcStrategy (MR)     │  • PositionSizer│
│  • BinanceFuturesAdapter     │  • TrendStrategy           │    - static     │
│  • DataService               │  • MetaStrategy (ensemble) │    - volatility │
│  • get_klines()              │  • generate_positions()    │    - kelly      │
└──────────────────────────────┴────────────────────────────┴─────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. EXECUTION LAYER          │  5. RISK LAYER             │  6. MONITORING  │
├──────────────────────────────┼────────────────────────────┼─────────────────┤
│  • place_limit_maker()       │  • RiskManager             │  • Prometheus   │
│  • market_order()            │  • Phoenix Protocol        │  • StoryWriter  │
│  • wait_for_fill()           │  • HWM Tracking            │  • AlertManager │
│  • CircuitBreaker            │  • Daily Loss Limit        │  • /status HTTP │
└──────────────────────────────┴────────────────────────────┴─────────────────┘
```

**Key Files by Layer:**

| Layer | Files | Responsibility |
|-------|-------|----------------|
| Entry Point | `live_executor.py` | Main loop, CLI args, state management |
| Data | `core/binance_adapter.py`, `core/futures_adapter.py` | Exchange API interaction |
| Signal | `core/ethbtc_accum_bot.py`, `core/trend_strategy.py`, `core/meta_strategy.py` | Trading signal generation |
| Sizing | `core/position_sizer.py` | Dynamic position sizing |
| Risk | `core/risk_manager.py` | Drawdown tracking, Phoenix Protocol |
| Execution | `core/order_manager.py`, `core/precision.py` | Order placement, fill confirmation |
| Resilience | `core/resilience.py` | Circuit breaker, retry logic |
| Monitoring | `core/metrics.py`, `core/story_writer.py`, `core/alert_manager.py` | Observability |

### 3. Step-by-Step Guide: Tracing a Trade

Follow a single trade from signal to execution:

1. **Bar Close Detected** (`live_executor.py` line ~300):
   ```python
   bar_ts = last_closed_bar_ts(now_s, interval)
   ```
   The system detects a new 15-minute bar has closed.

2. **Fetch OHLCV Data** (`core/binance_adapter.py`):
   ```python
   klines = adapter.get_klines(symbol, interval, limit=500)
   ```
   Retrieves last 500 bars from Binance.

3. **Generate Signal** (`core/meta_strategy.py`):
   ```python
   result = strategy.generate_positions(df, funding)
   target_w = result["target_w"].iloc[-1]  # e.g., 0.5 (50% long)
   ```
   Meta strategy calculates regime score and blends MR/Trend signals.

4. **Calculate Position Delta**:
   ```python
   current_w = base_value / wealth  # Current position weight
   delta_w = target_w - current_w   # e.g., 0.5 - 0.2 = 0.3
   ```

5. **Apply Position Sizing** (`core/position_sizer.py`):
   ```python
   step = sizer.calculate_step(realized_vol)
   delta_w = min(delta_w, step)  # Clamp to step size
   ```

6. **Risk Check** (`core/risk_manager.py`):
   ```python
   if risk_mgr.is_halted(state):
       return  # Skip trade if max DD or daily limit hit
   ```

7. **Execute Order** (`core/binance_adapter.py`):
   ```python
   order_id = adapter.place_limit_maker(symbol, "BUY", quantity, price)
   ```

8. **Wait for Fill**:
   ```python
   filled, qty = adapter.check_order(symbol, order_id)
   ```

9. **Update State & Metrics**:
   ```python
   save_state(state_file, state)
   WEALTH_GAUGE.labels(instance).set(wealth)
   ```

### 4. Real-World Use Case (The "Cookbook")

**Scenario:** Developer wants to add a new exchange adapter (e.g., Kraken).

**Step 1:** Create `core/kraken_adapter.py` implementing `ExchangeAdapter` interface:
```python
from core.exchange_adapter import ExchangeAdapter, Book, Filters

class KrakenAdapter(ExchangeAdapter):
    def get_klines(self, symbol: str, interval: str, limit: int = 500):
        # Implement Kraken API call
        pass
    
    def place_limit_maker(self, symbol: str, side: str, quantity: float, price: float):
        # Implement Kraken order placement
        pass
```

**Step 2:** Register in `live_executor.py`:
```python
elif cfg.execution.exchange_type == "kraken":
    from core.kraken_adapter import KrakenAdapter
    adapter = KrakenAdapter(client)
```

**Step 3:** No changes needed to signal generation, sizing, or risk layers.

**Expected Outcome:** New exchange works with all existing strategies.

### 5. Troubleshooting & Edge Cases

* **What can go wrong:**
  - **Circular Import:** Adding a new module that imports from `live_executor.py` can cause circular imports. Always import from `core/` modules, never from the entry point.
  - **State Corruption:** If `save_state()` is interrupted mid-write, the state file may be corrupted. The system uses atomic writes (write to temp, then rename).

* **Error Messages:**
  ```
  ImportError: cannot import name 'merge_strategy_params' from 'live_executor'
  ```
  **Cause:** Attempting to import from entry point instead of `core/strategy_factory`.
  **Fix:** Import from the correct module: `from core.strategy_factory import merge_strategy_params`

* **Edge Case:** The single-threaded architecture means a blocked API call will pause the entire system. This is mitigated by:
  - Circuit breaker (`core/resilience.py`) opens after 5 consecutive failures
  - All API calls have timeouts (default 5 seconds)
  - Auto-restart via Docker `restart: unless-stopped`

---

## 1.4 Non-Goals (Explicitly Out of Scope)

### 1. Concept & "The Why"

* **What it is:** An explicit list of features and capabilities that this system **does not** and **will not** support. This is not a "todo list for v2"—these are deliberate exclusions based on cost-benefit analysis.

* **Purpose:** 
  1. Set clear expectations for users evaluating the system
  2. Prevent scope creep during development
  3. Focus resources on what matters (capital preservation, reliability)
  
* **Location:** Documented in [`docs/Theory.md`](../Theory.md), Part 1, Section III.

### 2. The Exclusion List

| Feature | Why Excluded | Effort to Add | Benefit vs. Cost |
|---------|--------------|---------------|------------------|
| **High-Frequency Trading** | Requires co-location, FPGA, specialized infra | 6+ months | Low (not needed for swing trading) |
| **Cross-Exchange Arbitrage** | Legal complexity, capital fragmentation, latency requirements | 3+ months | Medium (risky for solo operators) |
| **Deep Learning Models** | Latency, cost, fragility, overfitting risk | 4+ months | Low (simple indicators perform comparably) |
| **Reinforcement Learning** | Multi-year research project, high overfit risk | 12+ months | Very Low (academic, not practical) |
| **Social Sentiment (Twitter/TikTok)** | API costs ($500+/mo), noise, unreliable signal | 2+ months | Very Low (noise > signal) |
| **L2/L3 Order Book Data** | $1000+/month data costs, marginal benefit | 1+ month | Very Low for swing trading |
| **Multi-Asset Portfolio** | Correlation tracking, rebalancing complexity | 2+ months | Medium (future consideration) |

### 3. Configuration & Parameters

There are **no configuration parameters** for excluded features. Attempting to enable them will have no effect.

**Hidden Logic:** The codebase contains no dead code for these features. They are architecturally excluded, not disabled.

### 4. Real-World Use Case (The "Cookbook")

**Scenario:** User wants to add Twitter sentiment analysis.

**Answer:** This is explicitly out of scope. The rationale:
1. Twitter/X API costs ~$100-$5000/month for firehose access
2. Sentiment signals are noisy and lag price action
3. NLP models require constant maintenance and fine-tuning
4. Simpler indicators (RSI, ADX) provide comparable signal quality at zero cost

**Alternative:** Use the existing `funding_counter_enabled` feature, which captures market sentiment via funding rate extremes (a leading, not lagging, indicator).

```json
{
  "strategy": {
    "funding_counter_enabled": true,
    "extreme_funding_long_threshold": 0.0005,
    "extreme_funding_short_threshold": -0.0005
  }
}
```

### 5. Troubleshooting & Edge Cases

* **What can go wrong:** Users expecting features that don't exist.
* **Error Messages:** None—the system simply doesn't have these features.
* **Edge Case:** External systems (e.g., a separate sentiment bot) can send signals via the configuration file. The bot will read the config on startup, but dynamic signal injection is not supported.

---

## Summary: Key Takeaways

| Aspect | This Bot Is | This Bot Is Not |
|--------|-------------|-----------------|
| **Trading Style** | Swing/Position trading (15m-1d) | High-frequency trading |
| **Signal Generation** | Rule-based, interpretable | ML/AI black box |
| **Risk Priority** | Capital preservation first | Maximum return chasing |
| **Architecture** | Simple, modular, testable | Complex, monolithic |
| **Target User** | Solo traders, small teams | Institutional desks |
| **Expected Returns** | 20-50% annually | 10x "moonshot" gains |
| **Drawdown Tolerance** | < 15% (halt at 20%) | Unlimited |

---

*Next Chapter: [Chapter 2: Installation & Deployment](./CHAPTER_02_INSTALLATION.md)*
