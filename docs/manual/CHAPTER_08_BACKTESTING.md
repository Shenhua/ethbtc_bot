# Chapter 8: Backtesting Engine

> **Purpose:** This chapter provides exhaustive documentation of the backtesting engine, covering the simulation loop, fee modeling, risk management, performance metrics, and report generation.

---

## 8.1 Backtester Architecture

### 1. Concept & "The Why"

* **What it is:** A vectorized backtesting engine that simulates trading strategy performance on historical data. Produces portfolio equity curves, trade logs, and comprehensive performance metrics.

* **Purpose:** 
  1. Validate strategies before risking real capital
  2. Optimize parameters via Walk-Forward Optimization
  3. Compare strategies objectively using standardized metrics
  4. Detect overfitting via out-of-sample testing

* **Location:** 
  - Main Backtester: [`core/ethbtc_accum_bot.py`](../../core/ethbtc_accum_bot.py) → `Backtester` class
  - Report Generator: [`core/backtest_report.py`](../../core/backtest_report.py) → `BacktestReport` class
  - CLI: [`core/ethbtc_accum_bot.py`](../../core/ethbtc_accum_bot.py) → `cmd_backtest()`

### 2. Configuration & Parameters

#### CLI Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--data` | ✅ | Path to OHLCV CSV file |
| `--config` | ✅ | Path to config JSON |
| `--funding-data` | ❌ | Path to funding rate CSV |
| `--bnb-data` | ❌ | Path to BNB price CSV (for fee calculation) |
| `--out` | ❌ | Path to save detailed output CSV |
| `--start` | ❌ | Start date (YYYY-MM-DD) |
| `--end` | ❌ | End date (YYYY-MM-DD) |
| `--basis-btc` | ❌ | Starting capital (overrides config) |
| `--base` | ❌ | Base asset name (default: auto-detected) |
| `--quote` | ❌ | Quote asset name (default: auto-detected) |
| `--story` | ❌ | Path to save story log file |
| `--report` | ❌ | Generate enhanced report |

### 3. Simulation Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    1. LOAD DATA & CONFIG                         │
│         Load CSV → Parse dates → Slice to start/end              │
│         Load JSON config → Build strategy via factory            │
└──────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                    2. GENERATE SIGNALS                           │
│       strategy.generate_positions(data) → target_w array         │
│       Pre-calculate: step[], threshold[], volatility[]           │
└──────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                    3. SIMULATION LOOP (per bar)                  │
│       For each bar i:                                            │
│         - Check risk limits (daily loss, max DD)                 │
│         - Apply position sizing (step calculation)               │
│         - Calculate trade delta                                  │
│         - Apply fees and slippage                                │
│         - Update balances (BTC, ETH, BNB)                        │
│         - Track equity curve                                     │
└──────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                    4. GENERATE RESULTS                           │
│       Portfolio DataFrame, Trades list, Summary dict             │
│       Optional: BacktestReport with 25+ metrics                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 8.2 Running a Backtest

### 1. Concept & "The Why"

* **What it is:** The command-line interface for executing backtests with custom parameters.

* **Purpose:** Enables rapid iteration on strategy parameters without code changes.

### 2. Step-by-Step Guide

1. **Basic backtest:**
   ```bash
   python core/ethbtc_accum_bot.py backtest \
     --data data/raw/ETHBTC_15m_2021-2025.csv \
     --config configs/prod_eth_long_wfo_robust.json
   ```

2. **With date range:**
   ```bash
   python core/ethbtc_accum_bot.py backtest \
     --data data/raw/BTCUSDT_15m_2021-2025_vision.csv \
     --config configs/prod_btc_meta_live.json \
     --start 2023-01-01 \
     --end 2024-12-31
   ```

3. **With enhanced report:**
   ```bash
   python core/ethbtc_accum_bot.py backtest \
     --data data/raw/BTCUSDT_15m_2021-2025_vision.csv \
     --config configs/prod_btc_meta_live.json \
     --report
   ```
   Output saved to: `results/backtest_report_BTCUSDT_YYYYMMDD_HHMMSS.md`

4. **With funding rates and BNB fees:**
   ```bash
   python core/ethbtc_accum_bot.py backtest \
     --data data/raw/BTCUSDT_15m_2021-2025_vision.csv \
     --config configs/prod_btc_meta_live.json \
     --funding-data data/raw/BTCUSDT_funding.csv \
     --bnb-data data/raw/BNBUSDT_15m.csv \
     --basis-btc 1.0 \
     --report
   ```

### 3. Expected Output

```json
{
  "initial_btc": 1.0,
  "final_btc": 1.4523,
  "total_return": 0.4523,
  "max_drawdown_pct": 0.1234,
  "fees_btc": 0.0045,
  "turnover_btc": 2.25,
  "n_trades": 156,
  "n_bars": 140160
}
```

---

## 8.3 Fee Modeling

### 1. Concept & "The Why"

* **What it is:** Realistic fee simulation including maker/taker fees, slippage, and BNB discount.

* **Purpose:** Accurate fee modeling is critical—strategies that look profitable without fees often lose money after fees.

* **Location:** [`core/ethbtc_accum_bot.py`](../../core/ethbtc_accum_bot.py) → `FeeParams` dataclass

### 2. Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `maker_fee` | float | 0.0002 | Maker fee (0.02%) |
| `taker_fee` | float | 0.0004 | Taker fee (0.04%) |
| `slippage_bps` | float | 0.0 | Slippage in basis points |
| `bnb_discount` | float | 0.25 | BNB fee discount (25% off) |
| `pay_fees_in_bnb` | bool | false | Use BNB for fees |

### 3. Fee Calculation

```python
# Fee calculation per trade
fee_discount = (1 - bnb_discount) if pay_fees_in_bnb else 1.0
effective_fee_rate = taker_fee * fee_discount

notional = abs(delta) * price
fee = notional * effective_fee_rate

# Slippage modeling (penalizes fill price)
slippage_mult = slippage_bps / 10000.0

if delta > 0:  # BUY
    fill_price = price * (1 + slippage_mult)  # Pay more
else:          # SELL
    fill_price = price * (1 - slippage_mult)  # Receive less
```

**Example Calculation:**
```
Trade: BUY 10 ETH @ 0.034 BTC
Notional: 10 × 0.034 = 0.34 BTC
Taker Fee: 0.34 × 0.04% = 0.000136 BTC
Slippage (1 bps): 0.034 × 0.0001 = 0.0000034 BTC per ETH
Total Cost Impact: 0.000136 + (10 × 0.0000034) = 0.000170 BTC
```

### 4. Real-World Use Case

**Scenario:** Conservative backtest with realistic fee assumptions.

**Configuration:**
```json
{
  "fees": {
    "maker_fee": 0.0002,
    "taker_fee": 0.0005,
    "slippage_bps": 2.0,
    "bnb_discount": 0.0,
    "pay_fees_in_bnb": false
  }
}
```

**Why:**
- Uses higher taker fee (realistic for non-maker fills)
- 2 bps slippage accounts for market impact
- No BNB discount (conservative assumption)

---

## 8.4 Risk Simulation

### 1. Concept & "The Why"

* **What it is:** Full simulation of risk management rules within the backtest, mirroring live behavior.

* **Purpose:** Ensures backtest results reflect real trading constraints.

### 2. Simulated Risk Features

| Feature | Description |
|---------|-------------|
| **Daily Loss Limit** | When daily loss exceeds threshold, skip trading until next UTC day |
| **Max Drawdown** | When DD exceeds threshold, halt trading (freeze `maxdd_hit = True`) |
| **Phoenix Protocol** | After `drawdown_reset_days` + `reset_score` threshold, reset and resume |
| **HWM Tracking** | Track peak equity, freeze during halt |

### 3. Risk Simulation Code

```python
# In simulate() loop

# Daily Loss Check
if timestamp.date() != current_day:
    current_day = timestamp.date()
    day_start_wealth = wealth
    daily_limit_hit = False

if not daily_limit_hit and max_daily_loss_btc > 0:
    day_loss = day_start_wealth - wealth
    if day_loss >= max_daily_loss_btc:
        daily_limit_hit = True

# Max Drawdown Check
if not maxdd_hit:
    if wealth > equity_high:
        equity_high = wealth

if not maxdd_hit and max_dd_frac > 0:
    dd = (equity_high - wealth) / equity_high
    if dd >= max_dd_frac:
        maxdd_hit = True
        maxdd_hit_ts = timestamp

# Phoenix Reset Check
if maxdd_hit and drawdown_reset_days > 0:
    time_passed = timestamp - maxdd_hit_ts
    current_score = plan["regime_score"].iat[i]
    
    if time_passed.days >= drawdown_reset_days and current_score >= drawdown_reset_score:
        maxdd_hit = False
        equity_high = wealth
```

---

## 8.5 Performance Metrics

### 1. Concept & "The Why"

* **What it is:** 25+ standardized performance metrics calculated from backtest results.

* **Purpose:** Enables objective strategy comparison and risk assessment.

* **Location:** [`core/backtest_report.py`](../../core/backtest_report.py) → `BacktestReport`

### 2. Available Metrics

#### Performance Metrics
| Metric | Formula | Target |
|--------|---------|--------|
| **Total Return** | `(final / initial) - 1` | > 0 |
| **CAGR** | `(final / initial)^(1/years) - 1` | > 15% |
| **Alpha** | `strategy_return - hodl_return` | > 0 |

#### Risk-Adjusted Metrics
| Metric | Formula | Target |
|--------|---------|--------|
| **Sharpe Ratio** | `mean(excess_return) / std(return) × √252` | > 1.0 |
| **Sortino Ratio** | `mean(excess_return) / downside_std × √252` | > 1.5 |
| **Calmar Ratio** | `CAGR / max_drawdown` | > 1.0 |
| **VaR (95%)** | 5th percentile of daily returns | < 3% |
| **CVaR (95%)** | Mean of returns below VaR | < 5% |

#### Drawdown Metrics
| Metric | Description |
|--------|-------------|
| **Max Drawdown** | Largest peak-to-trough decline |
| **Max DD Duration** | Longest time underwater |
| **Avg Drawdown** | Average of all underwater periods |
| **Volatility (Ann.)** | Annualized std of daily returns |

#### Trading Metrics
| Metric | Description | Target |
|--------|-------------|--------|
| **Win Rate** | % of profitable trades | > 45% |
| **Profit Factor** | Gross profit / gross loss | > 1.5 |
| **Avg Holding** | Average bars between trades | — |
| **Time in Market** | % of bars with exposure | — |
| **Total Fees** | Sum of all fees paid | Low |

### 3. Generating a Report

```bash
python core/ethbtc_accum_bot.py backtest \
  --data data/raw/BTCUSDT_15m_2021-2025_vision.csv \
  --config configs/prod_btc_meta_live.json \
  --report
```

**Terminal Output:**
```
═══════════════════════════════════════════════════════════════════════════
                    BACKTEST REPORT: BTCUSDT METASTRATEGY
                    Period: 2021-01-01 → 2025-01-01 (1461 days)
═══════════════════════════════════════════════════════════════════════════

💰 PERFORMANCE
   Initial Capital:        1.0000
   Final Capital:          2.2534
   Total Return:           +125.34%  (CAGR: +22.1%)

📊 BENCHMARK COMPARISON
   HODL Quote (USDT):      +0.00%  (Your baseline)
   HODL BTC:               +356.12%
   ✨ ALPHA vs HODL:       -230.78%  ← YOUR VALUE ADD

📉 RISK METRICS
   Sharpe Ratio:           1.45
   Sortino Ratio:          2.12
   Calmar Ratio:           1.89
   Max Drawdown:           11.7% (Duration: 45 days)
   Volatility (Ann.):      32.4%

📈 TRADING ANALYSIS
   Total Trades:           312
   Win Rate:               52.3%
   Profit Factor:          1.67
   Time in Market:         78.4%
```

### 4. Markdown Report Output

```markdown
# Backtest Report: BTCUSDT MetaStrategy

**Period:** 2021-01-01 → 2025-01-01 (1461 days)

## 💰 Performance Summary

| Metric | Value |
|--------|------:|
| Initial Capital | 1.0000 |
| Final Capital | 2.2534 |
| **Total Return** | **+125.34%** |
| CAGR | +22.10% |

## 📉 Risk Metrics

| Metric | Value | Assessment |
|--------|------:|------------|
| Sharpe Ratio | 1.45 | 🟢 Good |
| Max Drawdown | 11.7% | 🟢 Low |

### Monthly Returns (%)

| Year | Jan | Feb | Mar | ... | Total |
|------|----:|----:|----:|-----|------:|
| 2021 | +2.1 | -0.5 | +3.2 | ... | **+24.5** |
```

---

## 8.6 Data Requirements

### 1. OHLCV Data Format

The backtester expects CSV files with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `close_time` | int/datetime | Bar close timestamp |
| `open` | float | Open price |
| `high` | float | High price |
| `low` | float | Low price |
| `close` | float | Close price |
| `volume` | float | Trading volume |

**Example CSV:**
```csv
close_time,open,high,low,close,volume
1609459200000,0.0345,0.0348,0.0342,0.0347,1234.5
1609460100000,0.0347,0.0350,0.0346,0.0349,2345.6
```

### 2. Funding Rate Data Format

| Column | Type | Description |
|--------|------|-------------|
| `time` | datetime | Funding timestamp |
| `rate` | float | Funding rate (as decimal, e.g., 0.0001) |

### 3. Symbol Detection

The backtester auto-detects base/quote from filename:

```
BTCUSDT_15m_2021-2025.csv → BTC/USDT
ETHBTC_15m_data.csv → ETH/BTC
```

**Override with flags:**
```bash
--base ETH --quote BTC
```

---

## 8.7 Detailed Output

### 1. Portfolio DataFrame

| Column | Description |
|--------|-------------|
| `wealth_btc` | Total portfolio value in quote asset |

### 2. Trades DataFrame

| Column | Description |
|--------|-------------|
| `time` | Trade timestamp |
| `side` | BUY or SELL |
| `price` | Execution price |
| `qty` | Trade quantity |
| `fee` | Fee paid |

### 3. Diagnostics DataFrame

| Column | Description |
|--------|-------------|
| `target_w` | Target weight |
| `regime_score` | ADX-based score (Meta only) |
| `regime_state` | -1 (MR) or +1 (Trend) |
| `sig_mr` | Mean Reversion signal |
| `sig_trend` | Trend signal |

**Save detailed output:**
```bash
python core/ethbtc_accum_bot.py backtest \
  --data data/raw/BTCUSDT_15m.csv \
  --config configs/my_config.json \
  --out results/detailed_backtest.csv
```

---

## 8.8 Troubleshooting

### Common Errors

```
ValueError: close_time column not found
```
**Cause:** CSV format doesn't match expected format.
**Fix:** Ensure CSV has `close_time` column or use Vision format.

```
KeyError: 'close'
```
**Cause:** OHLC columns missing.
**Fix:** Verify CSV has open, high, low, close columns.

```
ValueError: Need OHLC
```
**Cause:** Strategy requires full OHLC but only close was passed.
**Fix:** Pass `full_df=df` parameter with complete OHLC data.

### Performance Tips

1. **Use date slicing for faster iteration:**
   ```bash
   --start 2024-01-01 --end 2024-06-01
   ```

2. **Pre-process data once:**
   - Convert CSV to Parquet for faster loading
   - Cache signal generation for repeated runs

3. **Enable DEBUG logging for issue investigation:**
   ```bash
   export LOGLEVEL=DEBUG
   ```

---

*Previous Chapter: [Chapter 7: Execution Layer](./CHAPTER_07_EXECUTION.md)*  
*Next Chapter: [Chapter 9: Walk-Forward Optimization](./CHAPTER_09_OPTIMIZATION.md)*
